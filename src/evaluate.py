"""Evalua una politica entrenada en NaoWalk-v0: metricas en varios episodios, demo en vivo o video.

Uso (desde la raiz del repo):
    uv run python -m src.evaluate --model checkpoints/best_model.zip --render
    uv run python -m src.evaluate --model runs/<run>/best_model.zip --episodes 20 --json results/metrics.json
    uv run python -m src.evaluate --model runs/<run>/best_model.zip --video media/demo.mp4 --video-episodes 1

Junto al modelo se buscan automaticamente:
    - la normalizacion de observaciones (VecNormalize .pkl) con el nombre correspondiente al modelo
    - la configuracion del entorno (config.yaml de la carpeta del run, o <modelo>_config.yaml)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import src.envs  # noqa: F401  (registra NaoWalk-v0)
from src.envs import MAX_EPISODE_STEPS
from src.utils.video import annotate, write_video

ROOT = Path(__file__).resolve().parents[1]
ENV_ID = "NaoWalk-v0"


def find_vecnormalize(model_path: Path) -> Path | None:
    """Normalizacion asociada a un modelo, segun las convenciones de train.py."""
    stem, folder = model_path.stem, model_path.parent
    candidates = [folder / f"{stem}_vecnormalize.pkl"]
    if stem in ("best_model", "final_model"):
        candidates.append(folder / f"{stem.split('_')[0]}_vecnormalize.pkl")
    match = re.fullmatch(r"model_(\d+)_steps", stem)
    if match:
        candidates.append(folder / f"model_vecnormalize_{match.group(1)}_steps.pkl")
    return next((c for c in candidates if c.exists()), None)


def find_config(model_path: Path) -> Path | None:
    folder = model_path.parent
    candidates = [folder / f"{model_path.stem}_config.yaml", folder / "config.yaml", folder.parent / "config.yaml"]
    return next((c for c in candidates if c.exists()), None)


def rel(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def run_episode(env, model, vecnorm, seed: int, deterministic: bool, frames: list | None) -> dict:
    base = env.unwrapped
    obs, _ = env.reset(seed=seed)
    x0, y0 = base.data.qpos[0], base.data.qpos[1]
    ret, steps, max_tilt, effort, reason = 0.0, 0, 0.0, 0.0, None
    double_support, foot_max = 0, np.zeros(2)  # calidad de marcha
    terminated = truncated = False
    while not (terminated or truncated):
        norm_obs = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
        action, _ = model.predict(norm_obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += reward
        steps += 1
        max_tilt = max(max_tilt, base.torso_tilt_deg())
        effort += float(np.sum(base.data.actuator_force[base._act_ids] ** 2))
        double_support += int(base.foot_contacts().all())
        feet_z = [base.data.site(f).xpos[2] for f in ("left_foot", "right_foot")]
        foot_max = np.maximum(foot_max, feet_z)
        reason = info["fall_reason"]
        if frames is not None:
            t = steps * base.dt
            status = f"CAYO ({reason})" if terminated else "de pie"
            frames.append(annotate(env.render(), [
                f"politica entrenada | t = {t:4.1f} s | {status}",
                f"avance {base.data.qpos[0] - x0:+.2f} m | velocidad media {(base.data.qpos[0] - x0) / t:+.3f} m/s",
            ]))
    duration = steps * base.dt
    distance = float(base.data.qpos[0] - x0)
    return {
        "semilla": seed,
        "duracion_s": round(duration, 3),
        "distancia_m": round(distance, 4),
        "velocidad_media_ms": round(distance / duration, 4),
        "desviacion_lateral_m": round(float(base.data.qpos[1] - y0), 4),
        "cayo": bool(terminated),
        "motivo_caida": reason,
        "retorno": round(float(ret), 2),
        "inclinacion_max_deg": round(max_tilt, 2),
        "esfuerzo_medio": round(effort / steps, 4),  # media por paso de sum(tau^2) [N^2 m^2]
        "doble_apoyo_pct": round(100 * double_support / steps, 1),  # % del tiempo con ambos pies en el suelo
        "altura_pie_izq_cm": round(100 * float(foot_max[0]), 2),  # altura maxima de la suela al levantarlo
        "altura_pie_der_cm": round(100 * float(foot_max[1]), 2),
    }


def summarize(episodes: list[dict]) -> dict:
    def stats(key):
        values = np.array([e[key] for e in episodes], dtype=float)
        return {"media": round(float(values.mean()), 4), "desv": round(float(values.std()), 4),
                "min": round(float(values.min()), 4), "max": round(float(values.max()), 4)}

    keys = ("distancia_m", "velocidad_media_ms", "duracion_s", "retorno", "desviacion_lateral_m",
            "inclinacion_max_deg", "esfuerzo_medio", "doble_apoyo_pct", "altura_pie_izq_cm", "altura_pie_der_cm")
    summary = {key: stats(key) for key in keys}
    summary["tasa_caidas"] = round(sum(e["cayo"] for e in episodes) / len(episodes), 4)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="archivo .zip del modelo")
    parser.add_argument("--vecnormalize", type=Path, default=None, help="por defecto se busca junto al modelo")
    parser.add_argument("--config", type=Path, default=None, help="por defecto se busca junto al modelo")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000, help="semilla del primer episodio (luego +1, +2...)")
    parser.add_argument("--stochastic", action="store_true", help="acciones muestreadas (por defecto deterministas)")
    parser.add_argument("--render", action="store_true", help="ver en vivo en el visor 3D")
    parser.add_argument("--video", type=Path, default=None, help="grabar .mp4 o .gif")
    parser.add_argument("--video-episodes", type=int, default=1, help="episodios a incluir en el video")
    parser.add_argument("--json", type=Path, default=None, help="guardar metricas en JSON")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"ERROR: no existe {args.model}")
        return 1
    config_path = args.config or find_config(args.model)
    env_kwargs = yaml.safe_load(config_path.read_text(encoding="utf-8"))["env"] if config_path else {}
    vecnorm_path = args.vecnormalize or find_vecnormalize(args.model)

    render_mode = "human" if args.render else ("rgb_array" if args.video else None)
    env = gym.make(ENV_ID, render_mode=render_mode, **env_kwargs)
    vecnorm = None
    if vecnorm_path is not None:
        vecnorm = VecNormalize.load(str(vecnorm_path), DummyVecEnv([lambda: gym.make(ENV_ID, **env_kwargs)]))
        vecnorm.training = False
        vecnorm.norm_reward = False
    model = PPO.load(args.model, device="cpu")

    print(f"Modelo:        {rel(args.model)}")
    print(f"Normalizacion: {rel(vecnorm_path) if vecnorm_path else 'NO ENCONTRADA (se usan observaciones crudas)'}")
    print(f"Config. env:   {rel(config_path) if config_path else 'valores por defecto'}")
    print(f"Episodios: {args.episodes} (semillas {args.seed}..{args.seed + args.episodes - 1}), "
          f"acciones {'estocasticas' if args.stochastic else 'deterministas'}, "
          f"maximo {MAX_EPISODE_STEPS * env.unwrapped.dt:.0f} s\n")

    frames = [] if args.video else None
    episodes = []
    print(f"  {'ep':>3}{'duracion':>10}{'distancia':>11}{'velocidad':>11}{'lateral':>9}{'retorno':>9}  resultado")
    for i in range(args.episodes):
        record = frames is not None and i < args.video_episodes
        ep = run_episode(env, model, vecnorm, args.seed + i, not args.stochastic, frames if record else None)
        episodes.append(ep)
        result = f"cayo ({ep['motivo_caida']})" if ep["cayo"] else "sin caer"
        print(f"  {i + 1:>3}{ep['duracion_s']:>9.1f}s{ep['distancia_m']:>+10.2f}m{ep['velocidad_media_ms']:>+9.3f}m/s"
              f"{ep['desviacion_lateral_m']:>+8.2f}m{ep['retorno']:>9.1f}  {result}")
        if record:
            frames.extend([frames[-1]] * int(0.6 * env.unwrapped.metadata["render_fps"]))

    summary = summarize(episodes)
    print("\nResumen (media +- desviacion estandar):")
    for key, label in (("distancia_m", "distancia recorrida [m]"), ("velocidad_media_ms", "velocidad media [m/s]"),
                       ("duracion_s", "duracion del episodio [s]"), ("retorno", "retorno"),
                       ("desviacion_lateral_m", "desviacion lateral [m]"),
                       ("inclinacion_max_deg", "inclinacion maxima [grados]"),
                       ("doble_apoyo_pct", "doble apoyo [% del tiempo]"),
                       ("altura_pie_izq_cm", "altura max. pie izquierdo [cm]"),
                       ("altura_pie_der_cm", "altura max. pie derecho [cm]")):
        print(f"  {label:<30}{summary[key]['media']:>+9.3f} +- {summary[key]['desv']:.3f}")
    print(f"  {'tasa de caidas':<30}{summary['tasa_caidas'] * 100:>8.0f} %")

    if args.video:
        out = write_video(frames, args.video, env.unwrapped.metadata["render_fps"])
        print(f"\nVideo: {rel(out)}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "modelo": rel(args.model),
            "normalizacion": rel(vecnorm_path) if vecnorm_path else None,
            "config_entorno": env_kwargs,
            "episodios": args.episodes,
            "semilla_inicial": args.seed,
            "deterministico": not args.stochastic,
            "resumen": summary,
            "por_episodio": episodes,
        }
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Metricas: {rel(args.json)}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
