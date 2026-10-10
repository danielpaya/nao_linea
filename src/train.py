"""Entrena una politica PPO para NaoWalk-v0.

Uso (desde la raiz del repo):
    uv run python -m src.train --config configs/ppo.yaml
    uv run python -m src.train --config configs/ppo.yaml --total-timesteps 200000 --run-name prueba

Salida en runs/<run-name>/:
    config.yaml              configuracion efectiva (con los cambios de linea de comandos)
    metadata.yaml            version del codigo, librerias y fecha
    checkpoints/             checkpoints periodicos (modelo + normalizacion)
    best_model.zip           mejor modelo segun las evaluaciones periodicas
    best_vecnormalize.pkl    normalizacion del mejor modelo
    final_model.zip          modelo al terminar
    final_vecnormalize.pkl   normalizacion al terminar
    tb/                      logs de TensorBoard
"""

import argparse
import copy
import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path

import gymnasium
import mujoco
import stable_baselines3
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from src.utils.callbacks import ProgressCallback, RewardTermsCallback, SaveVecNormalizeCallback
from src.utils.env_factory import make_vec_env

ROOT = Path(__file__).resolve().parents[1]
ACTIVATIONS = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}


def load_config(path: Path, args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    overrides = {
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "checkpoint_every": args.checkpoint_every,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if args.eval_every is not None:
        cfg["eval"]["every"] = args.eval_every
    return cfg


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return out.stdout.strip() + ("-modificado" if dirty.strip() else "")
    except OSError:
        return "desconocido"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ppo.yaml")
    parser.add_argument("--run-name", default=None, help="nombre de la carpeta en runs/ (por defecto: experimento + fecha)")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args)
    run_name = args.run_name or f"{cfg['experiment']}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = ROOT / "runs" / run_name
    if run_dir.exists():
        print(f"ERROR: {run_dir} ya existe; usa otro --run-name.")
        return 1
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    metadata = {
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "mujoco": mujoco.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": str(torch.__version__),
        "comando": " ".join(sys.argv),
    }
    (run_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    seed, n_envs = cfg["seed"], cfg["n_envs"]
    torch.set_num_threads(cfg["torch_threads"])
    set_random_seed(seed)
    env_kwargs = copy.deepcopy(cfg["env"])
    norm = cfg["normalize"]

    venv = make_vec_env(n_envs, seed=seed, env_kwargs=env_kwargs)
    venv = VecNormalize(venv, norm_obs=norm["obs"], norm_reward=norm["reward"],
                        clip_obs=norm["clip_obs"], gamma=cfg["ppo"]["gamma"])
    # Entorno de evaluacion: semillas distintas, normalizacion sincronizada con la de entrenamiento.
    eval_env = make_vec_env(1, seed=seed + 10_000, env_kwargs=env_kwargs, subprocess=False)
    eval_env = VecNormalize(eval_env, norm_obs=norm["obs"], norm_reward=False,
                            clip_obs=norm["clip_obs"], training=False)

    pol = cfg["policy"]
    policy_kwargs = {
        "net_arch": {"pi": list(pol["net_arch"]), "vf": list(pol["net_arch"])},
        "activation_fn": ACTIVATIONS[pol["activation"]],
        "log_std_init": pol["log_std_init"],
    }
    model = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, tensorboard_log=str(run_dir / "tb"),
                seed=seed, device="cpu", verbose=0, **cfg["ppo"])

    dt_control = venv.get_attr("dt", indices=[0])[0]
    callbacks = CallbackList([
        ProgressCallback(cfg["total_timesteps"], dt_control),
        RewardTermsCallback(),
        CheckpointCallback(save_freq=max(cfg["checkpoint_every"] // n_envs, 1),
                           save_path=str(run_dir / "checkpoints"), name_prefix="model",
                           save_vecnormalize=True),
        EvalCallback(eval_env, n_eval_episodes=cfg["eval"]["n_episodes"],
                     eval_freq=max(cfg["eval"]["every"] // n_envs, 1), deterministic=True,
                     best_model_save_path=str(run_dir), log_path=str(run_dir / "eval"), verbose=0,
                     callback_on_new_best=SaveVecNormalizeCallback(run_dir / "best_vecnormalize.pkl")),
    ])

    print(f"Entrenando '{run_name}': {cfg['total_timesteps']:,} pasos, {n_envs} entornos, semilla {seed}")
    print(f"Resultados en: {run_dir.relative_to(ROOT)}")
    try:
        model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks, tb_log_name="PPO")
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario: guardando el modelo actual.")
    finally:
        model.save(run_dir / "final_model.zip")
        venv.save(str(run_dir / "final_vecnormalize.pkl"))
        venv.close()
        eval_env.close()
    print(f"Listo. Modelo final: {(run_dir / 'final_model.zip').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
