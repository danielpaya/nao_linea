# -*- coding: utf-8 -*-
"""Benchmark de las 3 caminatas en el carril de 5 m: RL (la del equipo), tipo NAOqi y tipo B-Human.

Para cada caminata, con y sin vision, y con varios coeficientes de friccion, corre varias salidas (semillas
1000, 1001, ...) con un error de colocacion inicial aleatorio (el mismo para las 3 caminatas) y mide:
tiempo a la meta, avance, velocidad, desvio lateral, rumbo maximo, caidas, salidas del carril y esfuerzo.

La vision aqui es SIMULADA con el modelo de error medido (resultados/modelo_error.md): la medicion de
(d, psi) llega a 10 Hz con ruido gaussiano sigma_d = 1.6 cm y sigma_psi = 1.2 deg, y se suaviza con un filtro
de primer orden. Asi el benchmark es rapido y compara las CAMINATAS, no la camara (la camara completa ya se
probo en lazo cerrado con run_demo.py).

Friccion: el RL se entreno con mu = 1.0 (valor por defecto de MuJoCo) y SIN aleatorizarla. La duela real
probablemente es mas resbalosa, por eso se barre mu.

Uso:
    uv run python -m demo.caminatas.benchmark                       # completo (tarda ~15-30 min)
    uv run python -m demo.caminatas.benchmark --rapido              # 1 semilla, mu 0.6 y 1.0
    uv run python -m demo.caminatas.benchmark --caminatas naoqi bhuman --mu 1.0
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SIGMA_D, SIGMA_PSI = 0.016, math.radians(1.2)     # error de la vision (modelo medido)
K_PSI, K_D, W_MAX = 1.2, 2.0, 0.5                  # mismo control que run_demo: w = -(k_psi*psi + k_d*d)
TURN_GAIN, MAX_BIAS, GYRO_OBS = 0.215, 0.35, 5    # direccion del RL por sesgo del giroscopio (run_demo.py)
VX = {"naoqi": 0.12, "bhuman": 0.30}              # velocidad maxima de cada caminata clasica [m/s]
MAX_T = 90.0


class VisionSimulada(object):
    """(d, psi) verdaderos + ruido, a 10 Hz, filtrados. Devuelve la velocidad de giro deseada."""

    def __init__(self, rng, periodo=0.1, alfa=0.5):
        self.rng, self.periodo, self.alfa = rng, periodo, alfa
        self.t_ult = -1e9
        self.d = self.psi = None

    def giro(self, t, d_real, psi_real):
        if t - self.t_ult >= self.periodo:
            self.t_ult = t
            md = d_real + self.rng.normal(0, SIGMA_D)
            mp_ = psi_real + self.rng.normal(0, SIGMA_PSI)
            if self.d is None:
                self.d, self.psi = md, mp_
            else:
                self.d += self.alfa * (md - self.d)
                self.psi += self.alfa * (mp_ - self.psi)
        if self.d is None:
            return 0.0
        return float(np.clip(-(K_PSI * self.psi + K_D * self.d), -W_MAX, W_MAX))


def inicio(seed):
    """Error de colocacion en la salida (igual para todas las caminatas con la misma semilla)."""
    rng = np.random.RandomState(seed)
    return float(rng.normal(0, 0.02)), float(rng.normal(0, math.radians(2.0)))


# ---------------------------------------------------------------------------- caminatas clasicas
def correr_clasica(nombre, mu, seed, vision, grabar=False):
    from demo.caminatas.simular import correr
    d0, psi0 = inicio(seed)
    rng = np.random.RandomState(seed + 7)
    vis = VisionSimulada(rng) if vision else None
    vx = VX[nombre]

    def mando(sim):
        x, y, yaw = sim.pose()
        return vx, 0.0, vis.giro(sim.t, y, yaw)

    info = correr(nombre, mu, vx, d0, psi0, MAX_T, vision=mando if vision else None, log=None, grabar=grabar)
    return info


# ---------------------------------------------------------------------------- caminata RL
_RL = {}


def correr_rl(mu, seed, vision, grabar=False):
    import pickle

    import mujoco
    import yaml
    from stable_baselines3 import PPO
    import src.envs.nao_walk_env as E
    from demo.caminatas.simular import FOOT_HALF_WIDTH, LANE_HALF, LENGTH
    E.SCENE_PATH = ROOT / "assets" / "nao" / "scene_carril.xml"
    if not _RL:
        import torch
        torch.set_num_threads(1)
        cfg = yaml.safe_load(open(ROOT / "checkpoints/best_model_config.yaml"))
        _RL["model"] = PPO.load(ROOT / "checkpoints/best_model.zip", device="cpu")
        _RL["vn"] = pickle.load(open(ROOT / "checkpoints/best_model_vecnormalize.pkl", "rb"))
        _RL["env"] = E.NaoWalkEnv(**cfg["env"])
    model, vn, env = _RL["model"], _RL["vn"], _RL["env"]
    obs, _ = env.reset(seed=seed)
    m, d = env.model, env.data
    m.geom_friction[:, 0] = mu
    d0, psi0 = inicio(seed)
    d.qpos[1] = d0
    d.qpos[3:7] = [math.cos(psi0 / 2), 0, 0, math.sin(psi0 / 2)]
    mujoco.mj_forward(m, d)
    for _ in range(int(1.0 / env.dt)):          # de pie 1 s (como la postura inicial de las otras)
        d.ctrl[:] = env._home_ctrl
        mujoco.mj_step(m, d, nstep=env.frame_skip)
    env._step_count = 0
    env._prev_action[:] = 0.0
    obs = env._get_obs()
    vis = VisionSimulada(np.random.RandomState(seed + 7)) if vision else None
    torso = env._torso_id
    hist, resultado, torque2, qs = [], "tiempo agotado", 0.0, []
    for k in range(int(MAX_T / env.dt)):
        t = k * env.dt
        pl, pr = d.site("left_foot").xpos, d.site("right_foot").xpos
        R = d.xmat[torso].reshape(3, 3)
        x, y, yaw = 0.5 * (pl[0] + pr[0]), 0.5 * (pl[1] + pr[1]), math.atan2(R[1, 0], R[0, 0])
        o = obs.copy()
        if vis is not None:
            o[GYRO_OBS] += float(np.clip(-vis.giro(t, y, yaw) / TURN_GAIN, -MAX_BIAS, MAX_BIAS))
        action, _ = model.predict(vn.normalize_obs(o), deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        torque2 += float(np.sum(d.actuator_force[12:] ** 2)) * env.dt
        pl, pr = d.site("left_foot").xpos, d.site("right_foot").xpos
        R = d.xmat[torso].reshape(3, 3)
        x, y, yaw = 0.5 * (pl[0] + pr[0]), 0.5 * (pl[1] + pr[1]), math.atan2(R[1, 0], R[0, 0])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
        hist.append((t + env.dt, x, y, yaw, tilt))
        if grabar:
            qs.append(d.qpos.copy())
        if terminated or d.xpos[torso][2] < 0.2 or tilt > 45:
            resultado = "CAIDA"
            break
        if any(abs(d.site(f).xpos[1]) + FOOT_HALF_WIDTH > LANE_HALF for f in ("left_foot", "right_foot")):
            resultado = "SALIO DEL CARRIL"
            break
        if max(pl[0], pr[0]) >= LENGTH:
            resultado = "META"
            break
    h = np.array(hist)
    T = h[-1, 0]
    extra = {"qpos": np.array(qs)} if grabar else {}
    return {**extra, "caminata": "rl", "mu": mu, "resultado": resultado, "t": T, "avance": h[-1, 1],
            "vel_media": h[-1, 1] / T, "desvio_max": float(np.max(np.abs(h[:, 2]))), "desvio_final": float(h[-1, 2]),
            "rumbo_max_deg": float(np.degrees(np.max(np.abs(h[:, 3])))), "inclinacion_max": float(np.max(h[:, 4])),
            "torque2_por_m": torque2 / max(0.05, h[-1, 1]), "pasos": None, "hist": h}


# ---------------------------------------------------------------------------- orquestacion
def una(args):
    caminata, mu, seed, vision = args
    t0 = time.time()
    if caminata == "rl":
        info = correr_rl(mu, seed, vision)
    else:
        info = correr_clasica(caminata, mu, seed, vision)
    h = info.pop("hist")
    info.update(semilla=seed, vision=int(vision), seg_reales=round(time.time() - t0, 1))
    info["traza"] = h[::25, :3].round(4).tolist()     # x, y cada ~0.3-0.5 s para las figuras
    return info


def info_corrida():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:  # noqa: BLE001
        commit = "?"
    import mujoco
    return {"fecha": time.strftime("%Y-%m-%d %H:%M"), "commit": commit, "maquina": platform.node(),
            "python": platform.python_version(), "mujoco": mujoco.__version__,
            "vision_simulada": {"sigma_d_m": SIGMA_D, "sigma_psi_deg": 1.2, "hz": 10},
            "control": {"k_psi": K_PSI, "k_d": K_D, "w_max": W_MAX}, "vx_clasicas": VX, "max_t": MAX_T,
            "friccion_entrenamiento_rl": 1.0, "rl_friccion_aleatorizada": False}


def resumen(filas):
    """Tabla por caminata x vision x mu."""
    grupos = {}
    for f in filas:
        grupos.setdefault((f["caminata"], f["vision"], f["mu"]), []).append(f)
    out = []
    for (c, v, mu), fs in sorted(grupos.items()):
        metas = [f for f in fs if f["resultado"] == "META"]
        out.append({"caminata": c, "vision": v, "mu": mu, "n": len(fs), "metas": len(metas),
                    "caidas": sum(f["resultado"] == "CAIDA" for f in fs),
                    "salidas": sum(f["resultado"] == "SALIO DEL CARRIL" for f in fs),
                    "t_meta_medio": float(np.mean([f["t"] for f in metas])) if metas else float("nan"),
                    "t_meta_std": float(np.std([f["t"] for f in metas])) if metas else float("nan"),
                    "vel_media": float(np.mean([f["vel_media"] for f in fs])),
                    "avance_medio": float(np.mean([f["avance"] for f in fs])),
                    "desvio_max_medio_cm": 100 * float(np.mean([f["desvio_max"] for f in fs])),
                    "rumbo_max_medio_deg": float(np.mean([f["rumbo_max_deg"] for f in fs])),
                    "torque2_por_m": float(np.mean([f["torque2_por_m"] for f in fs]))})
    return out


def escribir_md(res, meta, ruta):
    L = ["# Benchmark de caminatas en el carril de 5 m", "",
         "Generado por `demo/caminatas/benchmark.py` el %s (commit %s, %s, MuJoCo %s)."
         % (meta["fecha"], meta["commit"], meta["maquina"], meta["mujoco"]), "",
         "- **RL**: politica del equipo (`checkpoints/best_model.zip`), entrenada con friccion **mu = 1.0 sin aleatorizar**.",
         "- **naoqi_like**: pasos planeados + control preview del ZMP (esquema de ALMotion), %.2f m/s pedidos." % VX["naoqi"],
         "- **bhuman_like**: esquema Walk2014 (pasos de 250 ms, FSR + giroscopio), %.2f m/s pedidos." % VX["bhuman"],
         "- **Vision**: simulada con el error medido (sigma_d 1.6 cm, sigma_psi 1.2 deg, 10 Hz).",
         "- Salida con error de colocacion aleatorio (sigma 2 cm y 2 deg), el mismo para las 3 caminatas.", "",
         "| Caminata | Vision | mu | Metas | Caidas | Salidas | Tiempo a meta [s] | Vel. media [m/s] | Desvio max [cm] | Rumbo max [deg] | Esfuerzo [N²m²s/m] |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        tm = "—" if r["metas"] == 0 else "%.1f ± %.1f" % (r["t_meta_medio"], r["t_meta_std"])
        L.append("| %s | %s | %.1f | %d/%d | %d | %d | %s | %.3f | %.1f | %.1f | %.0f |"
                 % (r["caminata"], "si" if r["vision"] else "no", r["mu"], r["metas"], r["n"], r["caidas"], r["salidas"],
                    tm, r["vel_media"], r["desvio_max_medio_cm"], r["rumbo_max_medio_deg"], r["torque2_por_m"]))
    open(ruta, "w", encoding="utf-8").write("\n".join(L) + "\n")


def figura(filas, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colores = {"rl": "#1a73e8", "naoqi": "#e8710a", "bhuman": "#0b8043"}
    nombres = {"rl": "RL (equipo)", "naoqi": "tipo NAOqi", "bhuman": "tipo B-Human"}
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for i, v in enumerate((0, 1)):
        a = ax[i]
        a.axhspan(0.35, 0.5, color="#f4c7c3")
        a.axhspan(-0.5, -0.35, color="#f4c7c3")
        for f in filas:
            if f["vision"] != v or f["mu"] != 1.0:
                continue
            tr = np.array(f["traza"])
            a.plot(tr[:, 1], tr[:, 2], color=colores[f["caminata"]], alpha=0.7, lw=1.2)
            mk = {"META": "o", "CAIDA": "x", "SALIO DEL CARRIL": "s"}.get(f["resultado"], ".")
            a.plot(tr[-1, 1], tr[-1, 2], mk, color=colores[f["caminata"]])
        a.set_ylim(-0.5, 0.5)
        a.set_ylabel("desvio lateral [m]")
        a.set_title(("CON vision" if v else "SIN vision") + " (mu = 1.0)  -  o meta, x caida, ■ salio")
        a.axvline(5.0, color="#c5221f", lw=2)
    for c in colores:
        ax[0].plot([], [], color=colores[c], label=nombres[c])
    ax[0].legend(loc="upper left", fontsize=9)
    ax[1].set_xlabel("avance [m]")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caminatas", nargs="+", default=["rl", "naoqi", "bhuman"])
    ap.add_argument("--mu", nargs="+", type=float, default=[0.4, 0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--vision", choices=("ambas", "si", "no"), default="ambas")
    ap.add_argument("--rapido", action="store_true", help="1 semilla, mu 0.6 y 1.0")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--out", default=str(ROOT / "resultados" / "benchmark"))
    a = ap.parse_args()
    if a.rapido:
        a.semillas, a.mu = 1, [0.6, 1.0]
    visiones = {"ambas": (False, True), "si": (True,), "no": (False,)}[a.vision]
    trabajos = [(c, mu, 1000 + s, v) for c in a.caminatas for v in visiones for mu in a.mu for s in range(a.semillas)]
    print("Corridas: %d (%d procesos)" % (len(trabajos), a.procesos))
    os.makedirs(a.out, exist_ok=True)
    filas = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procesos) as pool:
        for i, f in enumerate(pool.imap_unordered(una, trabajos)):
            filas.append(f)
            print("[%3d/%d %4.0f s] %-7s vision=%d mu=%.1f semilla=%d -> %-16s t=%5.1f avance=%.2f desvio max=%4.1f cm"
                  % (i + 1, len(trabajos), time.time() - t0, f["caminata"], f["vision"], f["mu"], f["semilla"],
                     f["resultado"], f["t"], f["avance"], 100 * f["desvio_max"]))
    filas.sort(key=lambda f: (f["caminata"], f["vision"], f["mu"], f["semilla"]))
    meta = info_corrida()
    campos = [k for k in filas[0] if k != "traza"]
    with open(os.path.join(a.out, "corridas.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    res = resumen(filas)
    json.dump({"meta": meta, "resumen": res, "corridas": filas}, open(os.path.join(a.out, "benchmark.json"), "w"), indent=1)
    escribir_md(res, meta, os.path.join(a.out, "benchmark.md"))
    try:
        figura(filas, os.path.join(a.out, "trayectorias.png"))
    except Exception as exc:  # noqa: BLE001
        print("sin figura:", exc)
    print("\nListo en %.0f s. Resultados en %s" % (time.time() - t0, a.out))
    print(open(os.path.join(a.out, "benchmark.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
