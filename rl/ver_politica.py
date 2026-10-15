# -*- coding: utf-8 -*-
"""Ver y medir una politica entrenada del NAO.

Uso (desde ~/nao_rl):
    PY=~/nao_rl/mujoco_playground/.venv/bin/python
    $PY ver_politica.py runs/nao_v01_XXXX                    # mide 5 carreras de 5 m y guarda un video
    $PY ver_politica.py runs/nao_v01_XXXX --vivo             # ademas abre el visor de MuJoCo (ventana en Windows)
    $PY ver_politica.py runs/nao_v01_XXXX --vx 0.2 --giro 0.2

Que mide (con el comando fijo, como en la carrera): si se cae, cuanto avanza, a que velocidad,
cuanto se desvia de lado y cuanto gira su rumbo sin que nadie lo corrija (sin vision).
"""
import argparse
import functools
import json
import math
import os
import pickle
import sys
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import nao_carrera as nc  # noqa: E402


def cargar(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    params = pickle.load(open(os.path.join(run, "politica.pkl"), "rb"))
    from ml_collections import config_dict
    cfg_env = nc.default_config()
    cfg_env.update(config_dict.ConfigDict(cfg["entorno"]))    # la MISMA configuracion del entrenamiento
    escena = cfg["escena"] if os.path.exists(cfg["escena"]) else os.path.join(AQUI, "nao_assets", "scene_mjx.xml")
    env = nc.NaoCarrera(escena, cfg_env)
    redes = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(cfg["ppo"]["politica"]), value_hidden_layer_sizes=tuple(cfg["ppo"]["critico"]),
        policy_obs_key="state", value_obs_key="privileged_state")
    politica = ppo_networks.make_inference_fn(redes)(params, deterministic=True)
    return env, jax.jit(politica), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="carpeta runs/<generacion>_<fecha>")
    ap.add_argument("--vx", type=float, default=None, help="por defecto, la velocidad maxima de la generacion")
    ap.add_argument("--giro", type=float, default=0.0)
    ap.add_argument("--carreras", type=int, default=5)
    ap.add_argument("--video", default=None, help="mp4 de salida (por defecto <run>/carrera.mp4); 'no' para omitir")
    ap.add_argument("--vivo", action="store_true", help="abrir el visor de MuJoCo")
    a = ap.parse_args()

    env, politica, cfg = cargar(a.run)
    print("Generacion %s: %s" % (cfg["generacion"], cfg["hipotesis"]))
    reset, paso = jax.jit(env.reset), jax.jit(env.step)
    if a.vx is None:
        a.vx = float(env._config.vx[1])
    cmd = jp.array([a.vx, a.giro])
    m = env.mj_model
    d = mujoco.MjData(m)
    torso = m.body("torso").id
    cuadros, viewer = [], None
    if a.vivo:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)
    renderer = None
    if a.video != "no":
        try:
            renderer = mujoco.Renderer(m, 368, 640)
        except Exception as exc:  # noqa: BLE001
            print("(sin video: %s)" % exc)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso
    cam.distance, cam.azimuth, cam.elevation = 1.4, 135, -20

    filas = []
    for i in range(a.carreras):
        st = reset(jax.random.PRNGKey(1000 + i))
        # rumbo inicial 0 (mirando a lo largo de la pista) y comando fijo, como en la carrera
        q = st.data.qpos.at[3:7].set(jp.array([1.0, 0, 0, 0]))
        st = st.replace(data=st.data.replace(qpos=q))
        st.info["comando"] = cmd
        st.info.update(nc.salida_recta(st.info))   # carril = eje x de la pista, robot centrado y derecho
        key = jax.random.PRNGKey(i)
        t, resultado, x0 = 0.0, "tiempo", float(st.data.qpos[0])
        rumbo_max = desvio_max = rumbo_2s = 0.0
        while t < 40.0:
            key, k = jax.random.split(key)
            accion, _ = politica(st.obs, k)
            st = paso(st, accion)
            st.info["comando"] = cmd
            st.info["paso"] = jp.zeros((), jp.int32)                 # que no cambie el comando
            t += env.dt
            qp = np.asarray(st.data.qpos)
            R = np.asarray(st.data.xmat[torso])
            rumbo = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            rumbo_max = max(rumbo_max, abs(rumbo))
            if t <= 2.0:
                rumbo_2s = rumbo                                          # cuanto giro al arrancar
            desvio_max = max(desvio_max, abs(qp[1]))
            if i == 0 and (renderer is not None or viewer is not None):
                d.qpos[:] = qp
                mujoco.mj_forward(m, d)
                if viewer is not None:
                    viewer.sync()
                    time.sleep(env.dt)
                if renderer is not None and len(cuadros) < 40 / env.dt:
                    renderer.update_scene(d, cam)
                    cuadros.append(renderer.render().copy())
            if bool(st.done):
                resultado = "CAIDA"
                break
            if qp[0] - x0 >= 5.0:
                resultado = "META"
                break
        avance = float(st.data.qpos[0]) - x0
        filas.append((resultado, t, avance, avance / t, 100 * desvio_max, rumbo_max, rumbo_2s, rumbo))
        print("carrera %d: %-6s t=%5.1f s  avance=%.2f m  vel=%.3f m/s  desvio lateral max=%5.1f cm  rumbo max=%5.1f deg"
              "  (a los 2 s: %+5.1f, al final: %+5.1f)" % ((i + 1,) + filas[-1]))
    metas = [f for f in filas if f[0] == "META"]
    print("\nMetas %d/%d | caidas %d | vel media %.3f m/s | desvio max medio %.1f cm | rumbo max medio %.1f deg (sin vision)"
          % (len(metas), len(filas), sum(f[0] == "CAIDA" for f in filas), np.mean([f[3] for f in filas]),
             np.mean([f[4] for f in filas]), np.mean([f[5] for f in filas])))
    if cuadros:
        import imageio
        salida = a.video or os.path.join(a.run, "carrera.mp4")
        imageio.mimsave(salida, cuadros[::2], fps=int(round(0.5 / env.dt)))
        print("Video:", salida)
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
