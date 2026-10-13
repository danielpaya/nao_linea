# -*- coding: utf-8 -*-
"""Entrena una generacion del NAO con PPO en la GPU (Brax + MJX).

Uso (desde ~/nao_rl, con el Python de Playground):
    ~/nao_rl/mujoco_playground/.venv/bin/python entrenar.py nao_v01
    ~/nao_rl/mujoco_playground/.venv/bin/python entrenar.py nao_v02 --pasos 50000000
    ~/nao_rl/mujoco_playground/.venv/bin/python entrenar.py nao_v01 --prueba      # corrida minima para ver que no falla

Cada generacion cambia UNA cosa respecto a la anterior (ver GENERACIONES). Todo queda en
runs/<generacion>_s<semilla>_<fecha>/:
    config.json      configuracion completa usada (entorno + PPO + aleatorizacion + versiones)
    progreso.csv     recompensa de evaluacion y de cada termino a lo largo del entrenamiento
    politica.pkl     pesos de la red (para verla, evaluarla y exportarla al NAO); se actualiza en cada evaluacion
    checkpoints/     copia de la politica en cada evaluacion (si algo falla de noche, no se pierde todo)
    estado.txt       'corriendo' o 'terminado' (para la cola nocturna)
"""
import argparse
import csv
import functools
import json
import os
import pickle
import platform
import sys
import time

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import nao_carrera as nc  # noqa: E402

ESCENA = os.path.join(AQUI, "nao_assets", "scene_mjx.xml")

# Cada generacion: hipotesis + cambios al entorno (acumulados) + aleatorizacion (None = mundo nominal)
RUMBO = {"rumbo.activo": True, "recompensa.escalas.seguir_rumbo": 1.0}
CADERA = {"recompensa.escalas.postura_cadera": -0.5}
MUNDO = dict(friccion=(0.3, 1.2), masa=(0.9, 1.1), kp=(0.8, 1.2))
GENERACIONES = {
    "nao_v01": dict(hipotesis="Base: mundo nominal. Debe caminar a 0.3 m/s pero ser fragil a cambios.",
                    entorno={}, aleatorizar=None),
    "nao_v01b": dict(hipotesis="Corrige v01 (gira ~40 grados al arrancar): la politica ve el error de rumbo y se "
                               "premia mantenerlo. Debe arrancar y caminar derecho.",
                     entorno=dict(RUMBO), aleatorizar=None),
    "nao_v01c": dict(hipotesis="Corrige v01 (piernas torcidas): castigo a HipYawPitch/HipRoll lejos de la postura. "
                               "Debe caminar con las piernas rectas.",
                     entorno=dict(RUMBO, **CADERA), aleatorizar=None),
    "nao_v02": dict(hipotesis="+ friccion, masas y kp al azar: generaliza a pisos y motores distintos.",
                    entorno=dict(RUMBO, **CADERA), aleatorizar=dict(MUNDO)),
    "nao_v03": dict(hipotesis="+ motor debil y retraso de 0-2 ciclos: aguanta fallas que v02 no aguanta.",
                    entorno=dict(RUMBO, **CADERA, latencia_max=2), aleatorizar=dict(MUNDO, motor_debil=(0.5, 1.0))),
    "nao_v04": dict(hipotesis="+ empujones: se recupera de golpes.",
                    entorno=dict(RUMBO, **CADERA, latencia_max=2, **{"empujones.activos": True}),
                    aleatorizar=dict(MUNDO, motor_debil=(0.5, 1.0))),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generacion", choices=sorted(GENERACIONES))
    ap.add_argument("--pasos", type=int, default=None, help="pasos de entrenamiento (por defecto 100 M)")
    ap.add_argument("--envs", type=int, default=None, help="robots en paralelo (por defecto 4096)")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--prueba", action="store_true", help="corrida minima para comprobar que todo funciona")
    a = ap.parse_args()

    gen = GENERACIONES[a.generacion]
    cfg_env = nc.default_config()
    cfg_ppo = nc.default_config_entrenamiento()
    if a.pasos:
        cfg_ppo.num_timesteps = a.pasos
    if a.envs:
        cfg_ppo.num_envs = a.envs
    if a.prueba:
        cfg_ppo.update(num_timesteps=20_000, num_evals=2, num_envs=32, batch_size=32, num_minibatches=4,
                       unroll_length=10)
        cfg_env.episode_length = 100

    env = nc.NaoCarrera(ESCENA, cfg_env, config_overrides=gen["entorno"])
    eval_env = nc.NaoCarrera(ESCENA, cfg_env, config_overrides=gen["entorno"])
    m = env.mj_model

    aleat = None
    if gen["aleatorizar"] is not None:
        ids_pierna = [m.actuator(n).id for n in nc.MOTORES_PIERNA]
        aleat = functools.partial(nc.aleatorizar, ids_pie=np.array([m.geom(n).id for n in nc.PIES]),
                                  id_piso=m.geom("floor").id, ids_act_pierna=ids_pierna, **gen["aleatorizar"])

    carpeta = os.path.join(AQUI, "runs", "%s_s%d_%s" % (a.generacion, a.semilla, time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(os.path.join(carpeta, "checkpoints"), exist_ok=True)
    open(os.path.join(carpeta, "estado.txt"), "w").write("corriendo\n")
    info = {
        "generacion": a.generacion, "hipotesis": gen["hipotesis"], "semilla": a.semilla,
        "entorno": env._config.to_dict(), "ppo": cfg_ppo.to_dict(), "aleatorizar": gen["aleatorizar"],
        "escena": ESCENA, "maquina": platform.node(), "jax": jax.__version__, "dispositivo": str(jax.devices()),
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(info, open(os.path.join(carpeta, "config.json"), "w"), indent=1, default=str)
    print("Generacion %s: %s" % (a.generacion, gen["hipotesis"]))
    print("Resultados en", carpeta)

    red = functools.partial(ppo_networks.make_ppo_networks,
                            policy_hidden_layer_sizes=cfg_ppo.politica, value_hidden_layer_sizes=cfg_ppo.critico,
                            policy_obs_key="state", value_obs_key="privileged_state")
    params_ppo = {k: v for k, v in cfg_ppo.to_dict().items() if k not in ("politica", "critico")}

    filas, t0 = [], [time.time()]

    def progreso(paso, metricas):
        fila = {"paso": int(paso), "minutos": round((time.time() - t0[0]) / 60, 2)}
        fila.update({k: float(v) for k, v in metricas.items() if k.startswith("eval/")})
        filas.append(fila)
        print("%11d pasos | %5.1f min | recompensa %7.3f | duracion episodio %5.0f"
              % (paso, fila["minutos"], fila.get("eval/episode_reward", float("nan")),
                 fila.get("eval/avg_episode_length", float("nan"))))
        with open(os.path.join(carpeta, "progreso.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in filas for k in r}))
            w.writeheader()
            w.writerows(filas)

    def guardar(paso, _make_policy, params):
        """Se llama en cada evaluacion: guarda la politica de ese momento."""
        params = jax.device_get(params)
        with open(os.path.join(carpeta, "checkpoints", "politica_%04dM.pkl" % (paso // 1_000_000)), "wb") as f:
            pickle.dump(params, f)
        with open(os.path.join(carpeta, "politica.pkl"), "wb") as f:
            pickle.dump(params, f)

    entrenar = functools.partial(ppo.train, policy_params_fn=guardar, **params_ppo, network_factory=red,
                                 randomization_fn=aleat,
                                 wrap_env_fn=wrapper.wrap_for_brax_training, seed=a.semilla,
                                 episode_length=cfg_env.episode_length)
    _, params, _ = entrenar(environment=env, eval_env=eval_env, progress_fn=progreso)
    with open(os.path.join(carpeta, "politica.pkl"), "wb") as f:
        pickle.dump(jax.device_get(params), f)
    open(os.path.join(carpeta, "estado.txt"), "w").write("terminado\n")
    print("Listo en %.1f min. Politica guardada en %s" % ((time.time() - t0[0]) / 60, carpeta))


if __name__ == "__main__":
    main()
