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
import glob
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
PIES = {"recompensa.sigma_pies": 0.0005}
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
    "nao_v01d": dict(hipotesis="Corrige v01b/v01c/v02 (arrastran los pies, tiempo_aire ~0): fase_pies con tolerancia "
                               "0.0005 en vez de 0.01, para que arrastrar el pie ya no de casi todo el premio. "
                               "Debe levantar los pies ~3 cm sin perder velocidad ni rumbo.",
                     entorno=dict(RUMBO, **CADERA, **PIES), aleatorizar=None),
    "nao_v01e": dict(hipotesis="v01d + los codos (LElbowRoll, RElbowRoll) en la accion: con los brazos mas utiles para "
                               "el equilibrio y para frenar el giro, debe ir mas recta (v01d se tuerce 8-10 grados) "
                               "sin perder robustez.",
                     entorno=dict(RUMBO, **CADERA, **PIES, codos=True), aleatorizar=None),
    "nao_v02r": dict(hipotesis="v01d + retraso de la accion de 0-2 ciclos (0-48 ms) al azar: deja de caerse con retraso "
                               "(v01d: 0/8 metas con 3 ciclos y 21 s en vez de 17 s con 1 ciclo).",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2), aleatorizar=None),
    "nao_v02d": dict(hipotesis="v01d + friccion, masas y kp al azar: generaliza aun mejor a pisos y motores distintos "
                               "(v01d ya llega 8/8 en esas condiciones pero se desvia mas).",
                     entorno=dict(RUMBO, **CADERA, **PIES), aleatorizar=dict(MUNDO)),
    # --- noche 2: base nao_v02r (v01d + retraso), la unica que aguanta 72 ms de retraso (bitacora [23]) ---
    "nao_v03c": dict(hipotesis="v02r + codos: suma lo recto de v01e (desvio 9.9 cm) a lo robusto de v02r.",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True), aleatorizar=None),
    "nao_v03m": dict(hipotesis="v02r + friccion, masas y kp al azar: menos desvio con piso resbaloso y motor debil "
                               "(v02r: friccion 0.2 -> 27.9 cm, motor 40 -> 37.8 cm).",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2), aleatorizar=dict(MUNDO)),
    "nao_v03e": dict(hipotesis="v02r + empujones (0.05-0.3 m/s cada 3-6 s): se recupera de golpes fuertes "
                               "(v02r: empujon 0.45 -> 1/8 metas).",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, **{"empujones.activos": True}),
                     aleatorizar=None),
    "nao_v04": dict(hipotesis="v02r + codos + mundo al azar + empujones (los tres cambios de v03c/v03m/v03e juntos, "
                              "medidos por separado en esas corridas). Candidata final de la etapa a 0.3 m/s.",
                    entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True, **{"empujones.activos": True}),
                    aleatorizar=dict(MUNDO)),
    # --- etapa de velocidad (2026-10-14): base nao_v04; cada una arranca desde la anterior con --desde ---
    "nao_v05a": dict(hipotesis="v04 + motores reales (el torque cae con la velocidad del motor, tope 4.5-7 rad/s; "
                               "brazos con su torque real): sigue caminando a 0.3 m/s con motores que el NAO si tiene.",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True, motores_reales=True,
                                  **{"empujones.activos": True}), aleatorizar=dict(MUNDO)),
    "nao_v05b": dict(hipotesis="v05a + comando hasta 0.4 m/s: 5 m en ~13 s sin perder robustez.",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True, motores_reales=True,
                                  vx=[0.0, 0.4], **{"empujones.activos": True}), aleatorizar=dict(MUNDO)),
    "nao_v05c": dict(hipotesis="v05b + comando hasta 0.5 m/s: 5 m en ~11 s sin perder robustez.",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True, motores_reales=True,
                                  vx=[0.0, 0.5], **{"empujones.activos": True}), aleatorizar=dict(MUNDO)),
    "nao_v05d": dict(hipotesis="v05c + comando hasta 0.6 m/s: mide el techo de velocidad con motores reales "
                               "(no es para la carrera: da margen para correr a 0.5).",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True, motores_reales=True,
                                  vx=[0.0, 0.6], **{"empujones.activos": True}), aleatorizar=dict(MUNDO)),
    # --- vision simulada (2026-10-14): base nao_v05c (0.5 m/s, motores reales) ---
    "nao_v06": dict(hipotesis="v05c + vision simulada (psi y d de la camara con ruido 1.6 cm / 1.2 grados, 10 Hz, "
                              "1 cuadro de retraso, 5 % de cuadros perdidos) en vez del rumbo del giroscopio, salida "
                              "chueca (+-10 cm, +-8 grados) y premio por ir por el centro (seguir_carril): se centra "
                              "sola y no se sale del carril, sin perder los ~11 s.",
                    entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True, motores_reales=True,
                                 vx=[0.0, 0.5], **{"empujones.activos": True, "vision.activa": True,
                                                   "recompensa.escalas.seguir_carril": 1.0}),
                    aleatorizar=dict(MUNDO)),
    # --- noche 5 (2026-10-14): base nao_v06 s0. validar_vision.py mostro que sin camara la marcha deriva y se sale
    # (v06 ya no ve el rumbo del giroscopio) y un sesgo de ~4-5 cm a la derecha (bitacora [44]) ---
    "nao_v07a": dict(hipotesis="v06 + apagones de camara (0.5-2 s, ~1 cada 8 s; la observacion se queda con el ultimo "
                               "cuadro) y la edad del ultimo cuadro en la observacion: aprende a no fiarse de un cuadro "
                               "viejo. Sin rumbo del giroscopio, se espera que mejore poco en apagones largos.",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True, motores_reales=True,
                                  vx=[0.0, 0.5], **{"empujones.activos": True, "vision.activa": True,
                                                    "recompensa.escalas.seguir_carril": 1.0,
                                                    "vision.apagones": True}),
                     aleatorizar=dict(MUNDO)),
    "nao_v07b": dict(hipotesis="v07a + el error de rumbo del giroscopio (con deriva) junto a la camara: durante un "
                               "apagon mantiene el rumbo con el giroscopio y no se sale del carril.",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True, motores_reales=True,
                                  vx=[0.0, 0.5], **{"empujones.activos": True, "vision.activa": True,
                                                    "recompensa.escalas.seguir_carril": 1.0,
                                                    "vision.apagones": True, "vision.giroscopio": True}),
                     aleatorizar=dict(MUNDO)),
    "nao_v07c": dict(hipotesis="v06 + seguir_carril mas estrecho (sigma 0.0025 m^2, ~5 cm en vez de ~10 cm): quita el "
                               "sesgo de ~4-5 cm a la derecha sin perder los ~11 s.",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True, motores_reales=True,
                                  vx=[0.0, 0.5], **{"empujones.activos": True, "vision.activa": True,
                                                    "recompensa.escalas.seguir_carril": 1.0,
                                                    "recompensa.sigma_carril": 0.0025}),
                     aleatorizar=dict(MUNDO)),
    # --- despliegue en el NAO real (2026-10-29): base nao_v07c s0 (--desde nao_v07c). Cierra la brecha simulacion-robot
    # que se encontro al revisar el NAO por dentro (docs de despliegue): no hay velocidad de articulaciones, el retraso
    # real es ~36-60 ms, la IMU tiene sesgo y el servo recibe objetivos de posicion. Cambios juntos por falta de tiempo
    # (no se separa el efecto de cada uno; si algo empeora, se corre una ablacion). ---
    "nao_v08": dict(hipotesis="v07c sin velocidades de articulacion en la observacion del actor (el NAO real no las "
                              "mide; la historia de 3 angulos le deja inferirlas), retraso de 0-3 ciclos (0-72 ms), IMU "
                              "con sesgo de giroscopio (+-3 grados/s) y giro de montaje (+-3 grados) y castigo mayor al "
                              "cambio de accion (-0.01 -> -0.03, mas suave para los motores): mantiene ~11 s y la "
                              "centrada de v07c con una observacion que el robot si puede dar y sin ser fragil a "
                              "retraso e IMU.",
                    entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=3, codos=True, motores_reales=True,
                                 vx=[0.0, 0.5], obs_velocidades=False,
                                 **{"empujones.activos": True, "vision.activa": True,
                                    "recompensa.escalas.seguir_carril": 1.0,
                                    "recompensa.sigma_carril": 0.0025,
                                    "recompensa.escalas.cambio_accion": -0.03,
                                    "imu.activa": True}),
                    aleatorizar=dict(MUNDO)),
    "nao_v06d": dict(hipotesis="v05d (0.6 m/s) + vision simulada y seguir_carril, igual que v06 sobre v05c: ~9.5 s y "
                               "centrada (plan B si el robot real aguanta 0.6 m/s).",
                     entorno=dict(RUMBO, **CADERA, **PIES, latencia_max=2, codos=True, motores_reales=True,
                                  vx=[0.0, 0.6], **{"empujones.activos": True, "vision.activa": True,
                                                    "recompensa.escalas.seguir_carril": 1.0}),
                     aleatorizar=dict(MUNDO)),
    "nao_v02": dict(hipotesis="+ friccion, masas y kp al azar: generaliza a pisos y motores distintos.",
                    entorno=dict(RUMBO, **CADERA), aleatorizar=dict(MUNDO)),
    "nao_v03": dict(hipotesis="+ motor debil y retraso de 0-2 ciclos: aguanta fallas que v02 no aguanta.",
                    entorno=dict(RUMBO, **CADERA, latencia_max=2), aleatorizar=dict(MUNDO, motor_debil=(0.5, 1.0))),
    # nunca se entreno: era la v04 de la linea vieja (base v01c). Renombrada el 2026-10-13 porque tapaba a la nao_v04
    # nueva (dos claves iguales en un dict: gana la ultima). Ver bitacora [25].
    "nao_v04_antigua": dict(hipotesis="+ empujones: se recupera de golpes.",
                    entorno=dict(RUMBO, **CADERA, latencia_max=2, **{"empujones.activos": True}),
                    aleatorizar=dict(MUNDO, motor_debil=(0.5, 1.0))),
}


def buscar_desde(desde):
    """Carpeta de la politica de partida: la ruta dada o la ultima corrida terminada de esa generacion."""
    if os.path.isdir(desde):
        return desde.rstrip("/")
    for ruta in (os.path.join(AQUI, desde), os.path.join(AQUI, "runs", desde)):
        if os.path.isdir(ruta):
            return ruta.rstrip("/")
    candidatas = []
    for d in glob.glob(os.path.join(AQUI, "runs", desde + "_s*_*")):
        try:
            if open(os.path.join(d, "estado.txt")).read().strip() == "terminado":
                candidatas.append(d)
        except OSError:
            pass
    if not candidatas:
        sys.exit("No encuentro una corrida terminada para --desde %s" % desde)
    return max(candidatas, key=os.path.getmtime)


def adaptar_desde(params, desde, env):
    """Si la politica de partida ve otra observacion (p. ej. v05c -> v06 con vision), trasplanta sus pesos."""
    from ml_collections import config_dict
    viejo = json.load(open(os.path.join(desde, "config.json")))
    cfg_vieja = nc.default_config()
    cfg_vieja.update(config_dict.ConfigDict(viejo["entorno"]))
    etq_vieja = nc.NaoCarrera(ESCENA, cfg_vieja).etiquetas()
    etq_nueva = env.etiquetas()
    if etq_vieja == etq_nueva:
        return params
    nuevas = [e for e in etq_nueva[1] if e not in etq_vieja[1]]
    print("Trasplante de pesos: %d -> %d entradas del actor, %d -> %d del critico; nuevas: %s"
          % (len(etq_vieja[0]), len(etq_nueva[0]), len(etq_vieja[1]), len(etq_nueva[1]), ", ".join(nuevas)))
    return nc.trasplantar(params, etq_vieja, etq_nueva)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generacion", choices=sorted(GENERACIONES))
    ap.add_argument("--pasos", type=int, default=None, help="pasos de entrenamiento (por defecto 100 M)")
    ap.add_argument("--envs", type=int, default=None, help="robots en paralelo (por defecto 4096)")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--prueba", action="store_true", help="corrida minima para comprobar que todo funciona")
    ap.add_argument("--desde", default=None,
                    help="arrancar desde una politica ya entrenada: carpeta runs/... o nombre de generacion "
                         "(usa su ultima corrida terminada). Debe tener la misma observacion y accion.")
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

    desde, params_desde = None, None
    if a.desde:
        desde = buscar_desde(a.desde)
        with open(os.path.join(desde, "politica.pkl"), "rb") as f:
            params_desde = pickle.load(f)              # (normalizador, politica, valor), como lo guarda guardar()
        print("Arranca desde", desde)

    env = nc.NaoCarrera(ESCENA, cfg_env, config_overrides=gen["entorno"])
    eval_env = nc.NaoCarrera(ESCENA, cfg_env, config_overrides=gen["entorno"])
    if params_desde is not None:
        params_desde = adaptar_desde(params_desde, desde, env)
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
        "escena": ESCENA, "desde": desde, "maquina": platform.node(), "jax": jax.__version__, "dispositivo": str(jax.devices()),
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
                                 episode_length=cfg_env.episode_length, restore_params=params_desde)
    _, params, _ = entrenar(environment=env, eval_env=eval_env, progress_fn=progreso)
    with open(os.path.join(carpeta, "politica.pkl"), "wb") as f:
        pickle.dump(jax.device_get(params), f)
    open(os.path.join(carpeta, "estado.txt"), "w").write("terminado\n")
    print("Listo en %.1f min. Politica guardada en %s" % ((time.time() - t0[0]) / 60, carpeta))


if __name__ == "__main__":
    main()
