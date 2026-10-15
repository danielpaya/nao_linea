# -*- coding: utf-8 -*-
"""Evalua una politica en condiciones VISTAS y NO VISTAS en el entrenamiento (la brecha de generalizacion).

Uso (desde ~/nao_rl):
    PY=~/nao_rl/mujoco_playground/.venv/bin/python
    $PY evaluar.py runs/nao_v01c_s0_XXXX                 # 8 carreras de 5 m por condicion
    $PY evaluar.py runs/nao_v01c_s0_XXXX --carreras 16

Cada carrera: sale del reposo mirando a lo largo de la pista, comando fijo (la velocidad maxima de la generacion: 0.3 m/s hasta v05a; sin giro), SIN vision, hasta
llegar a 5 m, caerse o 40 s. Todas las carreras de una condicion corren a la vez en la GPU.

Condiciones (una cosa distinta del mundo nominal cada vez):
    friccion 0.2 / 0.5 / 1.5      piso y pies (MuJoCo usa el maximo de los dos)
    kp 70 %                       todos los motores mas blandos
    motor 70 % / 40 %             UN motor de pierna (distinto en cada carrera) con menos torque maximo
    retraso 1 / 3 ciclos          la accion se aplica 24 / 72 ms tarde
    empujon 0.25 / 0.45 m/s       golpe horizontal en direccion al azar cada 3 s
Una condicion es "vista" si cae dentro de lo que esa generacion vio al entrenar (se lee de su config.json).

Guarda <run>/evaluacion.csv y lo imprime.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jp
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import nao_carrera as nc  # noqa: E402

CONDICIONES = [  # nombre, cambios respecto al mundo nominal
    ("nominal", {}),
    ("friccion_0.5", {"friccion": 0.5}),
    ("friccion_0.2", {"friccion": 0.2}),
    ("friccion_1.5", {"friccion": 1.5}),
    ("kp_70", {"kp": 0.7}),
    ("motor_70", {"motor": 0.7}),
    ("motor_40", {"motor": 0.4}),
    ("retraso_1", {"retraso": 1}),
    ("retraso_3", {"retraso": 3}),
    ("empujon_0.25", {"empujon": 0.25}),
    ("empujon_0.45", {"empujon": 0.45}),
]
LARGO, MITAD_CARRIL, MITAD_PIE = 5.0, 0.35, 0.045


def cargar(run):
    """Entorno con la misma configuracion del entrenamiento (con espacio para hasta 3 ciclos de retraso)."""
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from ml_collections import config_dict
    import pickle
    cfg = json.load(open(os.path.join(run, "config.json")))
    cfg_env = nc.default_config()
    cfg_env.update(config_dict.ConfigDict(cfg["entorno"]))
    cfg_env.latencia_max = max(3, cfg_env.latencia_max)
    cfg_env.empujones.activos = False          # los empujones de la evaluacion se aplican aqui, con magnitud fija
    escena = cfg["escena"] if os.path.exists(cfg["escena"]) else os.path.join(AQUI, "nao_assets", "scene_mjx.xml")
    env = nc.NaoCarrera(escena, cfg_env)
    params = pickle.load(open(os.path.join(run, "politica.pkl"), "rb"))
    redes = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(cfg["ppo"]["politica"]), value_hidden_layer_sizes=tuple(cfg["ppo"]["critico"]),
        policy_obs_key="state", value_obs_key="privileged_state")
    politica = ppo_networks.make_inference_fn(redes)(params, deterministic=True)
    return env, politica, cfg


def es_vista(cfg, cambios):
    """True si la condicion esta dentro de lo que la generacion vio al entrenar."""
    al = cfg.get("aleatorizar") or {}
    ent = cfg["entorno"]
    if not cambios:
        return True
    k, v = next(iter(cambios.items()))
    if k == "friccion":
        lo, hi = al.get("friccion", (1.0, 1.0))
        return lo <= v <= hi
    if k == "kp":
        lo, hi = al.get("kp", (1.0, 1.0))
        return lo <= v <= hi
    if k == "motor":
        md = al.get("motor_debil")
        return md is not None and md[0] <= v <= md[1]
    if k == "retraso":
        return v <= ent.get("latencia_max", 0)
    if k == "empujon":
        e = ent.get("empujones", {})
        return bool(e.get("activos")) and e["magnitud"][0] <= v <= e["magnitud"][1]
    return False


def hacer_evaluador(env, politica, n, pasos, vx=0.3):
    m0 = env.mjx_model
    mj = env.mj_model
    piso = mj.geom("floor").id
    pies = np.array([mj.geom(p).id for p in nc.PIES])
    sites = np.array([mj.site(p).id for p in nc.PIES])
    torso = mj.body("torso").id
    pierna = jp.array([mj.actuator(a).id for a in nc.MOTORES_PIERNA])
    cada_empujon = int(round(3.0 / env.dt))
    cmd = jp.array([vx, 0.0])

    def una(friccion, kp, motor_frac, motor_i, latencia, empujon, key):
        modelo = m0.tree_replace({
            "geom_friction": m0.geom_friction.at[piso, 0].set(friccion).at[pies, 0].set(friccion),
            "actuator_gainprm": m0.actuator_gainprm.at[:, 0].multiply(kp),
            "actuator_biasprm": m0.actuator_biasprm.at[:, 1].multiply(kp),
            "actuator_forcerange": m0.actuator_forcerange.at[pierna[motor_i]].multiply(motor_frac),
        })
        viejo = env._mjx_model
        env._mjx_model = modelo                 # el entorno usa este mundo mientras se traza la funcion
        try:
            k_reset, key = jax.random.split(key)
            st = env.reset(k_reset)
            q = st.data.qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))   # mirando a lo largo de la pista
            info = dict(st.info, comando=cmd, latencia=latencia)
            info.update(nc.salida_recta(info))     # carril = eje x de la pista, robot centrado y derecho
            st = st.replace(data=st.data.replace(qpos=q), info=info)
            x0, y0 = q[0], q[1]

            def paso(carry, k):
                st, key, fin, cayo, t_meta, avance, ymax, yawmax, fuera = carry
                key, k_pol, k_emp = jax.random.split(key, 3)
                # empujon de magnitud fija cada 3 s, direccion al azar
                ang = jax.random.uniform(k_emp, minval=0, maxval=2 * jp.pi)
                toca = (k % cada_empujon == cada_empujon - 1) & (empujon > 0)
                dv = jp.where(toca, empujon, 0.0) * jp.array([jp.cos(ang), jp.sin(ang)])
                st = st.replace(data=st.data.replace(qvel=st.data.qvel.at[0:2].add(dv)))
                accion, _ = politica(st.obs, k_pol)
                st = env.step(st, accion)
                st = st.replace(info=dict(st.info, comando=cmd, paso=jp.zeros_like(st.info["paso"])))
                d = st.data
                x = d.qpos[0] - x0
                yaw = jp.arctan2(d.xmat[torso][1, 0], d.xmat[torso][0, 0])
                pies_y = d.site_xpos[sites][:, 1] - y0
                t = (k + 1) * env.dt
                llega = (x >= LARGO) & ~fin
                cae = (st.done > 0) & ~fin
                vivo = ~fin
                t_meta = jp.where(llega, t, t_meta)
                cayo = cayo | cae
                avance = jp.where(vivo, x, avance)
                ymax = jp.where(vivo, jp.maximum(ymax, jp.abs(d.qpos[1] - y0)), ymax)
                yawmax = jp.where(vivo, jp.maximum(yawmax, jp.abs(yaw)), yawmax)
                fuera = fuera | (vivo & (jp.max(jp.abs(pies_y)) + MITAD_PIE > MITAD_CARRIL))
                fin = fin | llega | cae
                return (st, key, fin, cayo, t_meta, avance, ymax, yawmax, fuera), None

            c0 = (st, key, jp.array(False), jp.array(False), jp.array(jp.nan), jp.array(0.0), jp.array(0.0),
                  jp.array(0.0), jp.array(False))
            c, _ = jax.lax.scan(paso, c0, jp.arange(pasos))
            return {"cayo": c[3], "t_meta": c[4], "avance": c[5], "desvio_max": c[6], "rumbo_max": c[7],
                    "se_saldria": c[8]}
        finally:
            env._mjx_model = viejo

    lote = jax.vmap(una, in_axes=(None, None, None, 0, None, None, 0))
    return jax.jit(lote)


def evaluar_run(run, carreras=8, segundos=40.0, imprimir=True):
    t0 = time.time()
    env, politica, cfg = cargar(run)
    vx = float(env._config.vx[1])               # la velocidad maxima que aprendio (0.3 m/s hasta v05a)
    ev = hacer_evaluador(env, politica, carreras, int(round(segundos / env.dt)), vx=vx)
    filas = []
    for nombre, cambios in CONDICIONES:
        r = ev(jp.float32(cambios.get("friccion", 1.0)), jp.float32(cambios.get("kp", 1.0)),
               jp.float32(cambios.get("motor", 1.0)), jp.arange(carreras) % len(nc.MOTORES_PIERNA),
               jp.int32(cambios.get("retraso", 0)), jp.float32(cambios.get("empujon", 0.0)),
               jax.random.split(jax.random.PRNGKey(1000), carreras))
        r = {k: np.asarray(v) for k, v in r.items()}
        metas = np.isfinite(r["t_meta"])
        filas.append({
            "condicion": nombre, "grupo": "vista" if es_vista(cfg, cambios) else "no vista", "carreras": carreras,
            "metas": int(metas.sum()), "caidas": int(r["cayo"].sum()),
            "t_meta_medio": float(np.nanmean(r["t_meta"])) if metas.any() else float("nan"),
            "avance_medio": float(r["avance"].mean()),
            "desvio_max_cm": float(100 * r["desvio_max"].mean()),
            "rumbo_max_deg": float(np.degrees(r["rumbo_max"].mean())),
            "se_saldria_sin_vision": int(r["se_saldria"].sum()),
        })
    with open(os.path.join(run, "evaluacion.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    if imprimir:
        print("\n%s  (%s)  evaluado en %.1f min" % (os.path.basename(run.rstrip("/")), cfg["hipotesis"],
                                                     (time.time() - t0) / 60))
        print("%-14s %-9s %6s %6s %8s %9s %8s %9s" % ("condicion", "grupo", "metas", "caidas", "t meta",
                                                      "desvio cm", "rumbo", "se sale*"))
        for f_ in filas:
            print("%-14s %-9s %3d/%-2d %6d %8s %9.1f %8.1f %6d/%d" % (
                f_["condicion"], f_["grupo"], f_["metas"], carreras, f_["caidas"],
                "-" if math.isnan(f_["t_meta_medio"]) else "%.1f s" % f_["t_meta_medio"],
                f_["desvio_max_cm"], f_["rumbo_max_deg"], f_["se_saldria_sin_vision"], carreras))
        print("* se saldria del carril de 70 cm si no tuviera vision (la vision corrige esto despues)")
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="una o varias carpetas runs/...")
    ap.add_argument("--carreras", type=int, default=8)
    ap.add_argument("--segundos", type=float, default=40.0)
    a = ap.parse_args()
    for run in a.runs:
        evaluar_run(run, a.carreras, a.segundos)


if __name__ == "__main__":
    main()
