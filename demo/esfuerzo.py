# -*- coding: utf-8 -*-
"""Analisis de esfuerzo de los motores con las politicas exportadas (rl/politicas/*/politica_np.npz).

Uso (desde C:\\demo_carril o en Linux, con el entorno de la demo):
    uv run python -m demo.esfuerzo                      # todas las politicas, 3 semillas cada una
    uv run python -m demo.esfuerzo --semillas 5
Salida: results/esfuerzo/esfuerzo.json (y un resumen en la terminal).

Como se mide: la misma simulacion de demo/rl_vivo.py (MuJoCo normal, mismo modelo y limites de motor que en el
entrenamiento, vision simulada con el ruido del entrenamiento, salida centrada, sin perturbaciones, friccion
nominal). En cada paso de fisica (2 ms) se registra el par de cada motor, la velocidad y posicion de cada
articulacion, el limite de par que tenia el motor simulado en ese instante y la fuerza de cada pie contra el piso,
desde la salida hasta la meta (5 m).

Datos de los motores reales (en la articulacion = motor x reductora; documentacion de motores de Aldebaran, tabla
compartida por el equipo el 2026-10-15; el PDF del V6 no se pudo leer):
    Tipo 5  HipPitch, KneePitch, AnklePitch     reductora 130.85  parada 8.51 N m  continuo 2.33 N m  6.96 rad/s
    Tipo 1  HipYawPitch, HipRoll, AnkleRoll     reductora 201.3   parada 11.97 N m continuo 2.47 N m  4.16 rad/s
    Brazos: solo se tiene el par de parada (hombro 2.15, codo 2.48 N m) y la velocidad (7.46 y 6.47 rad/s);
            sin par continuo no se calcula "calor".
OJO: el entrenamiento uso 4.53 rad/s para el tipo 1 (8700 rpm / 201.3); aqui se usa 4.16 rad/s de la tabla nueva.

Metricas por motor:
    calor            par RMS / par continuo (el calor crece con el cuadrado de la corriente ~ par)
    sobre_continuo   % del tiempo con |par| > par continuo
    tope_sim         % del tiempo con el par en el tope del motor simulado (la politica pide mas de lo que recibe)
    fuera_curva      % del tiempo con |par| > par_parada * (1 - |w| / w_max): imposible para el motor real (w al
                     empezar el paso de 2 ms, la misma con la que el motor simulado limita su par)
    vel_p99, vel_max velocidad de la articulacion (rad/s) vs w_max
    cerca_tope       % del tiempo a menos de 3 grados del limite de la articulacion
Globales: fuerza pico y p99 de cada pie / peso del robot, energia mecanica por metro (sum |par * w| dt / distancia),
separacion minima entre pies.
Limitaciones: el servo simulado es un control PD con tope de par (no el servo real); el par continuo es una cota
superior (no incluye la eficiencia de la reductora); no se simula temperatura.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco  # noqa: E402
from demo.rl_vivo import Carrera, cargar_politicas, medicion_simulada, modelo_fisica  # noqa: E402

TIPO5 = dict(tipo="Tipo 5", parada=8.51, continuo=2.33, w_max=6.96)
TIPO1 = dict(tipo="Tipo 1", parada=11.97, continuo=2.47, w_max=4.16)
MOTOR_REAL = {
    "LHipPitch": TIPO5, "RHipPitch": TIPO5, "LKneePitch": TIPO5, "RKneePitch": TIPO5,
    "LAnklePitch": TIPO5, "RAnklePitch": TIPO5,
    "LHipYawPitch": TIPO1, "LHipRoll": TIPO1, "RHipRoll": TIPO1, "LAnkleRoll": TIPO1, "RAnkleRoll": TIPO1,
    "LShoulderPitch": dict(tipo="brazo", parada=2.15, continuo=None, w_max=7.46),
    "RShoulderPitch": dict(tipo="brazo", parada=2.15, continuo=None, w_max=7.46),
    "LElbowRoll": dict(tipo="brazo", parada=2.48, continuo=None, w_max=6.47),
    "RElbowRoll": dict(tipo="brazo", parada=2.48, continuo=None, w_max=6.47),
}
CERCA_TOPE = math.radians(3.0)


class CarreraRegistrada(Carrera):
    """Carrera que guarda lo que pasa en cada paso de fisica de 2 ms."""

    def __init__(self, mf, pol):
        super().__init__(mf, pol)
        self.pies = [mf.geom(n).id for n in ("left_foot", "right_foot")]
        self.sitios = [mf.site(n).id for n in ("left_foot", "right_foot")]
        self.rango = mf.jnt_range[mf.actuator_trnid[self.act, 0]]
        self.registro = None

    def _fuerza_pies(self):
        f = np.zeros(2)
        c6 = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            for k, g in enumerate(self.pies):
                if con.geom1 == g or con.geom2 == g:
                    mujoco.mj_contactForce(self.m, self.d, i, c6)
                    f[k] += abs(c6[0])
        return f

    def _fisica(self):
        m, d = self.m, self.d
        for _ in range(self.nsub):
            lim = self.tope.copy()
            w = d.qvel[self.vadr].copy()                 # velocidad al empezar el paso (con la que se limita)
            if self.mt["motores_reales"]:
                lim = self.tope * np.clip(1.0 - np.abs(w) / self.vel_max, 0.0, 1.0)
                m.actuator_forcerange[self.act, 0] = -lim
                m.actuator_forcerange[self.act, 1] = lim
            mujoco.mj_step(m, d)
            if self.registro is not None:
                r = self.registro
                r["par"].append(d.actuator_force[self.act].copy())
                r["w"].append(d.qvel[self.vadr].copy())
                r["q"].append(d.qpos[self.qadr].copy())
                r["lim"].append(lim)
                r["w0"].append(w)
                r["pies"].append(self._fuerza_pies())
                p = d.site_xpos[self.sitios]
                r["sep"].append(abs(p[0, 1] - p[1, 1]))
        m.actuator_forcerange[:] = self.base_fr


def una_carrera(c, semilla, max_t=25.0):
    rng = np.random.RandomState(semilla)
    c.colocar(0.0, 0.0)
    if semilla != 1000:                               # postura inicial con ruido, como el reset del entrenamiento
        c.d.qpos[c.qadr] += rng.uniform(-0.05, 0.05, len(c.qadr))
        c.d.qpos[c.m.joint("RHipYawPitch").qposadr[0]] = c.d.qpos[c.m.joint("LHipYawPitch").qposadr[0]]
        mujoco.mj_forward(c.m, c.d)
    for _ in range(int(round(1.0 / c.dt))):          # 1 s de pie, como en la demo
        c.de_pie(c.dt)
    c.arrancar(medicion_simulada(c, rng) if c.pol.vision else None)
    x0 = c.d.qpos[0]
    c.registro = {k: [] for k in ("par", "w", "w0", "q", "lim", "pies", "sep")}
    resultado = "TIEMPO"
    while c.t < max_t:
        c.paso(lambda: medicion_simulada(c, rng))
        if c.cayo():
            resultado = "CAIDA"
            break
        if c.meta():
            resultado = "META"
            break
    r = {k: np.array(v) for k, v in c.registro.items()}
    c.registro = None
    r.update(resultado=resultado, t=c.t, distancia=float(c.d.qpos[0] - x0), d_fin=float(c.d.qpos[1]))
    return r


def metricas(pol, carreras, dt_fis, masa):
    motores = pol.meta["motores"]
    par = np.concatenate([r["par"] for r in carreras])
    w = np.concatenate([r["w"] for r in carreras])
    q = np.concatenate([r["q"] for r in carreras])
    lim = np.concatenate([r["lim"] for r in carreras])
    w0 = np.concatenate([r["w0"] for r in carreras])
    rango = carreras[0]["rango"]
    por_motor = {}
    for j, nombre in enumerate(motores):
        mr = MOTOR_REAL[nombre]
        a, v = np.abs(par[:, j]), np.abs(w[:, j])
        # curva par-velocidad del motor real, con la velocidad del instante en que se aplica el par
        curva = mr["parada"] * np.clip(1.0 - np.abs(w0[:, j]) / mr["w_max"], 0.0, 1.0)
        rms = float(np.sqrt(np.mean(par[:, j] ** 2)))
        por_motor[nombre] = dict(
            tipo=mr["tipo"], par_rms=rms, par_p99=float(np.percentile(a, 99)), par_max=float(a.max()),
            calor=(rms / mr["continuo"]) if mr["continuo"] else None,
            sobre_continuo=float(np.mean(a > mr["continuo"])) if mr["continuo"] else None,
            tope_sim=float(np.mean(a >= 0.98 * np.maximum(lim[:, j], 1e-6))),
            fuera_curva=float(np.mean(a > curva + 1e-6)),
            vel_p99=float(np.percentile(v, 99)), vel_max=float(v.max()), w_max=mr["w_max"],
            cerca_tope=float(np.mean((q[:, j] - rango[j, 0] < CERCA_TOPE) | (rango[j, 1] - q[:, j] < CERCA_TOPE))),
        )
    peso = masa * 9.81
    pies = np.concatenate([r["pies"] for r in carreras])
    energia = [float(np.sum(np.abs(r["par"] * r["w"])) * dt_fis / max(r["distancia"], 1e-3)) for r in carreras]
    metas = [r for r in carreras if r["resultado"] == "META"]
    # serie de ejemplo (primera carrera, cada 12 ms, 4 s desde el segundo 3): par de pierna izquierda
    r0 = carreras[0]
    i0, i1 = int(3.0 / dt_fis), int(7.0 / dt_fis)
    serie = {n: [round(float(x), 3) for x in r0["par"][i0:i1:6, motores.index(n)]]
             for n in ("LHipPitch", "LKneePitch", "LAnklePitch", "LHipRoll", "LAnkleRoll")}
    serie_w = {n: [round(float(x), 3) for x in r0["w"][i0:i1:6, motores.index(n)]] for n in ("LKneePitch", "LHipRoll")}
    return dict(
        politica=pol.nombre, generacion=pol.meta["generacion"], vx=pol.meta["vx"], vision=pol.vision,
        carreras=len(carreras), metas=len(metas), caidas=sum(r["resultado"] == "CAIDA" for r in carreras),
        t_meta=[round(r["t"], 2) for r in metas], t_meta_medio=float(np.mean([r["t"] for r in metas])) if metas else None,
        energia_por_metro=float(np.mean(energia)),
        pie_pico_peso=float(pies.max() / peso), pie_p99_peso=float(np.percentile(pies.max(axis=1), 99) / peso),
        separacion_min_cm=float(100 * min(r["sep"].min() for r in carreras)),
        motores=por_motor, serie_par=serie, serie_vel=serie_w, serie_dt=6 * dt_fis,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--salida", default=str(ROOT / "results" / "esfuerzo" / "esfuerzo.json"))
    a = ap.parse_args()
    mf = modelo_fisica()
    masa = float(mf.body_subtreemass[mf.body("torso").id])
    pols = cargar_politicas()
    todo = []
    for pol in pols:
        t0 = time.time()
        c = CarreraRegistrada(mf, pol)
        carreras = []
        for s in range(a.semillas):
            r = una_carrera(c, 1000 + s)
            r["rango"] = c.rango
            carreras.append(r)
        res = metricas(pol, carreras, mf.opt.timestep, masa)
        todo.append(res)
        peor = max(((n, v["calor"]) for n, v in res["motores"].items() if v["calor"] is not None), key=lambda x: x[1])
        print("%-32s metas %d/%d  t %s  energia %.1f J/m  pie pico %.1fx peso  mas caliente %s %.0f %%  (%.0f s)"
              % (pol.nombre, res["metas"], res["carreras"], res["t_meta"], res["energia_por_metro"], res["pie_pico_peso"],
                 peor[0], 100 * peor[1], time.time() - t0))
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    json.dump(dict(fecha=time.strftime("%Y-%m-%d %H:%M"), semillas=a.semillas, masa_kg=masa, dt_fisica=mf.opt.timestep,
                   motores_reales={"Tipo 5": TIPO5, "Tipo 1": TIPO1}, politicas=todo),
              open(a.salida, "w"), indent=1)
    print("Guardado en", a.salida)


if __name__ == "__main__":
    main()
