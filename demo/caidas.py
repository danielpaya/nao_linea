# -*- coding: utf-8 -*-
"""Analisis de caidas: como sufre cada articulacion del NAO cuando se cae caminando con cada politica.

Sirve para las politicas de los dos repos, con el MISMO motor de caidas:
  --fuente daniel   politicas numpy de este repo (rl/politicas/*/politica_np.npz)
  --fuente camilo   politicas de CamiloGomezB/nao-v6-rl-race (policies/*/model.zip + model_vecnormalize.pkl), con su
                    propio entorno NaoRaceEnv para la observacion (camara simulada, IMU, reloj de marcha)

Uso (Windows: desde C:\\demo_carril; Ubuntu/WSL: desde la raiz del repo; igual en los dos):
    uv run python -m demo.caidas                                   # Daniel: 100 caidas por politica, todos los nucleos
    uv run python -m demo.caidas --caidas 200 --procesos 12
    uv run python -m demo.caidas --politicas v06 v07a
    uv run python -m demo.caidas --fuente camilo --repo-camilo C:\\ruta\\nao-v6-rl-race
    uv run python -m demo.caidas --fuente camilo --repo-camilo ~/nao-v6-rl-race --politicas rl_v1.2.1 carrera_v02
    uv run python -m demo.caidas --verificar results/caidas/caidas.json      # re-simula 3 caidas por politica y compara
    uv run python -m demo.caidas --videos 10                       # ademas, videos de las 10 peores (demo/caidas_video.py)
Salida: results/caidas/caidas.json (daniel) o results/caidas/caidas_camilo.json (camilo): una fila por caida, con
todo lo necesario para repetirla (semilla, causa, nivel) y un bloque "reproducibilidad" (comando, commits, versiones,
maquina, SHA-256 de cada politica y del modelo).

Velocidad: cada simulacion es independiente y se reparte entre procesos (--procesos, por defecto todos los nucleos
menos uno). En Windows y en Ubuntu funciona igual (procesos "spawn"). La politica de Camilo usa torch con 1 hilo por
proceso.

Reproducibilidad: la simulacion k de una politica usa la semilla semilla_base + k y la causa CAUSAS[k % 7]. Las caidas
que se guardan son las primeras N simulaciones (en orden de k) en que el robot se cae, asi que el resultado NO depende
de cuantos procesos se usen ni del orden en que terminan. MuJoCo es determinista en la misma maquina; entre Windows y
Linux (o versiones distintas de MuJoCo) los numeros pueden diferir en los ultimos decimales y, en caidas al limite,
cambiar el desenlace: por eso se guardan las versiones y se puede comprobar con --verificar.

Simulacion (MuJoCo normal, CPU):
  - Modelo COMPLETO del NAO (assets/nao/scene.xml, el mismo archivo en los dos repos): todas las piezas chocan con el
    piso y entre si. Friccion 1.0 en todas las piezas (la del modelo).
  - El robot sale centrado y recto, camina con la politica (sin las perturbaciones de entrenamiento) y entre 1.5 y
    4.5 s despues de arrancar se le provoca una caida. Cada simulacion tiene una causa y parametros al azar distintos
    (7 tipos de causa, rotando):
        empujon        velocidad horizontal de golpe al robot (0.5-1.2 m/s, direccion al azar)
        golpe_torso    fuerza horizontal en el torso (40-100 N durante 50-150 ms)
        tropiezo       el pie en el aire queda frenado (30-80 N hacia atras durante 100-200 ms)
        resbalon       el piso pasa a friccion 0.02-0.08 (mancha resbalosa)
        motor_muerto   un motor de pierna al azar deja de dar par
        motores_debiles  todos los motores de pierna dan solo el 25-45 % de su par
        retraso        la accion llega 5-10 ciclos tarde (120-240 ms); mientras tanto el motor mantiene el ultimo objetivo
    Si a los 3 s no se cayo, se repite la misma causa mas fuerte (x1.5, hasta 3 veces); si aun asi no cae, se anota
    como "resistio" y se pasa a la siguiente.
  - Caida = torso inclinado mas de 35 grados o a menos de 22 cm del piso. Desde ese instante se simulan dos casos:
        activo     la politica sigue mandando los motores (sin proteccion)
        protegido  los motores se sueltan (rigidez casi cero), como hace el gestor de caidas de NAOqi
    y se sigue 1.5 s, hasta que el robot queda en el piso.
  - Motores: cada politica con el modelo de motor con que se entreno (Daniel: par limitado por la curva par-velocidad
    si su generacion la usaba; Camilo: el forcerange del modelo).
Que se mide en cada caida, cada 2 ms (desde la causa hasta el final):
  por articulacion (las 24): par del motor, carga externa (qfrc_constraint: incluye golpes contra el piso y contra el
    tope), par contra el tope mecanico y velocidad. Picos de carga y de tope: maximo del promedio de 20 ms (el pico de
    un paso de 2 ms con piso rigido es un impulso que exagera la carga); se guarda tambien el pico instantaneo (ext_inst).
  por pieza del cuerpo: fuerza contra el piso (/ peso del robot) y cual toca primero.
  direccion de la caida y velocidad del torso al tocar el piso.
Criterios de riesgo por articulacion (results/... -> criterios):
  carga externa > par de parada del motor real; velocidad > velocidad maxima del motor; > 2 N m contra el tope.
Motores (en la articulacion): tipo 5 (HipPitch, KneePitch, AnklePitch) parada 8.51 N m, 6.96 rad/s; tipo 1
(HipYawPitch, HipRoll, AnkleRoll) parada 11.97 N m, 4.16 rad/s; hombro (pitch) 2.15 N m, 7.46 rad/s; codo (roll)
2.48 N m, 6.47 rad/s. El resto (cabeza, hombro roll, codo yaw, muneca) sin dato de motor.
Limitaciones: piezas de choque aproximadas (cajas y esferas), piso rigido; servo simulado como PD con tope. Es una
comparacion entre politicas y entre causas, no una prediccion exacta de dano.
"""
import argparse
import copy
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mujoco  # noqa: E402

T5 = dict(parada=8.51, w_max=6.96)
T1 = dict(parada=11.97, w_max=4.16)
MOTOR = {"HipPitch": T5, "KneePitch": T5, "AnklePitch": T5, "HipYawPitch": T1, "HipRoll": T1, "AnkleRoll": T1,
         "ShoulderPitch": dict(parada=2.15, w_max=7.46), "ElbowRoll": dict(parada=2.48, w_max=6.47)}
CAUSAS = ["empujon", "golpe_torso", "tropiezo", "resbalon", "motor_muerto", "motores_debiles", "retraso"]
PIERNA = ["LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
          "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"]
CASOS = ("activo", "protegido")
TOPE_GOLPE = 2.0           # N m contra el tope mecanico
INCL_CAIDA, ALTURA_CAIDA = 35.0, 0.22
POST = 1.5                 # s simulados despues de detectar la caida
VENTANA = 10               # pasos de 2 ms (20 ms) para los picos de carga y de tope
ESPERA_CAUSA = 3.0         # s que se espera la caida despues de la causa
SEMILLA_BASE = 10000
MAX_K = 4                  # se prueban hasta MAX_K x N simulaciones por politica
CAMPOS_REG = ("act", "ext", "tope", "w", "piso", "vz")


def modelo_completo():
    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "nao" / "scene.xml"))
    m.opt.timestep = 0.002
    return m


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# ============================================================================================== adaptadores
class AdaptadorDaniel(object):
    """Politica numpy de este repo (como demo/rl_vivo.Carrera). La fisica la hace el motor de caidas."""
    fuente = "daniel"

    def __init__(self, ruta):
        from demo.rl_vivo import Carrera, Politica, medicion_simulada
        self._medicion = medicion_simulada
        pol = Politica(ruta)
        self.ruta = Path(ruta)
        self.nombre = pol.meta["generacion"].replace("nao_", "")
        self.descripcion = pol.nombre
        self.vx = float(pol.meta["vx"])
        self.archivos = [self.ruta]
        adaptador = self

        class _C(Carrera):
            def _fisica(self_c):
                adaptador.motor.fisica()

        self.c = _C(modelo_completo(), pol)
        self.m, self.d = self.c.m, self.c.d
        c = self.c
        self.tope = c.tope.copy()                      # par maximo de cada motor de la politica

    @property
    def t(self):
        return self.c.t

    def limite(self, base):
        """Limite de par de cada actuador en este instante (como en el entrenamiento)."""
        c = self.c
        if c.mt["motores_reales"]:
            w = self.d.qvel[c.vadr]
            base[c.act] = self.tope * np.clip(1.0 - np.abs(w) / c.vel_max, 0.0, 1.0)
        return base

    def iniciar(self, rng, semilla):
        c = self.c
        c.colocar(0.0, 0.0)
        c.d.qpos[c.qadr] += rng.uniform(-0.03, 0.03, len(c.qadr))
        c.d.qpos[c.m.joint("RHipYawPitch").qposadr[0]] = c.d.qpos[c.m.joint("LHipYawPitch").qposadr[0]]
        mujoco.mj_forward(c.m, c.d)
        for _ in range(int(round(0.5 / c.dt))):
            c.de_pie(c.dt)
        self.rng = rng
        c.arrancar(self._medicion(c, rng) if c.pol.vision else None)

    def decidir(self):
        c = self.c
        self._a = c.pol(c.estado)
        ctrl = c.ctrl_home.copy()
        ctrl[c.act] = np.clip(c.postura + self._a * c.mt["action_scale"], c.lo, c.hi)
        return ctrl

    def actualizar(self):
        c = self.c
        c.accion_ant = self._a
        c.rumbo_est += c._gyro_z_mundo() * c.dt
        if c.pol.vision:
            if c.vis_cuenta <= 0:
                if c.vis_muestra is not None:
                    c.vis_obs, c.vis_edad = c.vis_muestra, 0.0
                else:
                    c.vis_edad += c.dt
                med = self._medicion(c, self.rng)
                c.vis_muestra = None if med is None else np.array(med)
                c.vis_cuenta = c.periodo - 1
            else:
                c.vis_cuenta -= 1
                c.vis_edad += c.dt
        f = c.fase + c.dfase
        c.fase = np.fmod(f + math.pi, 2 * math.pi) - math.pi
        c.t += c.dt
        c.estado = c._obs(c.estado)

    _ESTADO = ("t", "estado", "accion_ant", "fase", "dfase", "rumbo_ref", "rumbo_est", "vis_obs", "vis_muestra",
               "vis_cuenta", "vis_edad")

    def guardar(self):
        return copy.deepcopy({k: getattr(self.c, k) for k in self._ESTADO})

    def cargar(self, s):
        for k, v in copy.deepcopy(s).items():
            setattr(self.c, k, v)


class AdaptadorCamilo(object):
    """Politica SB3 de CamiloGomezB/nao-v6-rl-race con su NaoRaceEnv (observacion, camara simulada, IMU). La fisica
    la hace el motor de caidas: se reemplaza env._paso_fisica; lo demas de env.step se repite tal cual."""
    fuente = "camilo"
    _SIN_COPIA = {"model", "data", "_renderer", "observation_space", "action_space", "spec", "_np_random_seed"}

    def __init__(self, ruta, repo, vx):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        repo = Path(repo).resolve()
        if str(repo) not in sys.path[:1]:
            sys.path.insert(0, str(repo))
        if "src" in sys.modules and not str(getattr(sys.modules["src"], "__file__", "")).startswith(str(repo)):
            raise RuntimeError("Ya se importo otro paquete 'src' (no el de %s)" % repo)
        import torch
        import yaml
        from stable_baselines3 import PPO
        from src.envs.nao_race_env import NaoRaceEnv
        from src.race import contract as C
        torch.set_num_threads(1)
        self.C = C
        self.ruta = Path(ruta)
        carpeta = self.ruta.parent
        self.nombre = carpeta.name
        self.vx = float(vx)
        cfg_p, vn_p = carpeta / "config.yaml", carpeta / "model_vecnormalize.pkl"
        self.archivos = [self.ruta, vn_p, cfg_p]
        kw = dict(yaml.safe_load(cfg_p.read_text(encoding="utf-8")).get("env", {}))
        # Las generaciones anteriores al paso 58c (sin la clave) se entrenaron con el rumbo "arranque" (docstring del env).
        kw.setdefault("rumbo_objetivo", "arranque")
        kw.update(friction=1.0, vx_cambio_prob=0.0, perturbaciones=False, vision=True, sin_vision_prob=0.0)
        self.env_kwargs = kw
        self.env = NaoRaceEnv(**kw)
        self.env._paso_fisica = self._paso_fisica_env
        self.m, self.d = self.env.model, self.env.data
        self.ppo = PPO.load(str(self.ruta), device="cpu")
        with open(vn_p, "rb") as f:
            vn = pickle.load(f)
        self.media = np.asarray(vn.obs_rms.mean, dtype=np.float64)
        self.desv = np.sqrt(np.asarray(vn.obs_rms.var, dtype=np.float64) + vn.epsilon)
        self.clip_obs = float(vn.clip_obs)
        self.norm = bool(vn.norm_obs)
        self.descripcion = "%s (Camilo)  vx %.2f m/s  rumbo %s" % (self.nombre, self.vx, kw["rumbo_objetivo"])
        self.motor = None

    def _paso_fisica_env(self):
        self.motor.fisica()

    @property
    def t(self):
        return self.env.t

    def limite(self, base):
        return base                                      # se entreno con el forcerange del modelo

    def iniciar(self, rng, semilla):
        self.obs, _ = self.env.reset(seed=int(semilla), options={"y0": 0.0, "yaw0": 0.0, "vx_cmd": self.vx})

    def decidir(self):
        o = np.asarray(self.obs, dtype=np.float64)
        if self.norm:
            o = np.clip((o - self.media) / self.desv, -self.clip_obs, self.clip_obs)
        a, _ = self.ppo.predict(o.astype(np.float32), deterministic=True)
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        self._a = a
        obj = self.C.accion_a_objetivos(a)
        ctrl = self.d.ctrl.copy()
        ctrl[self.env._act] = np.array([obj[n] for n in self.C.ARTICULACIONES_ACCION]) + self.env._cal
        return ctrl

    def actualizar(self):
        env = self.env
        env._medir()
        env._avanzar_tiempo()
        env._prev_action[:] = self._a
        self.obs = env._obs()

    def guardar(self):
        e = {k: v for k, v in self.env.__dict__.items() if k not in self._SIN_COPIA and not callable(v)}
        return copy.deepcopy(dict(env=e, obs=self.obs))

    def cargar(self, s):
        s = copy.deepcopy(s)
        self.env.__dict__.update(s["env"])
        self.obs = s["obs"]


# ============================================================================================== motor de caidas
class MotorCaidas(object):
    """Fisica con perturbaciones y registro cada 2 ms, para cualquier adaptador."""

    def __init__(self, ad):
        self.ad = ad
        ad.motor = self
        m = self.m = ad.m
        self.d = ad.d
        self.nsub = int(round(0.024 / m.opt.timestep))
        self.dt_ctrl = self.nsub * m.opt.timestep
        self.base_fric = m.geom_friction.copy()
        self.base_fr = m.actuator_forcerange.copy()
        self.torso = m.body("torso").id
        self.jnt = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.jv = np.array([m.jnt_dofadr[j] for j in self.jnt])
        self.jnombre = [m.joint(j).name for j in self.jnt]
        self.rango = m.jnt_range[self.jnt]
        self.act_de_jnt = {m.actuator_trnid[a, 0]: a for a in range(m.nu)}
        self.act_pierna = {n: m.actuator(n).id for n in PIERNA}
        self.piso = m.geom("floor").id
        self.geoms = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and g != self.piso]
        self.gnombre = [m.geom(g).name for g in self.geoms]
        self.gidx = {g: i for i, g in enumerate(self.geoms)}
        self.pies = {m.geom("left_foot").id, m.geom("right_foot").id}
        self.peso = float(m.body_subtreemass[self.torso]) * 9.81
        self._ja = np.array([self.act_de_jnt.get(j, -1) for j in self.jnt])
        self._con_act = self._ja >= 0
        self._gpos = np.full(m.ngeom, -1)
        self._gpos[self.geoms] = np.arange(len(self.geoms))
        self._c6 = np.zeros(6)
        self.registro = None
        self.grabador = None        # funcion(motor) llamada en cada paso de 2 ms (videos)
        self.reiniciar_mundo()

    def reiniciar_mundo(self):
        self.m.geom_friction[:] = self.base_fric
        self.m.actuator_forcerange[:] = self.base_fr
        self.factor = np.ones(self.m.nu)
        self.xfrc, self.retraso, self.suelto = None, 0, False
        self.cola = deque()
        self.ultimo_ctrl = None
        self.d.xfrc_applied[:] = 0
        self.t_sim = 0.0

    # -- fisica
    def fisica(self):
        """24 ms (12 pasos de 2 ms)."""
        m, d = self.m, self.d
        for _ in range(self.nsub):
            if self.suelto:
                m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1] = -0.05, 0.05
            else:
                lim = self.ad.limite(self.base_fr[:, 1].copy()) * self.factor
                m.actuator_forcerange[:, 0] = -lim
                m.actuator_forcerange[:, 1] = lim
            if self.xfrc is not None:
                cuerpo, f, n = self.xfrc
                d.xfrc_applied[cuerpo, :3] = f
                self.xfrc = (cuerpo, f, n - 1) if n > 1 else None
            else:
                d.xfrc_applied[:] = 0
            mujoco.mj_step(m, d)
            self.t_sim += m.opt.timestep
            if self.registro is not None:
                self._registrar()
            if self.grabador is not None:
                self.grabador(self)
        m.actuator_forcerange[:] = self.base_fr

    def _registrar(self):
        m, d, r = self.m, self.d, self.registro
        act = np.zeros(len(self.jnt))
        act[self._con_act] = d.actuator_force[self._ja[self._con_act]]
        tope = np.zeros(m.njnt)
        n = d.nefc
        if n:
            sel = d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
            if sel.any():
                np.add.at(tope, d.efc_id[:n][sel], np.abs(d.efc_force[:n][sel]))
        piso = np.zeros(len(self.geoms))
        n = d.ncon
        if n:
            g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
            for i in np.flatnonzero((g1 == self.piso) | (g2 == self.piso)):
                g = self._gpos[g2[i] if g1[i] == self.piso else g1[i]]
                if g >= 0:
                    mujoco.mj_contactForce(m, d, int(i), self._c6)
                    piso[g] += abs(self._c6[0])
        r["act"].append(np.abs(act))
        r["ext"].append(np.abs(d.qfrc_constraint[self.jv]))
        r["tope"].append(tope[self.jnt])
        r["w"].append(np.abs(d.qvel[self.jv]))
        r["piso"].append(piso)
        r["vz"].append(d.qvel[2])

    def paso(self):
        """Un paso de control (24 ms): la politica decide, la accion llega (o se retrasa), fisica, observacion."""
        ctrl = self.ad.decidir()
        if self.retraso:
            self.cola.append(ctrl)
            ctrl = self.cola.popleft() if len(self.cola) > self.retraso else self.ultimo_ctrl
        self.d.ctrl[:] = ctrl
        self.ultimo_ctrl = ctrl.copy()
        self.fisica()
        self.ad.actualizar()

    def cae(self):
        R = self.d.xmat[self.torso].reshape(3, 3)
        return R[2, 2] < math.cos(math.radians(INCL_CAIDA)) or self.d.xpos[self.torso, 2] < ALTURA_CAIDA

    def direccion(self):
        """Hacia donde cae, respecto al rumbo del robot (con la inclinacion del torso)."""
        R = self.d.xmat[self.torso].reshape(3, 3)
        eje = R[:, 2]
        yaw = math.atan2(R[1, 0], R[0, 0])
        ad = eje[0] * math.cos(yaw) + eje[1] * math.sin(yaw)
        iz = -eje[0] * math.sin(yaw) + eje[1] * math.cos(yaw)
        if abs(ad) >= abs(iz):
            return "adelante" if ad > 0 else "atras"
        return "izquierda" if iz > 0 else "derecha"

    def provocar(self, causa, fuerza, rng):
        """Aplica la causa de caida; devuelve sus parametros."""
        m, d = self.m, self.d
        if causa == "empujon":
            v, ang = rng.uniform(0.5, 1.2) * fuerza, rng.uniform(0, 2 * math.pi)
            d.qvel[0:2] += v * np.array([math.cos(ang), math.sin(ang)])
            return dict(v_ms=round(v, 2), direccion_deg=round(math.degrees(ang)))
        if causa == "golpe_torso":
            f, ang, dur = rng.uniform(40, 100) * fuerza, rng.uniform(0, 2 * math.pi), rng.uniform(0.05, 0.15)
            self.xfrc = (self.torso, np.array([f * math.cos(ang), f * math.sin(ang), 0.0]), int(dur / m.opt.timestep))
            return dict(fuerza_n=round(f), direccion_deg=round(math.degrees(ang)), duracion_s=round(dur, 3))
        if causa == "tropiezo":
            izq, der = m.body("LFoot").id, m.body("RFoot").id
            pie = izq if d.xpos[izq, 2] > d.xpos[der, 2] else der
            f, dur = rng.uniform(30, 80) * fuerza, rng.uniform(0.10, 0.20)
            self.xfrc = (pie, np.array([-f, 0.0, 0.0]), int(dur / m.opt.timestep))
            return dict(pie="izq" if pie == izq else "der", fuerza_n=round(f), duracion_s=round(dur, 3))
        if causa == "resbalon":
            mu = max(0.005, rng.uniform(0.02, 0.08) / fuerza)
            m.geom_friction[:, 0] = mu
            return dict(friccion=round(mu, 3))
        if causa == "motor_muerto":
            n = min(3, int(round(fuerza)))
            idx = rng.choice(len(PIERNA), n, replace=False)
            for i in idx:
                self.factor[self.act_pierna[PIERNA[i]]] = 0.0
            return dict(motores=[PIERNA[i] for i in idx])
        if causa == "motores_debiles":
            frac = rng.uniform(0.25, 0.45) / fuerza
            for nombre in PIERNA:
                self.factor[self.act_pierna[nombre]] = frac
            return dict(fraccion_par=round(frac, 2))
        if causa == "retraso":
            k = int(rng.integers(5, 11) * fuerza)
            self.retraso = k
            self.ultimo_ctrl = self.d.ctrl.copy()
            return dict(ciclos=k, ms=24 * k)
        raise ValueError(causa)

    def resumir(self, r):
        """Picos por articulacion y por pieza, y primera pieza (no pie) en tocar el piso."""
        arr = {k: np.array(v) for k, v in r.items()}

        def suave(x, n=VENTANA):
            if len(x) < n:
                return x
            c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
            return (c[n:] - c[:-n]) / n

        out = {"act": arr["act"].max(0), "ext": suave(arr["ext"]).max(0), "ext_inst": arr["ext"].max(0),
               "tope": suave(arr["tope"]).max(0), "w": arr["w"].max(0)}
        piso = arr["piso"]
        out["piso"] = piso.max(0) / self.peso
        no_pie = [i for i, g in enumerate(self.geoms) if g not in self.pies]
        toca = (piso[:, no_pie] > 1e-3)
        if toca.any():
            k = int(np.argmax(toca.any(1)))
            out["primera"] = self.gnombre[no_pie[int(np.argmax(toca[k]))]]
            out["t_suelo"] = round(k * self.m.opt.timestep, 3)
            out["vz_suelo"] = round(float(arr["vz"][k]), 3)
        else:
            out["primera"], out["t_suelo"], out["vz_suelo"] = None, None, None
        return out

    def simular(self, causa, fuerza, semilla, casos=CASOS, al_detectar=None):
        """Una simulacion. (None, None) si se cayo antes de la causa; (False, params) si resistio; si se cayo,
        (resultado con los casos pedidos, params). al_detectar(motor): se llama en el instante de la caida."""
        rng = np.random.default_rng(semilla)
        self.reiniciar_mundo()
        self.ad.iniciar(rng, semilla)
        t_causa = rng.uniform(1.5, 4.5)
        while self.ad.t < t_causa:
            self.paso()
            if self.cae():
                return None, None
        self.registro = {k: [] for k in CAMPOS_REG}
        self.t_causa_sim = self.t_sim
        params = self.provocar(causa, fuerza, rng)
        t0 = self.ad.t
        while self.ad.t - t0 < ESPERA_CAUSA:
            self.paso()
            if self.cae():
                break
        else:
            self.registro = None
            return False, params
        t_det = self.ad.t - t0
        dirc = self.direccion()
        self.t_caida_sim = self.t_sim
        if al_detectar:
            al_detectar(self)
        estado = mujoco.MjData(self.m)
        mujoco.mj_copyData(estado, self.m, self.d)
        copia = dict(registro={k: list(v) for k, v in self.registro.items()}, xfrc=self.xfrc, cola=deque(self.cola),
                     ultimo=self.ultimo_ctrl.copy(), factor=self.factor.copy(), fric=self.m.geom_friction.copy(),
                     t_sim=self.t_sim, ad=self.ad.guardar())
        res = {}
        npasos = int(round(POST / self.dt_ctrl))
        for caso in casos:
            mujoco.mj_copyData(self.d, self.m, estado)
            self.registro = {k: list(v) for k, v in copia["registro"].items()}
            self.xfrc, self.cola, self.ultimo_ctrl = copia["xfrc"], deque(copia["cola"]), copia["ultimo"].copy()
            self.factor, self.t_sim = copia["factor"].copy(), copia["t_sim"]
            self.m.geom_friction[:] = copia["fric"]
            self.ad.cargar(copia["ad"])
            self.suelto = caso == "protegido"
            self.caso = caso
            for _ in range(npasos):
                if caso == "activo":
                    self.paso()
                else:
                    self.fisica()
            res[caso] = self.resumir(self.registro)
            self.suelto = False
        self.registro = None
        return dict(t_hasta_caer=round(t_det, 3), direccion=dirc, casos=res), params

    def criterios(self):
        filas = []
        for i, nombre in enumerate(self.jnombre):
            mr = MOTOR.get(nombre[1:]) if nombre[0] in "LR" else None
            filas.append(dict(articulacion=nombre, parada=mr["parada"] if mr else None, w_max=mr["w_max"] if mr else None,
                              lim_inf=float(self.rango[i, 0]), lim_sup=float(self.rango[i, 1])))
        return filas


def redondear(v):
    return [round(float(x), 3) for x in v]


def fila_caso(v):
    return dict(primera=v["primera"], t_suelo=v["t_suelo"], vz_suelo=v["vz_suelo"], act=redondear(v["act"]),
                ext=redondear(v["ext"]), ext_inst=redondear(v["ext_inst"]), tope=redondear(v["tope"]),
                w=redondear(v["w"]), piso=redondear(v["piso"]))


def simular_k(motor, k, base=SEMILLA_BASE):
    """La simulacion k de una politica: causa CAUSAS[k % 7], semilla base + k, hasta 3 niveles de fuerza."""
    causa, semilla = CAUSAS[k % len(CAUSAS)], base + k
    for nivel in range(3):
        res, params = motor.simular(causa, 1.5 ** nivel, semilla)
        if res is None and params is None:
            return dict(k=k, tipo="sola", intentos=nivel + 1, causa=causa)
        if res:
            casos = {c: fila_caso(v) for c, v in res["casos"].items()}
            return dict(k=k, tipo="caida", intentos=nivel + 1, causa=causa, nivel=nivel, semilla=semilla, params=params,
                        t_hasta_caer=res["t_hasta_caer"], direccion=res["direccion"], casos=casos)
    return dict(k=k, tipo="resistio", intentos=3, causa=causa)


# ============================================================================================== politicas
def politicas_daniel(nombres=None):
    rutas = sorted(glob.glob(str(ROOT / "rl" / "politicas" / "*" / "politica_np.npz")))
    out = []
    for r in rutas:
        meta = json.loads(str(np.load(r)["meta"]))
        gen = meta["generacion"].replace("nao_", "")
        if not nombres or gen in nombres:
            out.append(dict(fuente="daniel", nombre=gen, ruta=r, vx=float(meta["vx"])))
    return out


def politicas_camilo(repo, nombres=None, vx=0.5):
    rutas = sorted(glob.glob(str(Path(repo) / "policies" / "*" / "model.zip")))
    out = []
    for r in rutas:
        n = Path(r).parent.name
        if not nombres or n in nombres:
            out.append(dict(fuente="camilo", nombre=n, ruta=r, vx=vx, repo=str(Path(repo).resolve())))
    return out


def crear_adaptador(p):
    if p["fuente"] == "daniel":
        return AdaptadorDaniel(p["ruta"])
    return AdaptadorCamilo(p["ruta"], p["repo"], p["vx"])


# ============================================================================================== procesos
_CACHE = {}


def _iniciar_proceso():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _motor(p):
    clave = (p["fuente"], p["ruta"])
    if clave not in _CACHE:
        _CACHE[clave] = MotorCaidas(crear_adaptador(p))
    return _CACHE[clave]


def _tarea(p, k, base):
    return p["nombre"], simular_k(_motor(p), k, base)


def _tarea_repetir(p, semilla, causa, nivel):
    res, _ = _motor(p).simular(causa, 1.5 ** nivel, semilla)
    return {c: fila_caso(v) for c, v in res["casos"].items()} if res else None, res and res["t_hasta_caer"]


def correr(pols, n_caidas, procesos, base, log=print):
    """Simula hasta juntar n_caidas por politica, repartiendo las simulaciones k entre procesos. Devuelve
    {nombre: lista de resultados por k (solo hasta la n-esima caida)}."""
    est = {p["nombre"]: dict(p=p, sig=0, res={}, pend=0, listo=False, t0=time.time()) for p in pols}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=procesos, mp_context=ctx, initializer=_iniciar_proceso) as ex:
        vuelo = {}

        def prefijo(e):
            n, k = 0, 0
            while k in e["res"]:
                n += e["res"][k]["tipo"] == "caida"
                if n >= n_caidas:
                    return n, k
                k += 1
            return n, None

        def rellenar():
            activos = [e for e in est.values() if not e["listo"]]
            if not activos:
                return
            cupo = max(2, (2 * procesos) // len(activos) + 1)
            for e in activos:
                caidas = sum(r["tipo"] == "caida" for r in e["res"].values())
                tasa = caidas / len(e["res"]) if len(e["res"]) >= 7 else 0.7
                while (e["pend"] < cupo and e["sig"] < MAX_K * n_caidas
                       and caidas + e["pend"] * max(tasa, 0.3) < n_caidas + procesos * 0.5):
                    f = ex.submit(_tarea, e["p"], e["sig"], base)
                    vuelo[f] = e["p"]["nombre"]
                    e["sig"] += 1
                    e["pend"] += 1

        rellenar()
        while vuelo:
            hechos, _ = wait(list(vuelo), return_when=FIRST_COMPLETED)
            for f in hechos:
                nombre = vuelo.pop(f)
                e = est[nombre]
                e["pend"] -= 1
                if e["listo"] or f.cancelled():
                    continue
                _, r = f.result()
                e["res"][r["k"]] = r
                n, k_fin = prefijo(e)
                if k_fin is not None or (e["sig"] >= MAX_K * n_caidas and e["pend"] == 0):
                    e["listo"] = True
                    for g, nm in list(vuelo.items()):
                        if nm == nombre:
                            g.cancel()
                    log("  %-12s listo: %d caidas en %.0f s" % (nombre, n, time.time() - e["t0"]))
                elif r["tipo"] == "caida" and n and n % 25 == 0 and not e.get("aviso_%d" % n):
                    e["aviso_%d" % n] = True
                    log("  %-12s %d/%d caidas (%.0f s)" % (nombre, n, n_caidas, time.time() - e["t0"]))
            rellenar()
    salida = {}
    for nombre, e in est.items():
        lista, n = [], 0
        for k in sorted(e["res"]):
            if n >= n_caidas:
                break
            if k > 0 and k - 1 not in e["res"]:
                break
            lista.append(e["res"][k])
            n += e["res"][k]["tipo"] == "caida"
        salida[nombre] = dict(lista=lista, segundos=round(time.time() - e["t0"], 1))
    return salida


# ============================================================================================== reproducibilidad
def _git(carpeta):
    try:
        c = subprocess.run(["git", "-C", str(carpeta), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        s = subprocess.run(["git", "-C", str(carpeta), "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        h = c.stdout.strip()
        return (h + ("-modificado" if s.stdout.strip() else "")) if h else None
    except Exception:
        return None


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def reproducibilidad(args, pols, procesos):
    repos = {"daniel": _git(ROOT)}
    if any(p["fuente"] == "camilo" for p in pols):
        repos["camilo"] = _git(pols[0]["repo"])
    archivos = {}
    for p in pols:
        ad_arch = [p["ruta"]] + ([str(Path(p["ruta"]).parent / "model_vecnormalize.pkl"),
                                  str(Path(p["ruta"]).parent / "config.yaml")] if p["fuente"] == "camilo" else [])
        for a in ad_arch:
            archivos[os.path.relpath(a, p.get("repo", ROOT))] = sha256(a)
    modelo = ROOT / "assets" / "nao"
    return dict(
        comando=" ".join([Path(sys.executable).name] + sys.argv), argumentos=vars(args), procesos=procesos,
        semilla_base=args.semilla_base, commits=repos, python=platform.python_version(),
        sistema=platform.platform(), cpu=platform.processor() or platform.machine(), nucleos=os.cpu_count(),
        versiones={m: _version(m) for m in ("mujoco", "numpy", "torch", "stable_baselines3", "gymnasium")},
        sha256=dict(archivos, **{"assets/nao/scene.xml": sha256(modelo / "scene.xml"),
                                 "assets/nao/nao.xml": sha256(modelo / "nao.xml")}),
        nota="Caida k de cada politica = semilla semilla_base + k, causa CAUSAS[k % 7]. Repetir: --verificar.")


def verificar(ruta_json, n, procesos, repo_camilo, log=print):
    """Re-simula las primeras n caidas de cada politica de un resultado y compara los numeros."""
    d = json.load(open(ruta_json, encoding="utf-8"))
    tareas = []
    for pol in d["politicas"]:
        p = dict(fuente=pol["fuente"], nombre=pol["politica"], vx=pol["vx"])
        if pol["fuente"] == "camilo":
            repo = Path(repo_camilo)
            p.update(repo=str(repo.resolve()), ruta=str(repo / pol["ruta"]))
        else:
            p["ruta"] = str(ROOT / pol["ruta"])
        for f in pol["caidas"][:n]:
            tareas.append((p, f))
    peor, iguales = 0.0, 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=procesos, mp_context=ctx, initializer=_iniciar_proceso) as ex:
        futs = [(ex.submit(_tarea_repetir, p, f["semilla"], f["causa"], f["nivel"]), p, f) for p, f in tareas]
        for fut, p, f in futs:
            casos, t = fut.result()
            if casos is None:
                log("  %-12s caida %3d: NO se cayo al repetirla" % (p["nombre"], f["id"]))
                peor = float("inf")
                continue
            dif = max(max(abs(a - b) for a, b in zip(casos[c][k], f["casos"][c][k]))
                      for c in f["casos"] for k in ("ext", "act", "tope", "w", "piso"))
            dif = max(dif, abs(t - f["t_hasta_caer"]))
            peor = max(peor, dif)
            iguales += dif < 1e-6
            log("  %-12s caida %3d (%s): diferencia maxima %.3g %s" % (p["nombre"], f["id"], f["causa"], dif,
                                                                    "(igual)" if dif < 1e-6 else ""))
    log("Verificacion: %d de %d caidas identicas; diferencia maxima %.3g" % (iguales, len(tareas), peor))
    return peor


# ============================================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", choices=("daniel", "camilo"), default="daniel")
    ap.add_argument("--repo-camilo", default=os.environ.get("REPO_CAMILO", str(ROOT.parent / "nao-v6-rl-race")),
                    help="raiz del repo de Camilo (o variable REPO_CAMILO)")
    ap.add_argument("--vx-camilo", type=float, default=0.5, help="vx_cmd de las politicas de Camilo (su eval: 0.5)")
    ap.add_argument("--caidas", type=int, default=100)
    ap.add_argument("--politicas", nargs="*", default=None, help="p. ej. v06 v07a  o  rl_v1.2.1 carrera_v02")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--semilla-base", type=int, default=SEMILLA_BASE)
    ap.add_argument("--salida", default=None, help="por defecto results/caidas/caidas.json o caidas_camilo.json")
    ap.add_argument("--verificar", default=None, metavar="JSON", help="re-simula caidas de un resultado y compara")
    ap.add_argument("--n-verificar", type=int, default=3)
    ap.add_argument("--videos", type=int, default=0, help="al terminar, videos de las N peores caidas")
    a = ap.parse_args()

    if a.verificar:
        peor = verificar(a.verificar, a.n_verificar, a.procesos, a.repo_camilo)
        sys.exit(0 if peor < 1e-6 else 1)

    if a.fuente == "daniel":
        pols = politicas_daniel(a.politicas)
        salida = a.salida or str(ROOT / "results" / "caidas" / "caidas.json")
    else:
        if not (Path(a.repo_camilo) / "policies").is_dir():
            sys.exit("No encuentro %s/policies: pasa --repo-camilo <raiz del repo nao-v6-rl-race>" % a.repo_camilo)
        pols = politicas_camilo(a.repo_camilo, a.politicas, a.vx_camilo)
        salida = a.salida or str(ROOT / "results" / "caidas" / "caidas_camilo.json")
    if not pols:
        sys.exit("No hay politicas (revisa --politicas)")
    procesos = max(1, min(a.procesos, len(pols) * a.caidas))
    print("%d politicas (%s), %d caidas c/u, %d procesos" % (len(pols), a.fuente, a.caidas, procesos), flush=True)

    # Metadatos (en este proceso: modelo, nombres de articulaciones y piezas, descripcion de cada politica)
    t0 = time.time()
    descripcion = {}
    motor0 = None
    for p in pols:
        ad = crear_adaptador(p)
        descripcion[p["nombre"]] = ad.descripcion
        motor0 = motor0 or MotorCaidas(ad)
    res = correr(pols, a.caidas, procesos, a.semilla_base, log=lambda s: print(s, flush=True))

    todo = []
    for p in pols:
        r = res[p["nombre"]]
        caidas, intentos, resistio, solas = [], 0, {k: 0 for k in CAUSAS}, 0
        for x in r["lista"]:
            intentos += x["intentos"]
            if x["tipo"] == "caida":
                fila = {k: v for k, v in x.items() if k not in ("tipo", "intentos")}
                fila = dict(politica=p["nombre"], id=len(caidas), **fila)
                caidas.append(fila)
            elif x["tipo"] == "resistio":
                resistio[x["causa"]] += 1
            else:
                solas += 1
        base_rel = p.get("repo", str(ROOT))
        todo.append(dict(politica=p["nombre"], fuente=p["fuente"], nombre=descripcion[p["nombre"]], vx=p["vx"],
                         ruta=os.path.relpath(p["ruta"], base_rel).replace("\\", "/"), caidas=caidas, intentos=intentos,
                         resistio=resistio, se_cayo_solo=solas, segundos=r["segundos"]))
        dirs = {}
        for f in caidas:
            dirs[f["direccion"]] = dirs.get(f["direccion"], 0) + 1
        print("%-12s %d caidas en %d simulaciones (resistio: %s; se cayo antes de la causa: %d) | direcciones %s"
              % (p["nombre"], len(caidas), intentos, ", ".join("%s %d" % kv for kv in resistio.items() if kv[1]) or "-",
                 solas, dirs), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    crit = motor0.criterios()
    rep = reproducibilidad(a, pols, procesos)
    rep["segundos_total"] = round(time.time() - t0, 1)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(dict(fecha=time.strftime("%Y-%m-%d %H:%M"), fuente=a.fuente, ventana_ms=2 * VENTANA,
                       articulaciones=[x["articulacion"] for x in crit], criterios=crit, piezas=motor0.gnombre,
                       umbral_tope_nm=TOPE_GOLPE, inclinacion_caida_deg=INCL_CAIDA, segundos_despues=POST,
                       reproducibilidad=rep, politicas=todo), f)
    print("Guardado en %s (%.0f s)" % (salida, time.time() - t0), flush=True)
    if a.videos:
        from demo import caidas_video
        caidas_video.hacer_videos(salida, a.videos, repo_camilo=a.repo_camilo)


if __name__ == "__main__":
    main()
