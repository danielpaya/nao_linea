# -*- coding: utf-8 -*-
"""Corre una caminata clasica (tipo NAOqi o tipo B-Human) en el carril del simulador.

Uso rapido (desde la raiz del repo):
    uv run python -m demo.caminatas.simular --caminata naoqi --vx 0.1
    uv run python -m demo.caminatas.simular --caminata bhuman --vx 0.2 --vivo
    uv run python -m demo.caminatas.simular --caminata naoqi --mu 0.6 --d0 0.1

Usa MuJoCo directo (sin el entorno de RL): ciclo de control de 12 ms = 6 pasos de fisica de 2 ms.
La caminata solo recibe sensores del NAO real (giroscopio, inclinacion, FSR); la posicion verdadera se usa
solo para medir el resultado (meta, salida del carril, caida).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from demo.caminatas.base import DT  # noqa: E402

ESCENA = ROOT / "assets" / "nao" / "scene_carril.xml"
LANE_HALF, LENGTH = 0.35, 5.0
FOOT_HALF_WIDTH = 0.045
SUBPASOS = int(round(DT / 0.002))


def crear(nombre, model, **kw):
    if nombre.startswith("naoqi"):
        from demo.caminatas.naoqi_like import CaminataNaoqi
        return CaminataNaoqi(model, **kw)
    if nombre.startswith("bhuman"):
        from demo.caminatas.bhuman_like import CaminataBHuman
        return CaminataBHuman(model, **kw)
    raise ValueError("caminata desconocida: %s" % nombre)


def cargar_modelo(mu=1.0):
    """Modelo del carril con friccion de deslizamiento mu en el piso y en los pies (MuJoCo usa el maximo)."""
    m = mujoco.MjModel.from_xml_path(str(ESCENA))
    m.geom_friction[:, 0] = mu
    return m


class Simulacion(object):
    def __init__(self, model, caminata, x0=0.0, y0=0.0, psi0=0.0, carril_y=0.0):
        self.m, self.cam = model, caminata
        self.d = mujoco.MjData(model)
        self.carril_y = carril_y
        mujoco.mj_resetDataKeyframe(model, self.d, model.key("home").id)
        self.d.qpos[0], self.d.qpos[1] = x0, carril_y + y0
        self.d.qpos[3:7] = [math.cos(psi0 / 2), 0, 0, math.sin(psi0 / 2)]
        mujoco.mj_forward(model, self.d)
        self.act = {model.actuator(i).name: i for i in range(model.nu)}
        self.torso = model.body("torso").id
        self.pie_body = {"L": model.body("LFoot").id, "R": model.body("RFoot").id}
        self.gyro_adr = model.sensor_adr[model.sensor("gyro").id]
        self.t = 0.0
        self._f6 = np.zeros(6)

    # ---------------------------------------------------------------- sensores (solo los del NAO real)
    def sensores(self):
        R = self.d.xmat[self.torso].reshape(3, 3)
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        fsr = {"L": 0.0, "R": 0.0}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for lado, b in self.pie_body.items():
                if self.m.geom_bodyid[c.geom1] == b or self.m.geom_bodyid[c.geom2] == b:
                    mujoco.mj_contactForce(self.m, self.d, i, self._f6)
                    fsr[lado] += abs(self._f6[0]) / 9.81
        return {"gyro": self.d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy(), "angulo": (roll, pitch), "fsr": fsr}

    # ---------------------------------------------------------------- verdad del simulador (solo para medir)
    def pose(self):
        pl, pr = self.d.site("left_foot").xpos, self.d.site("right_foot").xpos
        R = self.d.xmat[self.torso].reshape(3, 3)
        return 0.5 * (pl[0] + pr[0]), 0.5 * (pl[1] + pr[1]) - self.carril_y, math.atan2(R[1, 0], R[0, 0])

    def inclinacion(self):
        R = self.d.xmat[self.torso].reshape(3, 3)
        return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))

    def cayo(self):
        return self.d.xpos[self.torso][2] < 0.2 or self.inclinacion() > 45

    def fuera(self):
        return any(abs(self.d.site(f).xpos[1] - self.carril_y) + FOOT_HALF_WIDTH > LANE_HALF
                   for f in ("left_foot", "right_foot"))

    def meta(self):
        return max(self.d.site("left_foot").xpos[0], self.d.site("right_foot").xpos[0]) >= LENGTH

    # ---------------------------------------------------------------- control
    def aplicar(self, ang, mezcla=1.0):
        home = self.m.key("home").ctrl
        for n, i in self.act.items():
            if n in ang:
                self.d.ctrl[i] = home[i] + mezcla * (ang[n] - home[i])

    def avanzar(self):
        mujoco.mj_step(self.m, self.d, nstep=SUBPASOS)
        self.t += DT

    def postura_inicial(self, segundos=1.0, viewer=None):
        """Pasa de la postura 'home' a la postura de caminar (lo que haria StandInit) y espera quieto."""
        ang = self.cam.postura()
        n = int(segundos / DT)
        for k in range(n):
            self.aplicar(ang, min(1.0, 2.0 * k / n))
            self.avanzar()
            if viewer is not None:
                viewer.sync()


def correr(nombre="naoqi", mu=1.0, vx=0.1, y0=0.0, psi0=0.0, max_t=90.0, vivo=False, vision=None, log=print,
           kw_caminata=None, carril_y=0.0, grabar=False):
    """Corre la caminata hasta la meta, caida, salida del carril o tiempo maximo. Devuelve dict de metricas.
    vision: funcion opcional (sim) -> (vx, vy, wz) que reemplaza el comando fijo (lazo cerrado con la camara)."""
    m = cargar_modelo(mu)
    cam = crear(nombre, m, **(kw_caminata or {}))
    sim = Simulacion(m, cam, y0=y0, psi0=psi0, carril_y=carril_y)
    viewer = None
    if vivo:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, sim.d)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.torso
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 150, -20
    sim.postura_inicial(viewer=viewer)
    t0 = sim.t
    hist = []
    resultado = "tiempo agotado"
    torque2 = 0.0
    qs = []
    while sim.t - t0 < max_t:
        cmd = vision(sim) if vision is not None else (vx, 0.0, 0.0)
        ang = cam.paso(cmd[0], cmd[1], cmd[2], sim.sensores())
        sim.aplicar(ang)
        sim.avanzar()
        torque2 += float(np.sum(sim.d.actuator_force[12:] ** 2)) * DT
        x, y, yaw = sim.pose()
        hist.append((sim.t - t0, x, y, yaw, sim.inclinacion()))
        if grabar:
            qs.append(sim.d.qpos.copy())
        if viewer is not None:
            if not viewer.is_running():
                resultado = "detenido"
                break
            viewer.sync()
            time.sleep(max(0.0, DT * 0.5))
        if sim.cayo():
            resultado = "CAIDA"
            break
        if sim.fuera():
            resultado = "SALIO DEL CARRIL"
            break
        if sim.meta():
            resultado = "META"
            break
    h = np.array(hist) if hist else np.zeros((1, 5))
    T = h[-1, 0]
    info = {"caminata": nombre, "mu": mu, "resultado": resultado, "t": T, "avance": h[-1, 1],
            "vel_media": h[-1, 1] / T if T > 0 else 0.0, "desvio_max": float(np.max(np.abs(h[:, 2]))),
            "desvio_final": float(h[-1, 2]), "rumbo_max_deg": float(np.degrees(np.max(np.abs(h[:, 3])))),
            "inclinacion_max": float(np.max(h[:, 4])), "torque2_por_m": torque2 / max(0.05, h[-1, 1]),
            "pasos": getattr(cam, "pasos_dados", None)}
    if log:
        log("%-10s mu=%.1f -> %-16s t=%5.1f s avance=%.2f m vel=%.3f m/s |desvio| max=%.1f cm rumbo max=%.1f deg"
            % (nombre, mu, resultado, T, info["avance"], info["vel_media"], 100 * info["desvio_max"], info["rumbo_max_deg"]))
    if viewer is not None:
        viewer.close()
    info["hist"] = h
    if grabar:
        info["qpos"] = np.array(qs)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caminata", default="naoqi", choices=("naoqi", "bhuman"))
    ap.add_argument("--vx", type=float, default=None, help="velocidad pedida [m/s] (por defecto la maxima de la caminata)")
    ap.add_argument("--mu", type=float, default=1.0, help="friccion piso-pie (el RL se entreno con 1.0)")
    ap.add_argument("--d0", type=float, default=0.0, help="desvio lateral inicial [m] (+ izquierda)")
    ap.add_argument("--psi0", type=float, default=0.0, help="rumbo inicial [grados] (+ izquierda)")
    ap.add_argument("--max-t", type=float, default=90.0)
    ap.add_argument("--vivo", action="store_true", help="abrir el visor de MuJoCo")
    a = ap.parse_args()
    vx = a.vx if a.vx is not None else {"naoqi": 0.1, "bhuman": 0.2}[a.caminata]
    correr(a.caminata, a.mu, vx, a.d0, math.radians(a.psi0), a.max_t, a.vivo)


if __name__ == "__main__":
    main()
