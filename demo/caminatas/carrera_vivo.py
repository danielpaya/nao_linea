# -*- coding: utf-8 -*-
"""Carrera EN VIVO en el visor de MuJoCo: los 3 NAOs a la vez, cada uno en su carril.

    uv run python -m demo.caminatas.carrera_vivo                 # con vision
    uv run python -m demo.caminatas.carrera_vivo --sin-vision    # sin vision (se ve como se desvian)
    uv run python -m demo.caminatas.carrera_vivo --mu 0.6        # piso mas resbaloso

Carril izquierdo: tipo B-Human · centro: RL del equipo · derecho: tipo NAOqi (izquierda/derecha del robot).

Como funciona: cada robot se simula en su propia fisica (no se tocan, igual que en la carrera real, donde
corren por separado) y en el visor se dibujan los tres juntos en una sola escena. La vision es la simulada
con el error medido (sigma 1.6 cm y 1.2 deg, 10 Hz), la misma del benchmark.

En el visor: ESPACIO pausa, BACKSPACE no hace nada (para reiniciar, cierra y vuelve a correr). Con el
raton se mueve la camara; doble clic en un robot y Ctrl+clic derecho para seguirlo.
macOS: usar `uv run mjpython -m demo.caminatas.carrera_vivo`.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CARRILES = {"bhuman": 0.719, "rl": 0.0, "naoqi": -0.719}     # centro de cada carril [m] (izquierda +)
PREFIJO = {"rl": "", "bhuman": "B_", "naoqi": "N_"}
NOMBRES = {"bhuman": "tipo B-Human (carril izq.)", "rl": "RL del equipo (centro)", "naoqi": "tipo NAOqi (carril der.)"}


# ---------------------------------------------------------------------------- un robot por fisica
class CorredorClasico(object):
    def __init__(self, nombre, mu, seed, vision):
        from demo.caminatas.benchmark import VX, VisionSimulada, inicio
        from demo.caminatas.simular import Simulacion, cargar_modelo, crear
        self.nombre = nombre
        m = cargar_modelo(mu)
        self.cam = crear(nombre, m)
        d0, psi0 = inicio(seed)
        self.sim = Simulacion(m, self.cam, y0=d0, psi0=psi0)
        self.sim.postura_inicial()
        self.t0 = self.sim.t
        self.vx = VX[nombre]
        self.vis = VisionSimulada(np.random.RandomState(seed + 7)) if vision else None
        self.resultado, self.t_fin = None, None

    @property
    def t(self):
        return self.sim.t - self.t0

    @property
    def qpos(self):
        return self.sim.d.qpos

    def paso(self):
        x, y, yaw = self.sim.pose()
        wz = self.vis.giro(self.sim.t, y, yaw) if self.vis is not None else 0.0
        ang = self.cam.paso(self.vx, 0.0, wz, self.sim.sensores())
        self.sim.aplicar(ang)
        self.sim.avanzar()
        if self.sim.cayo():
            self.resultado = "CAIDA"
        elif self.sim.fuera():
            self.resultado = "SALIO DEL CARRIL"
        elif self.sim.meta():
            self.resultado = "META"
        if self.resultado:
            self.t_fin = self.t

    def pose(self):
        return self.sim.pose()


class CorredorRL(object):
    def __init__(self, mu, seed, vision):
        import pickle

        import mujoco
        import yaml
        from stable_baselines3 import PPO
        import src.envs.nao_walk_env as E
        from demo.caminatas.benchmark import VisionSimulada, inicio
        E.SCENE_PATH = ROOT / "assets" / "nao" / "scene_carril.xml"
        cfg = yaml.safe_load(open(ROOT / "checkpoints/best_model_config.yaml"))
        self.model = PPO.load(ROOT / "checkpoints/best_model.zip", device="cpu")
        self.vn = pickle.load(open(ROOT / "checkpoints/best_model_vecnormalize.pkl", "rb"))
        self.env = E.NaoWalkEnv(**cfg["env"])
        self.obs, _ = self.env.reset(seed=seed)
        m, d = self.env.model, self.env.data
        m.geom_friction[:, 0] = mu
        d0, psi0 = inicio(seed)
        d.qpos[1] = d0
        d.qpos[3:7] = [math.cos(psi0 / 2), 0, 0, math.sin(psi0 / 2)]
        mujoco.mj_forward(m, d)
        for _ in range(int(1.0 / self.env.dt)):
            d.ctrl[:] = self.env._home_ctrl
            mujoco.mj_step(m, d, nstep=self.env.frame_skip)
        self.env._step_count = 0
        self.env._prev_action[:] = 0.0
        self.obs = self.env._get_obs()
        self.vis = VisionSimulada(np.random.RandomState(seed + 7)) if vision else None
        self.k = 0
        self.nombre = "rl"
        self.resultado, self.t_fin = None, None

    @property
    def t(self):
        return self.k * self.env.dt

    @property
    def qpos(self):
        return self.env.data.qpos

    def pose(self):
        d = self.env.data
        pl, pr = d.site("left_foot").xpos, d.site("right_foot").xpos
        R = d.xmat[self.env._torso_id].reshape(3, 3)
        return 0.5 * (pl[0] + pr[0]), 0.5 * (pl[1] + pr[1]), math.atan2(R[1, 0], R[0, 0])

    def paso(self):
        from demo.caminatas.benchmark import GYRO_OBS, MAX_BIAS, TURN_GAIN
        from demo.caminatas.simular import FOOT_HALF_WIDTH, LANE_HALF, LENGTH
        x, y, yaw = self.pose()
        o = self.obs.copy()
        if self.vis is not None:
            o[GYRO_OBS] += float(np.clip(-self.vis.giro(self.t, y, yaw) / TURN_GAIN, -MAX_BIAS, MAX_BIAS))
        action, _ = self.model.predict(self.vn.normalize_obs(o), deterministic=True)
        self.obs, _, terminated, _, _ = self.env.step(action)
        self.k += 1
        d = self.env.data
        R = d.xmat[self.env._torso_id].reshape(3, 3)
        if terminated or d.xpos[self.env._torso_id][2] < 0.2 or R[2, 2] < math.cos(math.radians(45)):
            self.resultado = "CAIDA"
        elif any(abs(d.site(f).xpos[1]) + FOOT_HALF_WIDTH > LANE_HALF for f in ("left_foot", "right_foot")):
            self.resultado = "SALIO DEL CARRIL"
        elif max(d.site("left_foot").xpos[0], d.site("right_foot").xpos[0]) >= LENGTH:
            self.resultado = "META"
        if self.resultado:
            self.t_fin = self.t


# ---------------------------------------------------------------------------- escena con los 3 robots
def escena_tres(mujoco):
    """Pista con 3 NAOs (solo para dibujar: nunca se simula, se copian las poses de cada fisica)."""
    esc = mujoco.MjSpec.from_file(str(ROOT / "assets" / "nao" / "scene_carril.xml"))
    for nombre in ("bhuman", "naoqi"):
        robot = mujoco.MjSpec.from_file(str(ROOT / "assets" / "nao" / "nao_cam.xml"))
        for k in list(robot.keys):
            robot.delete(k)
        esc.worldbody.add_frame().attach_body(robot.body("torso"), PREFIJO[nombre], "")
    for k in list(esc.keys):
        esc.delete(k)
    return esc.compile()


def mapa_qpos(mujoco, m_uno, m_tres, prefijo):
    """Indices de qpos de un robot en la escena de 3 (por nombre de articulacion)."""
    origen, destino = [], []
    for j in range(m_uno.njnt):
        nom = m_uno.joint(j).name
        n = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(int(m_uno.jnt_type[j]), 1)
        a = m_uno.jnt_qposadr[j]
        b = m_tres.joint(prefijo + nom).qposadr[0]
        origen += range(a, a + n)
        destino += range(b, b + n)
    return np.array(origen), np.array(destino)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-vision", action="store_true")
    ap.add_argument("--mu", type=float, default=1.0, help="friccion piso-pie (el RL se entreno con 1.0)")
    ap.add_argument("--semilla", type=int, default=1000, help="cambia el error de colocacion en la salida")
    ap.add_argument("--velocidad", type=float, default=1.0, help="1 = tiempo real, 2 = el doble de rapido")
    ap.add_argument("--captura", default=None, help="(pruebas) sin visor: guarda una imagen a los 8 s y termina")
    a = ap.parse_args()
    vision = not a.sin_vision

    print("Preparando los 3 robots...")
    corredores = [CorredorRL(a.mu, a.semilla, vision)]          # primero el RL: importa torch antes que OpenGL
    import mujoco
    corredores += [CorredorClasico(n, a.mu, a.semilla, vision) for n in ("bhuman", "naoqi")]
    m = escena_tres(mujoco)
    d = mujoco.MjData(m)
    mapas = {c.nombre: mapa_qpos(mujoco, (c.env.model if c.nombre == "rl" else c.sim.m), m, PREFIJO[c.nombre])
             for c in corredores}

    def dibujar():
        for c in corredores:
            o, de = mapas[c.nombre]
            d.qpos[de] = c.qpos[o]
            d.qpos[de[1]] += CARRILES[c.nombre]            # cada uno en su carril
        mujoco.mj_forward(m, d)

    def avanzar_hasta(t):
        for c in corredores:
            while c.resultado is None and c.t < t:
                c.paso()

    def textos(t):
        izq = ["Carrera 5 m", "vision", "friccion", "tiempo", ""]
        der = ["", "SI" if vision else "NO", "%.1f" % a.mu, "%.1f s" % t, ""]
        for c in sorted(corredores, key=lambda c: list(CARRILES).index(c.nombre)):
            izq.append(NOMBRES[c.nombre])
            der.append("%s %.1f s" % (c.resultado, c.t_fin) if c.resultado else "%.2f m" % c.pose()[0])
        return "\n".join(izq), "\n".join(der)

    dibujar()
    if a.captura:                                              # modo de prueba sin pantalla
        avanzar_hasta(8.0)
        dibujar()
        r = mujoco.Renderer(m, 540, 960)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 4.6, 20, -30
        cam.lookat[:] = [2.0, 0, 0.1]
        r.update_scene(d, camera=cam)
        from PIL import Image
        Image.fromarray(r.render()).save(a.captura)
        print(textos(8.0)[1])
        return

    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [1.2, 0.0, 0.15]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 4.6, 20, -28
        print("Listo: corriendo. Cierra la ventana para terminar.")
        t_sim, ultimo = 0.0, time.time()
        while v.is_running():
            ahora = time.time()
            t_sim += min(0.1, ahora - ultimo) * a.velocidad
            ultimo = ahora
            avanzar_hasta(t_sim)
            dibujar()
            xs = [c.pose()[0] for c in corredores]
            v.cam.lookat[0] += 0.05 * (min(4.2, max(1.2, 0.5 * (min(xs) + max(xs)) + 0.3)) - v.cam.lookat[0])
            try:
                v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPRIGHT) + textos(t_sim))
            except Exception:  # noqa: BLE001  (versiones viejas del visor no tienen textos)
                pass
            v.sync()
            if all(c.resultado for c in corredores) and t_sim > max(c.t_fin for c in corredores) + 3:
                pass                                           # se queda mostrando el final hasta cerrar
            time.sleep(max(0.0, 1.0 / 60 - (time.time() - ahora)))
        for c in corredores:
            print("%-26s %s" % (NOMBRES[c.nombre], ("%s en %.1f s" % (c.resultado, c.t_fin)) if c.resultado else "sin terminar"))


if __name__ == "__main__":
    main()
