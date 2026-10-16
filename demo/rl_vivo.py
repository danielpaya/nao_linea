# -*- coding: utf-8 -*-
"""Demo en vivo con las politicas NUEVAS (rama rl-nao-mjx): el mismo panel y la misma pista de la demo de main.

Ventanas (igual que demo/run_demo.py --live):
  - Visor 3D de MuJoCo con la pista (duela, lineas de 1.9 cm, meta roja).
  - "Lo que ve el NAO": la camara inferior, las lineas que detecta la vision y un panel con:
      deslizadores de posicion inicial (corrido y girado), lista de POLITICAS, casilla "Vision de carril" y
      botones Correr / Reiniciar y Detener.

Uso (Windows, desde C:\\demo_carril):
    uv run python -m demo.rl_vivo
    uv run python -m demo.rl_vivo --inicio-d 12 --inicio-psi 8
    uv run python -m demo.rl_vivo --prueba 3                    # sin ventanas: 3 carreras con cada politica
macOS: uv run mjpython -m demo.rl_vivo        Linux sin pantalla (solo --prueba): MUJOCO_GL=egl

Las politicas son las exportadas a numpy con rl/exportar.py (rl/politicas/<gen>_s<semilla>/politica_np.npz):
no hace falta JAX ni GPU. La fisica es MuJoCo normal con el mismo modelo que se uso para entrenar (assets/nao/
scene.xml con solo los pies chocando con el piso, como rl/escena_mjx.py) y los limites reales de los motores.

Vision: la camara inferior simulada ve la pista y el modulo carril/ estima d y psi (el mismo estimador con el que
se midio el ruido que se uso al entrenar). Igual que en el entrenamiento: un cuadro cada 4 pasos de 24 ms (~10 Hz)
y la politica recibe el cuadro anterior (1 cuadro de retraso). Si la vision no encuentra las lineas, se mantiene el
ultimo valor (cuadro perdido). Sin vision (casilla apagada) la politica recibe psi = d = 0.
Las politicas sin vision (v05c, v05d) caminan con su giroscopio: no saben si salieron torcidas.
"""
import argparse
import glob
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
from carril import CameraModel, LaneEstimator, LaneFilter  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene  # noqa: E402

LANE_HALF, LENGTH = 0.35, 5.0
FOOT_HALF_WIDTH = 0.045
CAM_W, CAM_H = 320, 240
HFOV, VFOV = 54.7, 42.5
SIGMA_D, SIGMA_PSI = 0.016, 0.021        # solo para --vision simulada (el ruido del entrenamiento)
CHOCAN = {"floor", "left_foot", "right_foot"}


# ---------------------------------------------------------------------------------------------- modelos
def modelo_fisica():
    """El mismo modelo del entrenamiento: assets/nao/scene.xml con solo pies y piso chocando y solver liviano."""
    spec = mujoco.MjSpec.from_file(str(ROOT / "assets" / "nao" / "scene.xml"))
    for g in spec.geoms:
        if g.name not in CHOCAN and (g.contype or g.conaffinity):
            g.contype, g.conaffinity = 0, 0
    spec.option.iterations = 4
    spec.option.ls_iterations = 8
    m = spec.compile()
    m.opt.timestep = 0.002
    return m


def modelo_dibujo():
    """La pista de la demo (con la camara inferior del NAO). Solo para dibujar y para la camara."""
    return mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "nao" / "scene_carril.xml"))


# ---------------------------------------------------------------------------------------------- politica
class Politica(object):
    def __init__(self, ruta):
        z = np.load(ruta)
        self.meta = json.loads(str(z["meta"]))
        self.media, self.desv = z["media"].astype(np.float64), z["desviacion"].astype(np.float64)
        n = len([k for k in z.files if k.startswith("W")])
        self.capas = [(z["W%d" % i].astype(np.float64), z["b%d" % i].astype(np.float64)) for i in range(n)]
        mt = self.meta
        self.vision = mt["vision"]["activa"]
        gen = mt["generacion"].replace("nao_", "")
        self.nombre = "%s s%s  %.1f m/s  %s" % (gen, mt.get("semilla") or 0, mt["vx"],
                                                "con vision" if self.vision else "sin vision")

    def __call__(self, obs):
        x = (obs - self.media) / self.desv
        for i, (W, b) in enumerate(self.capas):
            x = x @ W + b
            if i < len(self.capas) - 1:
                x = x / (1.0 + np.exp(-x))
        return np.tanh(x[: x.size // 2])


def cargar_politicas():
    rutas = sorted(glob.glob(str(ROOT / "rl" / "politicas" / "*" / "politica_np.npz")))
    pols = [Politica(r) for r in rutas]
    pols.sort(key=lambda p: (-p.meta["vx"], not p.vision, p.meta["generacion"]))   # la mas rapida primero
    return pols


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------------------------------------------------- carrera
class Carrera(object):
    """Un NAO con una politica, paso a paso (24 ms), como NaoCarrera de rl/nao_carrera.py pero en numpy."""

    def __init__(self, mf, pol):
        self.m, self.d, self.pol = mf, mujoco.MjData(mf), pol
        mt = pol.meta
        self.mt = mt
        self.act = np.array([mf.actuator(n).id for n in mt["motores"]])
        jnt = mf.actuator_trnid[self.act, 0]
        self.qadr, self.vadr = mf.jnt_qposadr[jnt], mf.jnt_dofadr[jnt]
        self.postura = np.array(mt["postura"])
        self.lo, self.hi = np.array(mt["lo"]), np.array(mt["hi"])
        self.vel_max, self.par_max = np.array(mt["vel_max"]), np.array(mt["par_max"])
        self.base_fr = mf.actuator_forcerange.copy()
        self.tope = np.minimum(self.base_fr[self.act, 1], self.par_max)
        self.key = mf.keyframe("home").id
        self.ctrl_home = mf.key_ctrl[self.key].copy()
        self.torso = mf.body("torso").id
        self.dt = mt["ctrl_dt"]
        self.nsub = int(round(mt["ctrl_dt"] / mt["sim_dt"]))
        self.n_obs, self.historia = mt["n_obs"], mt["historia"]
        v = mt["vision"]
        self.periodo = v["periodo"]
        self.vision_on = True

    # -- estado
    def colocar(self, d0, psi_deg):
        """Robot en la salida: corrido d0 [m] (+ izquierda) y girado psi_deg [grados] (+ izquierda)."""
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, self.key)
        d.qpos[1] = d0
        h = math.radians(psi_deg) / 2
        d.qpos[3:7] = [math.cos(h), 0.0, 0.0, math.sin(h)]
        d.ctrl[:] = self.ctrl_home
        mujoco.mj_forward(m, d)
        self.t = 0.0

    def de_pie(self, segundos):
        """Quieto en la postura base (la vision mide mientras tanto)."""
        self.d.ctrl[:] = self.ctrl_home
        self._fisica()

    def arrancar(self, medicion):
        """medicion: (psi, d) de la vision al salir (o None). psi = angulo del carril respecto al robot."""
        f = np.mean(self.mt["frecuencia_paso"])
        self.fase = np.array([0.0, math.pi])
        self.dfase = 2 * math.pi * f * self.dt
        self.accion_ant = np.zeros(len(self.act))
        self.rumbo_ref = medicion[0] if (medicion is not None and self.pol.vision) else 0.0
        self.rumbo_est = 0.0
        self.vis_obs = np.array(medicion if medicion is not None else (0.0, 0.0))
        self.vis_muestra = None if medicion is None else np.array(medicion)
        self.vis_cuenta, self.vis_edad, self.t = 0, 0.0, 0.0
        self.estado = self._obs(np.zeros(self.n_obs * self.historia))

    def _gyro_z_mundo(self):
        return (self.d.xmat[self.torso].reshape(3, 3) @ self.d.qvel[3:6])[2]

    def _obs(self, historia):
        d, mt = self.d, self.mt
        actual = [d.qvel[3:6], d.xmat[self.torso].reshape(3, 3)[2, :], d.qpos[self.qadr] - self.postura,
                  d.qvel[self.vadr], self.accion_ant, [mt["vx"], 0.0], [math.cos(self.fase[0]), math.sin(self.fase[0])]]
        em = mt["error_max"]
        if self.pol.vision:
            v = mt["vision"]
            psi, dd = (self.vis_obs if self.vision_on else (0.0, 0.0))
            actual.append([np.clip(psi, -em, em), np.clip(dd, -v["d_max"], v["d_max"])])
            if v["apagones"]:
                actual.append([min(self.vis_edad, v["edad_max"])])
            if v["giroscopio"]:
                actual.append([np.clip(wrap(self.rumbo_ref - self.rumbo_est), -em, em)])
        elif mt["rumbo_activo"]:
            actual.append([np.clip(wrap(self.rumbo_ref - self.rumbo_est), -em, em)])
        actual = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in actual])
        assert actual.size == self.n_obs, (actual.size, self.n_obs)
        return np.concatenate([actual, historia[:-self.n_obs]])         # la mas nueva primero

    def _fisica(self):
        m, d = self.m, self.d
        for _ in range(self.nsub):
            if self.mt["motores_reales"]:
                w = d.qvel[self.vadr]
                lim = self.tope * np.clip(1.0 - np.abs(w) / self.vel_max, 0.0, 1.0)
                m.actuator_forcerange[self.act, 0] = -lim
                m.actuator_forcerange[self.act, 1] = lim
            mujoco.mj_step(m, d)
        m.actuator_forcerange[:] = self.base_fr

    def paso(self, medir):
        """Un paso de 24 ms. medir(): (psi, d) de la camara ahora mismo, o None si no vio las lineas."""
        a = self.pol(self.estado)
        objetivo = np.clip(self.postura + a * self.mt["action_scale"], self.lo, self.hi)
        self.d.ctrl[:] = self.ctrl_home
        self.d.ctrl[self.act] = objetivo
        self._fisica()
        self.accion_ant = a
        self.rumbo_est += self._gyro_z_mundo() * self.dt
        if self.pol.vision:                              # como _paso_vision: publica el cuadro anterior y toma uno
            if self.vis_cuenta <= 0:
                if self.vis_muestra is not None:
                    self.vis_obs, self.vis_edad = self.vis_muestra, 0.0
                else:
                    self.vis_edad += self.dt
                med = medir()
                self.vis_muestra = None if med is None else np.array(med)
                self.vis_cuenta = self.periodo - 1
            else:
                self.vis_cuenta -= 1
                self.vis_edad += self.dt
        f = self.fase + self.dfase
        self.fase = np.fmod(f + math.pi, 2 * math.pi) - math.pi
        self.t += self.dt
        self.estado = self._obs(self.estado)

    # -- medidas
    def pose(self):
        """x, y del torso y rumbo (y = distancia al centro del carril, + izquierda)."""
        R = self.d.xmat[self.torso].reshape(3, 3)
        return self.d.qpos[0], self.d.qpos[1], math.atan2(R[1, 0], R[0, 0])

    def cayo(self):
        arriba = self.d.xmat[self.torso].reshape(3, 3)[2, 2]
        return arriba < math.cos(math.radians(self.mt["inclinacion_max_deg"])) or \
            self.d.xpos[self.torso, 2] < self.mt["altura_min"]

    def fuera(self):
        return any(abs(self.d.site(s).xpos[1]) + FOOT_HALF_WIDTH > LANE_HALF for s in ("left_foot", "right_foot"))

    def meta(self):
        return max(self.d.site("left_foot").xpos[0], self.d.site("right_foot").xpos[0]) >= LENGTH


# ---------------------------------------------------------------------------------------------- vision
class Vision(object):
    """Camara inferior + estimador de carril + filtro (como demo/run_demo.py)."""

    def __init__(self, md, polaridad, pose_noise_deg=1.0, seed=0):
        self.md, self.dd = md, mujoco.MjData(md)
        self.cam_id = md.camera("CameraBottom").id
        self.renderer = mujoco.Renderer(md, CAM_H, CAM_W)
        self.est = LaneEstimator(CameraModel(CAM_W, CAM_H, HFOV, VFOV), polaridad=polaridad)
        self.filt = LaneFilter()
        self.rng = np.random.RandomState(seed)
        self.pose_noise = pose_noise_deg
        self.img, self.pts, self.ultima = None, None, None
        self.torso = md.body("torso").id

    def copiar(self, qpos):
        self.dd.qpos[:] = qpos
        mujoco.mj_forward(self.md, self.dd)

    def _marco(self):
        pl, pr = self.dd.site("left_foot").xpos, self.dd.site("right_foot").xpos
        R = self.dd.xmat[self.torso].reshape(3, 3)
        return np.array([(pl[0] + pr[0]) / 2, (pl[1] + pr[1]) / 2, 0.0]), math.atan2(R[1, 0], R[0, 0])

    def reiniciar(self):
        self.filt.reset()
        self.prev = self._marco()
        self.ultima = None

    def odometria(self):
        origin, yaw = self._marco()
        po, pyaw = self.prev
        dw = origin - po
        c, s = math.cos(pyaw), math.sin(pyaw)
        self.filt.predict(1.1 * (c * dw[0] + s * dw[1]), (-s * dw[0] + c * dw[1]), 0.9 * wrap(yaw - pyaw))
        self.prev = (origin, yaw)

    def _T(self):
        p = self.dd.cam_xpos[self.cam_id]
        Rm = self.dd.cam_xmat[self.cam_id].reshape(3, 3)
        Rn = np.column_stack((-Rm[:, 2], -Rm[:, 0], Rm[:, 1]))
        origin, yaw = self._marco()
        c, s = math.cos(-yaw), math.sin(-yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Rn
        T[:3, 3] = Rz @ (p - origin)
        if self.pose_noise > 0:
            a, b = np.radians(self.rng.normal(0, self.pose_noise, 2))
            ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
            T[:3, :3] = T[:3, :3] @ np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]) @ \
                np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        return T

    def medir(self):
        """(psi, d) en la convencion de la politica (psi = angulo del carril respecto al robot), o None."""
        self.renderer.update_scene(self.dd, camera=self.cam_id)
        self.img = self.renderer.render()
        e = self.est.process(np.ascontiguousarray(self.img[:, :, ::-1]), self._T())
        self.pts = self.est._uv.copy() if len(self.est._uv) else None
        self.filt.update(e)
        if not (e.valid and self.filt.initialized):
            return None
        self.ultima = (-self.filt.psi, self.filt.d)
        return self.ultima


def medicion_simulada(c, rng):
    _, y, yaw = c.pose()
    return (-yaw + SIGMA_PSI * rng.normal(), y + SIGMA_D * rng.normal())


# ---------------------------------------------------------------------------------------------- panel
def hacer_panel(pols, d0, psi0, vision):
    from run_demo import CameraWindow, Desplegable

    class PanelRL(CameraWindow):
        def _build_controls(self):
            from matplotlib.widgets import Button, CheckButtons, Slider
            f = self.fig
            self.s_d = Slider(f.add_axes([0.33, 0.285, 0.52, 0.032]), "Inicio: corrido [cm]\n(+ izquierda)", -25, 25,
                              valinit=self.d0, valstep=1)
            self.s_p = Slider(f.add_axes([0.33, 0.230, 0.52, 0.032]), "Inicio: girado [grados]\n(+ izquierda)",
                              -15, 15, valinit=self.psi0, valstep=1)
            nombres = [p.nombre for p in pols]
            self.politica = nombres[0]
            self.d_pol = Desplegable(f, [0.20, 0.150, 0.77, 0.042], "Politica", nombres, nombres[0],
                                     al_cambiar=self._cambio_politica)
            self.c_v = CheckButtons(f.add_axes([0.03, 0.03, 0.30, 0.09]), ["Vision de carril"], [self.vision])
            self.b_run = Button(f.add_axes([0.38, 0.035, 0.28, 0.08]), "Correr / Reiniciar", color="#c8e6c9",
                                hovercolor="#a5d6a7")
            self.b_stop = Button(f.add_axes([0.69, 0.035, 0.27, 0.08]), "Detener", color="#ffcdd2",
                                 hovercolor="#ef9a9a")

            def on_d(v):
                self.d0 = float(v)
                self.dirty = True

            def on_p(v):
                self.psi0 = float(v)
                self.dirty = True

            def on_v(_label):
                self.vision = bool(self.c_v.get_status()[0])
                self.dirty = True

            self.s_d.on_changed(on_d)
            self.s_p.on_changed(on_p)
            self.c_v.on_clicked(on_v)
            self.b_run.on_clicked(lambda _e: setattr(self, "request", "run"))
            self.b_stop.on_clicked(lambda _e: setattr(self, "request", "stop"))

        def _cambio_politica(self, nombre):
            self.politica = nombre
            self.dirty = True

    return PanelRL(controls=True, d0=d0, psi0=psi0, vision=vision)


# ---------------------------------------------------------------------------------------------- carrera en vivo
def correr(c, vis, panel=None, viewer=None, modo_vision="camara", seed=0, log=print):
    rng = np.random.RandomState(seed)
    vis.copiar(c.d.qpos)
    vis.reiniciar()
    medir_cam = (lambda: (vis.copiar(c.d.qpos), vis.medir())[1])
    medir = medir_cam if modo_vision == "camara" else (lambda: medicion_simulada(c, rng))
    ph_med, ph_run = "1. MIDIENDO su posicion (de pie)", "2. CARRERA"
    if not c.pol.vision:
        ph_med = ph_run = "SIN VISION: camina con su giroscopio"
    elif not c.vision_on:
        ph_med = ph_run = "VISION APAGADA: el robot no sabe donde esta"
    detenido = [False]

    def mostrar(k, fase, forzar=False):
        if viewer is not None:
            viewer.sync()
        if panel is not None and (forzar or k % 8 == 0):
            _, y, yaw = c.pose()
            txt = "%s\nt = %4.1f s   real: d = %+5.1f cm  psi = %+5.1f deg" % (c.pol.nombre, c.t, 100 * y, math.degrees(yaw))
            if c.pol.vision and vis.ultima is not None and modo_vision == "camara":
                txt += "\nvision:  d = %+5.1f cm  psi = %+5.1f deg" % (100 * vis.ultima[1], -math.degrees(vis.ultima[0]))
            panel.update(vis.img if vis.img is not None else np.zeros((CAM_H, CAM_W, 3), np.uint8), vis.pts, fase, txt)
            if panel.request is not None:
                detenido[0] = True

    # fase 1: de pie ~1 s, la vision mide donde quedo
    med = None
    for k in range(int(round(1.0 / c.dt))):
        c.de_pie(c.dt)
        vis.copiar(c.d.qpos)
        if c.pol.vision and k % c.periodo == 0:
            med = medir()
        mostrar(k, ph_med)
        if viewer is not None:
            time.sleep(c.dt)
        if detenido[0] or (viewer is not None and not viewer.is_running()):
            return dict(resultado="DETENIDO", t=0.0)
    c.arrancar(med if c.pol.vision else None)
    resultado = "TIEMPO"
    for k in range(int(40.0 / c.dt)):
        t0 = time.time()
        c.paso(medir)
        vis.copiar(c.d.qpos)
        vis.odometria()
        mostrar(k, ph_run)
        if detenido[0] or (viewer is not None and not viewer.is_running()):
            resultado = "DETENIDO"
            break
        if c.cayo():
            resultado = "CAIDA"
            break
        if c.fuera():
            resultado = "SALIO DEL CARRIL"
            break
        if c.meta():
            resultado = "META"
            break
        if viewer is not None:
            time.sleep(max(0.0, c.dt - (time.time() - t0)))          # tiempo real
    _, y, yaw = c.pose()
    info = dict(resultado=resultado, t=c.t, x=float(c.d.qpos[0]), d_fin=100 * y)
    log("%-32s -> %-16s t=%5.1f s  avance=%.2f m  d final=%+5.1f cm" % (c.pol.nombre, resultado, c.t, info["x"], 100 * y))
    if panel is not None:
        mostrar(0, ph_run, forzar=True)
        msg = {"META": "LLEGO A LA META en %.1f s" % c.t, "SALIO DEL CARRIL": "SE SALIO DEL CARRIL",
               "CAIDA": "SE CAYO", "DETENIDO": "Detenido"}.get(resultado, resultado)
        panel.set_title(msg, "#0b8043" if resultado == "META" else "#c5221f")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inicio-d", type=float, default=0.0, help="corrido a la izquierda (+) o derecha (-) [cm]")
    ap.add_argument("--inicio-psi", type=float, default=0.0, help="girado a la izquierda (+) o derecha (-) [grados]")
    ap.add_argument("--sin-vision", action="store_true")
    ap.add_argument("--vision", choices=("camara", "simulada"), default="camara",
                    help="camara: estimador de carril sobre la imagen (por defecto); simulada: verdad + ruido")
    ap.add_argument("--prueba", type=int, default=0, help="sin ventanas: N carreras con cada politica")
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()

    pols = cargar_politicas()
    if not pols:
        sys.exit("No hay politicas exportadas en rl/politicas/*/politica_np.npz (correr rl/exportar.py en WSL).")
    info_pista = build_scene.build(build_scene.load_config(), verbose=False)
    mf, md = modelo_fisica(), modelo_dibujo()
    vis = Vision(md, info_pista["polaridad"], seed=a.seed)
    carreras = {p.nombre: Carrera(mf, p) for p in pols}
    print("Politicas: " + " | ".join(p.nombre for p in pols))

    if a.prueba:
        for p in pols:
            for i in range(a.prueba):
                c = carreras[p.nombre]
                c.vision_on = not a.sin_vision
                c.colocar(a.inicio_d / 100.0, a.inicio_psi)
                correr(c, vis, modo_vision=a.vision, seed=a.seed + i)
        return

    from mujoco import viewer as mj_viewer
    panel = hacer_panel(pols, a.inicio_d, a.inicio_psi, not a.sin_vision)
    dd = vis.dd
    viewer = mj_viewer.launch_passive(md, dd)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = md.body("torso").id
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 150, -25
    print("Usa el panel de 'Lo que ve el NAO': posicion inicial, politica, vision y 'Correr / Reiniciar'. "
          "Cierra cualquiera de las ventanas para salir.")
    while viewer.is_running() and panel.is_open():
        c = carreras[panel.politica]
        if panel.request == "run":
            panel.request = None
            c.vision_on = panel.vision
            c.colocar(panel.d0 / 100.0, panel.psi0)
            with viewer.lock():
                pass
            correr(c, vis, panel=panel, viewer=viewer, modo_vision=a.vision, seed=a.seed)
            panel.dirty = False
            continue
        if panel.request == "stop":
            panel.request = None
        if panel.dirty:                                   # vista previa: de pie en la nueva posicion
            panel.dirty = False
            c.colocar(panel.d0 / 100.0, panel.psi0)
            vis.copiar(c.d.qpos)
            vis.renderer.update_scene(vis.dd, camera=vis.cam_id)
            panel.update(vis.renderer.render(), None, "Listo: ajusta el inicio y presiona Correr",
                         "%s\nposicion inicial: d = %+3.0f cm   psi = %+3.0f deg   vision: %s"
                         % (c.pol.nombre, panel.d0, panel.psi0,
                            ("SI" if panel.vision else "NO") if c.pol.vision else "no tiene"))
        viewer.sync()
        if panel.interactive:
            panel.plt.pause(0.05)
        else:
            time.sleep(0.05)
    viewer.close()


if __name__ == "__main__":
    main()
