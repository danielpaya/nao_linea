# -*- coding: utf-8 -*-
"""Demo: NAO V6 caminando con la politica RL (PPO) en la pista de la Carrera, con vision de carril.

Compara dos corridas sobre la misma pista de 5 m (carril de 70 cm, duela, meta roja):
  1. "Solo RL": la politica camina lo mas recto que puede, sin ver el carril.
  2. "RL + vision": la camara inferior del NAO detecta las lineas, estima d y psi, y una capa de control
     corrige el rumbo. La politica no se reentreno: se le inyecta una senal de giro en la observacion del
     giroscopio y ella misma la compensa (aprendio a anular giros), lo que la hace girar hacia el centro.

Uso:
    uv run python demo/run_demo.py --video demo.mp4        # graba las dos corridas (recomendado)
    uv run python demo/run_demo.py --live                  # visor 3D en vivo (corrida con vision)
    uv run python demo/run_demo.py --live --sin-vision     # visor 3D en vivo (solo RL)
    uv run python demo/run_demo.py --live --inicio-d 12 --inicio-psi 8
        # colocado chueco: 12 cm a la izquierda y girado 8 grados; mide, se centra y corre
    uv run python demo/run_demo.py --live --piso gris_oscuro --linea blanco
        # otros colores (tambien en demo/pista.yaml o desde el panel)
En Linux sin pantalla:  MUJOCO_GL=egl (u osmesa) uv run python demo/run_demo.py --video demo.mp4
"""
import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO  # noqa: E402  (importar torch antes que el contexto OpenGL de MuJoCo)
import mujoco  # noqa: E402
import src.envs.nao_walk_env as E  # noqa: E402
from carril import CameraModel, LaneEstimator, LaneFilter  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene  # noqa: E402  (colores de la pista: demo/pista.yaml)

E.SCENE_PATH = ROOT / "assets" / "nao" / "scene_carril.xml"
LANE_HALF, LENGTH = 0.35, 5.0
FOOT_HALF_WIDTH = 0.045          # mitad del ancho de la planta del pie [m]
GYRO_OBS = 5                     # indice de la observacion "giro alrededor de z" (giroscopio * 0.25)
TURN_GAIN = 0.215                # rad/s de giro que produce +1 de sesgo en esa observacion (medido)
CAM_W, CAM_H = 320, 240
HFOV, VFOV = 54.7, 42.5          # camara inferior del modelo B-Human
PISTA = {}                       # colores y polaridad de la pista generada (ver build_scene.describe)


def load_policy():

    cfg = yaml.safe_load(open(ROOT / "checkpoints/best_model_config.yaml"))
    model = PPO.load(ROOT / "checkpoints/best_model.zip", device="cpu")
    with open(ROOT / "checkpoints/best_model_vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    return cfg, model, vn


def torso_yaw(env):
    m = env.data.xmat[env._torso_id].reshape(3, 3)
    return math.atan2(m[1, 0], m[0, 0])


def robot_frame(env):
    """Origen entre los pies en el suelo y rumbo del torso (como FRAME_ROBOT de NAOqi)."""
    pl, pr = env.data.site("left_foot").xpos, env.data.site("right_foot").xpos
    return np.array([(pl[0] + pr[0]) / 2, (pl[1] + pr[1]) / 2, 0.0]), torso_yaw(env)


def camera_T(env, cam_id, rng, pose_noise_deg):
    """Pose camara -> marco del robot en la convencion de NAOqi (x eje optico, y izquierda, z arriba)."""
    p = env.data.cam_xpos[cam_id]
    Rm = env.data.cam_xmat[cam_id].reshape(3, 3)          # columnas: ejes x, y, z de la camara MuJoCo
    Rn = np.column_stack((-Rm[:, 2], -Rm[:, 0], Rm[:, 1]))
    origin, yaw = robot_frame(env)
    c, s = math.cos(-yaw), math.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Rn
    T[:3, 3] = Rz @ (p - origin)
    if pose_noise_deg > 0:  # error de la pose que reportaria el robot real
        a, b = np.radians(rng.normal(0, pose_noise_deg, 2))
        ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
        T[:3, :3] = T[:3, :3] @ np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]) @ np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    return T


def foot_out(env):
    for f in ("left_foot", "right_foot"):
        if abs(env.data.site(f).xpos[1]) + FOOT_HALF_WIDTH > LANE_HALF:
            return True
    return False


class Desplegable(object):
    """Lista desplegable para matplotlib (no trae una): un boton con el valor actual y una flecha.
    Al hacer clic se abre la lista encima del boton; al elegir una opcion se cierra."""

    def __init__(self, fig, rect, etiqueta, opciones, inicial, al_cambiar=None, max_visibles=10):
        from matplotlib.widgets import Button
        self.fig, self.rect, self.al_cambiar = fig, rect, al_cambiar
        self.opciones = list(opciones)
        if inicial not in self.opciones:       # p. ej. un "#rrggbb" puesto en pista.yaml
            self.opciones.insert(0, inicial)
        self.opciones = self.opciones[:max_visibles + 1]
        self.valor = inicial
        fig.text(rect[0] - 0.01, rect[1] + rect[3] / 2, etiqueta, ha="right", va="center", fontsize=10)
        self.boton = Button(fig.add_axes(rect), self._texto(), color="#f1f3f4", hovercolor="#e0e0e0")
        self.boton.label.set_fontsize(10)
        self.boton.on_clicked(lambda _e: self.alternar())
        self.lista_ax = None
        self.radio = None

    def _texto(self):
        return "%s   \u25bc" % self.valor

    def alternar(self):
        if self.lista_ax is not None:
            self.cerrar()
            return
        from matplotlib.widgets import RadioButtons
        x, y, w, h = self.rect
        alto = 0.032 * len(self.opciones)
        self.lista_ax = self.fig.add_axes([x, y + h + 0.005, w, alto], facecolor="white", zorder=10)
        for lado in self.lista_ax.spines.values():
            lado.set_edgecolor("#9aa0a6")
        self.radio = RadioButtons(self.lista_ax, self.opciones, active=self.opciones.index(self.valor))
        for t in self.radio.labels:
            t.set_fontsize(9)
        self.radio.on_clicked(self.elegir)
        self.fig.canvas.draw_idle()

    def elegir(self, opcion):
        self.valor = opcion
        self.boton.label.set_text(self._texto())
        self.cerrar()
        if self.al_cambiar is not None:
            self.al_cambiar(opcion)

    def cerrar(self):
        if self.lista_ax is not None:
            self.radio.disconnect_events()
            self.lista_ax.remove()
            self.lista_ax, self.radio = None, None
            self.fig.canvas.draw_idle()


class CameraWindow(object):
    """Ventana aparte (matplotlib) con lo que ve la camara inferior del NAO y lo que detecta la vision.
    Con controls=True agrega un panel: posicion inicial, vision si/no y botones Correr / Detener."""

    def __init__(self, controls=False, d0=0.0, psi0=0.0, vision=True, piso="madera_clara", linea="negro"):
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt
        self.interactive = matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo")
        plt.ion()
        self.request = None      # "run" o "stop" (lo leen la demo y la corrida en curso)
        self.dirty = True        # la posicion inicial cambio: actualizar la vista previa
        self.d0, self.psi0, self.vision = d0, psi0, vision
        self.piso, self.linea = str(piso), str(linea)
        if controls:
            self.fig = plt.figure(figsize=(6.8, 8.4))
            self.ax = self.fig.add_axes([0.02, 0.43, 0.96, 0.51])
        else:
            self.fig, self.ax = plt.subplots(figsize=(6.4, 5.6))
        try:
            self.fig.canvas.manager.set_window_title("Lo que ve el NAO (camara inferior)")
        except Exception:  # noqa: BLE001
            pass
        self.im = self.ax.imshow(np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8))
        self.sc = self.ax.scatter([], [], s=10, c="#00e5ff")
        self.ax.set_xlim(0, CAM_W)
        self.ax.set_ylim(CAM_H, 0)
        self.ax.axis("off")
        self.title = self.ax.set_title("", fontsize=12, fontweight="bold")
        if controls:
            self.info = self.fig.text(0.04, 0.355, "", fontsize=10, family="monospace")
            self._build_controls()
        else:
            self.info = self.fig.text(0.03, 0.03, "", fontsize=11, family="monospace")
            self.fig.tight_layout(rect=(0, 0.1, 1, 1))

    def _build_controls(self):
        from matplotlib.widgets import Button, CheckButtons, Slider
        f = self.fig
        self.s_d = Slider(f.add_axes([0.33, 0.285, 0.52, 0.032]), "Inicio: corrido [cm]\n(+ izquierda)", -25, 25,
                          valinit=self.d0, valstep=1)
        self.s_p = Slider(f.add_axes([0.33, 0.230, 0.52, 0.032]), "Inicio: girado [grados]\n(+ izquierda)", -15, 15,
                          valinit=self.psi0, valstep=1)
        # Colores: nombre (madera_oscura, gris, blanco...) o codigo "#rrggbb"; ver demo/pista.yaml
        # Colores: listas desplegables con los nombres de build_scene (mas colores HEX en demo/pista.yaml)
        self.d_piso = Desplegable(f, [0.12, 0.150, 0.27, 0.042], "Piso", list(build_scene.PISOS), self.piso)
        self.d_linea = Desplegable(f, [0.49, 0.150, 0.21, 0.042], "Linea", list(build_scene.LINEAS), self.linea)
        self.b_col = Button(f.add_axes([0.73, 0.145, 0.24, 0.052]), "Aplicar colores", color="#e3f2fd", hovercolor="#bbdefb")
        self.b_col.on_clicked(lambda _e: self._apply_colors())
        self.c_v = CheckButtons(f.add_axes([0.03, 0.03, 0.30, 0.09]), ["Vision de carril"], [self.vision])
        self.b_run = Button(f.add_axes([0.38, 0.035, 0.28, 0.08]), "Correr / Reiniciar", color="#c8e6c9", hovercolor="#a5d6a7")
        self.b_stop = Button(f.add_axes([0.69, 0.035, 0.27, 0.08]), "Detener", color="#ffcdd2", hovercolor="#ef9a9a")

        def on_d(v):
            self.d0 = float(v)
            self.dirty = True

        def on_p(v):
            self.psi0 = float(v)
            self.dirty = True

        def on_v(_label):
            self.vision = bool(self.c_v.get_status()[0])

        self.s_d.on_changed(on_d)
        self.s_p.on_changed(on_p)
        self.c_v.on_clicked(on_v)
        self.b_run.on_clicked(lambda _e: setattr(self, "request", "run"))
        self.b_stop.on_clicked(lambda _e: setattr(self, "request", "stop"))

    def _apply_colors(self):
        self.d_piso.cerrar()
        self.d_linea.cerrar()
        self.piso, self.linea = self.d_piso.valor, self.d_linea.valor
        self.request = "colores"

    def update(self, img, pts, phase, text):
        self.im.set_data(img)
        self.sc.set_offsets(pts if pts is not None and len(pts) else np.zeros((0, 2)))
        self.title.set_text(phase)
        self.info.set_text(text)
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)

    def set_title(self, text, color="black"):
        self.title.set_text(text)
        self.title.set_color(color)
        if self.interactive:
            self.fig.canvas.draw_idle()

    def is_open(self):
        return (not self.interactive) or self.plt.fignum_exists(self.fig.number)


def place_robot(env, d0, psi0_deg):
    """Coloca al robot en la salida corrido d0 [m] (+ izquierda) y girado psi0 [grados] (+ izquierda)."""
    q = env.data.qpos
    q[1] = d0
    half = math.radians(psi0_deg) / 2
    q[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
    mujoco.mj_forward(env.model, env.data)


def run(vision, seed=1000, video=None, live=False, pose_noise_deg=1.0, k_psi=1.2, k_d=2.0, max_bias=0.35, max_bias_corr=0.35,
        start_d=0.0, start_psi_deg=0.0, measure_s=1.5, show_cam=False, log=print, ctx=None, polaridad=None):
    """ctx: si se pasa (modo interactivo), reutiliza env, politica, visor, ventana y renderer ya creados."""
    if ctx is None:
        cfg, model, vn = load_policy()
        env = E.NaoWalkEnv(**cfg["env"])
    else:
        cfg, model, vn = ctx["policy"]
        env = ctx["env"]
    rng = np.random.RandomState(seed)
    obs, _ = env.reset(seed=seed)
    place_robot(env, start_d, start_psi_deg)
    m, d = env.model, env.data
    cam_id = m.camera("CameraBottom").id
    cam = CameraModel(CAM_W, CAM_H, HFOV, VFOV)
    if polaridad is None:   # linea mas oscura o mas clara que el piso, segun los colores de la pista
        polaridad = ctx["polaridad"] if ctx is not None and "polaridad" in ctx else PISTA.get("polaridad", "oscura")
    est = LaneEstimator(cam, polaridad=polaridad)
    filt = LaneFilter()
    if ctx is not None:
        cam_renderer, cam_win, viewer = ctx["renderer"], ctx["cam_win"], ctx["viewer"]
        if cam_win is not None:
            cam_win.set_title("")
    else:
        cam_renderer = mujoco.Renderer(m, CAM_H, CAM_W) if (vision or video or show_cam) else None
        cam_win = CameraWindow() if show_cam else None
        viewer = None
    if live and viewer is None:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env._torso_id
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 150, -25
    bias = 0.0
    state = dict(e=None)
    hist = dict(x=[], d_true=[], d_est=[], psi_true=[], psi_est=[], bias=[])
    frames = []
    result, t = "tiempo agotado", 0.0
    PH_MEAS, PH_CORR, PH_RUN = "1. MIDIENDO su posicion (de pie)", "2. CORRIGIENDO hacia el centro", "3. CARRERA"
    PH_BLIND = "SOLO RL: sin vision"

    def sense(k, t, phase):
        """Camara + vision cada 3 pasos de 20 ms (~16.7 Hz)."""
        if cam_renderer is None or k % 3:
            return
        cam_renderer.update_scene(d, camera=cam_id)
        img = cam_renderer.render()
        e = None
        if vision:
            T = camera_T(env, cam_id, rng, pose_noise_deg)
            e = est.process(np.ascontiguousarray(img[:, :, ::-1]), T)
            filt.update(e)
            state["e"] = e
        pts = est._uv.copy() if (vision and len(est._uv)) else None
        if cam_win is not None:
            origin, yaw = robot_frame(env)
            txt = "posicion real   : d = %+5.1f cm   psi = %+5.1f deg\n" % (100 * origin[1], math.degrees(yaw))
            if vision and filt.initialized:
                txt += "estimada (vision): d = %+5.1f cm   psi = %+5.1f deg" % (100 * filt.d, math.degrees(filt.psi))
            else:
                txt += "vision apagada: el robot no sabe donde esta"
            cam_win.update(img, pts, phase, txt)
        if video is not None:
            frames.append(make_frame_data(env, img, pts, t, phase))

    interrupted = [False]

    def show():
        if ctx is not None and cam_win is not None and cam_win.request is not None:
            interrupted[0] = True      # el usuario pidio Detener o Reiniciar
            return False
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, env.dt - 0.002))
            return viewer.is_running()
        return True

    # ---------------- fase 1: de pie en la salida, la vision mide donde quedo colocado
    n_meas = int(round(measure_s / env.dt))
    home_ctrl = env._home_ctrl.copy()
    for k in range(n_meas):
        sense(k, (k - n_meas) * env.dt, PH_MEAS if vision else PH_BLIND)
        d.ctrl[:] = home_ctrl
        mujoco.mj_step(m, d, nstep=env.frame_skip)
        if not show():
            break
    if vision and filt.initialized:
        log("Medicion inicial (de pie): d=%+.1f cm psi=%+.1f deg   | real: d=%+.1f cm psi=%+.1f deg"
            % (100 * filt.d, math.degrees(filt.psi), 100 * start_d, start_psi_deg))
    env._step_count = 0          # el reloj de marcha empieza al arrancar
    env._prev_action[:] = 0.0
    obs = env._get_obs()
    prev_origin, prev_yaw = robot_frame(env)
    correcting = vision
    calm = 0

    # ---------------- fases 2 y 3: caminar corrigiendo
    steps = int(60 / env.dt)
    for k in range(steps):
        t = k * env.dt
        phase = (PH_CORR if correcting else PH_RUN) if vision else PH_BLIND
        sense(k, t, phase)
        if vision and filt.initialized:
            gain = 1.6 if correcting else 1.0
            lim = max_bias_corr if correcting else max_bias
            w_des = -gain * (k_psi * filt.psi + k_d * filt.d)
            bias = float(np.clip(-w_des / TURN_GAIN, -lim, lim))
            if correcting:  # termina la correccion cuando lleva 0.5 s cerca del centro y alineado
                calm = calm + 1 if (abs(filt.d) < 0.05 and abs(filt.psi) < math.radians(4)) else 0
                if calm >= 25:
                    correcting = False
                    log("Centrado tras %.1f s (%.2f m)" % (t, robot_frame(env)[0][0]))
        o = obs.copy()
        o[GYRO_OBS] += bias
        action, _ = model.predict(vn.normalize_obs(o), deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        # odometria (desplazamiento real con 10% de error, como la del NAO)
        origin, yaw = robot_frame(env)
        dw = origin - prev_origin
        c, s = math.cos(prev_yaw), math.sin(prev_yaw)
        filt.predict(1.1 * (c * dw[0] + s * dw[1]), (-s * dw[0] + c * dw[1]), 0.9 * (yaw - prev_yaw))
        prev_origin, prev_yaw = origin, yaw
        hist["x"].append(origin[0]); hist["d_true"].append(origin[1]); hist["psi_true"].append(yaw)
        hist["d_est"].append(filt.d if (vision and filt.initialized) else np.nan)
        hist["psi_est"].append(filt.psi if (vision and filt.initialized) else np.nan)
        hist["bias"].append(bias)
        if not show():
            break
        if terminated:
            result = "CAIDA"
            break
        if foot_out(env):
            result = "SALIO DEL CARRIL"
            break
        if max(d.site("left_foot").xpos[0], d.site("right_foot").xpos[0]) >= LENGTH:
            result = "META"
            break
    if video is not None:  # unos cuadros extra al final
        for _ in range(10):
            frames.append(frames[-1])
    if interrupted[0]:
        result = "DETENIDO"
    if not hist["x"]:
        hist["x"].append(robot_frame(env)[0][0]); hist["d_true"].append(start_d)
    hist = {k_: np.array(v) for k_, v in hist.items()}
    info = dict(result=result, t=t + env.dt, x=float(hist["x"][-1]), max_d=float(np.max(np.abs(hist["d_true"]))))
    log("%-14s -> %-17s t=%5.1f s  avance=%.2f m  |desvio| max=%.1f cm"
        % ("RL + vision" if vision else "Solo RL", result, info["t"], info["x"], 100 * info["max_d"]))
    if ctx is not None:
        if cam_win is not None:
            msg = {"META": "LLEGO A LA META en %.1f s" % info["t"], "SALIO DEL CARRIL": "SE SALIO DEL CARRIL",
                   "CAIDA": "SE CAYO", "DETENIDO": "Detenido"}.get(result, result)
            cam_win.set_title(msg, "#0b8043" if result == "META" else "#c5221f")
        return info, hist, frames
    if viewer is not None:
        # Dejar las ventanas abiertas mostrando el resultado hasta que el usuario cierre la 3D
        print("Resultado: %s. Cierra la ventana 3D para terminar." % result)
        while viewer.is_running():
            viewer.sync()
            if cam_win is not None and cam_win.interactive and cam_win.is_open():
                cam_win.plt.pause(0.05)
            else:
                time.sleep(0.05)
        viewer.close()
        cam_renderer = None
    else:
        env.close()
    return info, hist, frames


# ---------------------------------------------------------------------------------------------- video

def make_frame_data(env, img_rgb, pts, t, phase):
    return dict(cam=img_rgb.copy(), pts=pts, t=t, phase=phase,
                qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy())


def render_video(path, runs, fps=16.7):
    """runs: lista de (titulo, info, hist, frames). Compone 3D + camara del robot + grafica de desvio."""
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    cfg, _, _ = load_policy()
    env = E.NaoWalkEnv(**cfg["env"])
    m, d = env.model, env.data
    r3d = mujoco.Renderer(m, 480, 640)
    camv = mujoco.MjvCamera()
    camv.type = mujoco.mjtCamera.mjCAMERA_FREE
    camv.distance, camv.azimuth, camv.elevation = 2.2, 200, -32
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    canvas = FigureCanvasAgg(fig)
    for title, info, hist, frames in runs:
        for i, fr in enumerate(frames):
            d.qpos[:] = fr["qpos"]
            d.qvel[:] = fr["qvel"]
            mujoco.mj_forward(m, d)
            camv.lookat[:] = [d.qpos[0] + 0.3, d.qpos[1] * 0.5, 0.2]
            r3d.update_scene(d, camv)
            view = r3d.render()
            fig.clf()
            ax1 = fig.add_axes([0.0, 0.08, 0.6, 0.84])
            ax1.imshow(view)
            ax1.axis("off")
            ax1.set_title(title, fontsize=17, fontweight="bold", loc="left")
            ax2 = fig.add_axes([0.62, 0.50, 0.36, 0.42])
            ax2.imshow(fr["cam"])
            if fr["pts"] is not None:
                ax2.scatter(fr["pts"][:, 0], fr["pts"][:, 1], s=6, c="#00e5ff")
            ax2.set_xlim(0, CAM_W)
            ax2.set_ylim(CAM_H, 0)
            ax2.axis("off")
            ax2.set_title("Lo que ve el NAO" + (" + lineas detectadas" if fr["pts"] is not None else " (la vision no se usa)")
                          + "\n" + fr["phase"], fontsize=10)
            ax3 = fig.add_axes([0.66, 0.10, 0.32, 0.32])
            n = max(0, int(min(len(hist["x"]), round(fr["t"] / env.dt) + 1)))
            ax3.axhspan(LANE_HALF - FOOT_HALF_WIDTH - 0.0, 0.6, color="#f4c7c3")
            ax3.axhspan(-0.6, -LANE_HALF + FOOT_HALF_WIDTH, color="#f4c7c3")
            ax3.plot(hist["x"][:n], 100 * 0 + hist["d_true"][:n], color="#1a73e8", lw=2, label="posicion real")
            if not np.all(np.isnan(hist["d_est"])):
                ax3.plot(hist["x"][:n], hist["d_est"][:n], color="#e8710a", lw=1.2, ls="--", label="estimada por vision")
            ax3.set_xlim(0, LENGTH)
            ax3.set_ylim(-0.45, 0.45)
            ax3.set_xlabel("avance en el carril [m]")
            ax3.set_ylabel("desvio lateral [m]")
            ax3.legend(loc="lower left", fontsize=8)
            ax3.grid(alpha=0.3)
            ax3.text(0.05, 0.40, "fuera del carril", fontsize=8, color="#a50e0e")
            status = ("t = %4.1f s" % fr["t"]) if fr["t"] >= 0 else "preparacion: de pie en la salida"
            if i >= len(frames) - 10:
                status += "   ->  " + info["result"] + ("  (%.1f s)" % info["t"] if info["result"] == "META" else "")
            fig.text(0.01, 0.02, status, fontsize=15, fontweight="bold",
                     color="#0b8043" if info["result"] == "META" and i >= len(frames) - 10 else "#202124")
            canvas.draw()
            img = np.asarray(canvas.buffer_rgba())[:, :, :3]
            writer.append_data(img)
    writer.close()
    plt.close(fig)


def live_app(a, pista_cfg):
    """Modo interactivo: visor 3D + ventana de camara con panel para repetir corridas y cambiar colores."""
    from mujoco import viewer as mj_viewer
    policy = load_policy()
    cam_win = CameraWindow(controls=True, d0=a.inicio_d, psi0=a.inicio_psi, vision=not a.sin_vision,
                           piso=pista_cfg["piso"], linea=pista_cfg["linea"])
    ctx = dict(policy=policy, cam_win=cam_win, viewer=None, polaridad=PISTA["polaridad"])

    def open_world():
        """(Re)crea el entorno, el visor 3D y el renderer de la camara con la pista actual."""
        if ctx["viewer"] is not None:
            ctx["viewer"].close()
            time.sleep(0.3)
        env = E.NaoWalkEnv(**policy[0]["env"])
        env.reset(seed=a.seed)
        place_robot(env, cam_win.d0 / 100.0, cam_win.psi0)
        viewer = mj_viewer.launch_passive(env.model, env.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env._torso_id
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 150, -25
        ctx.update(env=env, viewer=viewer, renderer=mujoco.Renderer(env.model, CAM_H, CAM_W))
        cam_win.dirty = True

    open_world()
    print("Usa el panel de la ventana 'Lo que ve el NAO': mueve los deslizadores, marca o desmarca la vision, "
          "cambia los colores y presiona 'Correr / Reiniciar'. Cierra cualquiera de las dos ventanas para salir.")
    while ctx["viewer"].is_running() and cam_win.is_open():
        env, renderer, viewer = ctx["env"], ctx["renderer"], ctx["viewer"]
        if cam_win.request == "colores":
            cam_win.request = None
            try:
                cfg = dict(pista_cfg, piso=cam_win.piso, linea=cam_win.linea)
                info = build_scene.build(cfg)
            except ValueError as exc:
                cam_win.set_title(str(exc)[:60], "#c5221f")
                print(exc)
                continue
            pista_cfg = cfg
            PISTA.update(info)
            ctx["polaridad"] = info["polaridad"]
            open_world()
            if info["contraste"] < 45:
                cam_win.set_title("Ojo: la linea casi no se distingue del piso", "#c5221f")
            continue
        if cam_win.request == "run":
            cam_win.request = None
            run(vision=cam_win.vision, live=True, ctx=ctx, seed=a.seed, pose_noise_deg=a.pose_noise,
                start_d=cam_win.d0 / 100.0, start_psi_deg=cam_win.psi0)
            cam_win.dirty = False
            continue
        if cam_win.request == "stop":
            cam_win.request = None
        if cam_win.dirty:   # vista previa: el robot de pie en la nueva posicion inicial
            cam_win.dirty = False
            env.reset(seed=a.seed)
            place_robot(env, cam_win.d0 / 100.0, cam_win.psi0)
            renderer.update_scene(env.data, camera=env.model.camera("CameraBottom").id)
            cam_win.update(renderer.render(), None, "Listo: ajusta el inicio y presiona Correr",
                           "posicion inicial: d = %+3.0f cm   psi = %+3.0f deg\nvision: %s   piso: %s   linea: %s"
                           % (cam_win.d0, cam_win.psi0, "SI" if cam_win.vision else "NO", cam_win.piso, cam_win.linea))
        viewer.sync()
        if cam_win.interactive:
            cam_win.plt.pause(0.05)
        else:
            time.sleep(0.05)
    ctx["viewer"].close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None, help="archivo .mp4 de salida (graba las dos corridas)")
    ap.add_argument("--live", action="store_true", help="visor 3D en vivo")
    ap.add_argument("--sin-vision", action="store_true", help="con --live: solo la politica RL")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--pose-noise", type=float, default=1.0, help="error de pose de la camara [grados]")
    ap.add_argument("--inicio-d", type=float, default=0.0,
                    help="colocacion inicial: corrido a la izquierda (+) o derecha (-) [cm]")
    ap.add_argument("--inicio-psi", type=float, default=0.0,
                    help="colocacion inicial: girado a la izquierda (+) o derecha (-) [grados]")
    ap.add_argument("--sin-camara", action="store_true", help="con --live: no abrir la ventana de la camara")
    build_scene.add_color_args(ap)   # --piso, --linea, --meta, --sin-veta (si no se pasan: demo/pista.yaml)
    a = ap.parse_args()
    pista_cfg = build_scene.config_from_args(a)
    PISTA.update(build_scene.build(pista_cfg))   # regenera la pista con esos colores
    kw = dict(seed=a.seed, pose_noise_deg=a.pose_noise, start_d=a.inicio_d / 100.0, start_psi_deg=a.inicio_psi)
    if a.live:
        if a.sin_camara:
            run(vision=not a.sin_vision, live=True, **kw)
        else:
            live_app(a, pista_cfg)
        return
    out = a.video
    runs = []
    for vision, title in ((False, "1. Solo RL: lo mas recto posible, sin ver el carril"),
                          (True, "2. RL + vision de carril")):
        info, hist, frames = run(vision=vision, video=out, **kw)
        runs.append((title, info, hist, frames))
    if out:
        print("Componiendo video...")
        render_video(out, runs)
        print("Video:", out)


if __name__ == "__main__":
    main()
