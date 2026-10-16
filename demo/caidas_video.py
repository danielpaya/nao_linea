# -*- coding: utf-8 -*-
"""Videos de las peores caidas de un resultado de demo/caidas.py.

Cada caida se vuelve a simular desde su semilla (causa, nivel y semilla guardados en el JSON), asi que el video es
exactamente esa caida; antes de grabar se comprueba que la repeticion de los mismos numeros.

Uso (Windows o Ubuntu, desde la raiz del repo):
    uv run python -m demo.caidas_video                                  # 10 peores de results/caidas/caidas.json
    uv run python -m demo.caidas_video --entrada results/caidas/caidas_camilo.json --repo-camilo C:\\ruta\\nao-v6-rl-race
    uv run python -m demo.caidas_video --peores 5 --criterio pierna     # ordenar por la carga en las piernas
    uv run python -m demo.caidas_video --por-politica 3                 # las 3 peores de CADA politica
    uv run python -m demo.caidas_video --caso protegido                 # el caso con los motores sueltos
Criterios para elegir las peores (caso activo por defecto):
    todas   mayor carga / par de parada en cualquier articulacion con dato de motor (por defecto)
    pierna  mayor carga / par de parada en una articulacion de pierna (la "severidad" del tablero)
    cabeza  golpe mas fuerte de la cabeza contra el piso
Salida: videos/caidas/<fuente>/NN_<politica>_<causa>_id<ID>.mp4 y videos/caidas/<fuente>/indice.json (que caida es
cada video y sus numeros). videos/ no se sube a git: el JSON de resultados y este script bastan para regenerarlos.

Cada video: vista 3D que sigue al robot (de lado si cae hacia adelante o atras, de frente si cae de lado) y, a la
derecha, la carga de cada articulacion con dato de motor en ese instante (promedio de 20 ms / par de parada; la
raya fina es el maximo desde la causa). Tiempo real hasta poco antes de la caida, camara lenta x4 en la caida y
tiempo real despues. Sin pantalla en Linux se usa MUJOCO_GL=egl; si falla: MUJOCO_GL=osmesa.
"""
import argparse
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

try:                          # torch (politicas de Camilo) antes que el OpenGL de MuJoCo: con algunas instalaciones de
    import torch              # torch (las que traen triton/CUDA) cargarlo despues de OSMesa/EGL termina en un
    import torch._dynamo      # noqa: F401  "Segmentation fault"
except Exception:
    pass
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mujoco  # noqa: E402
from demo import caidas as CA  # noqa: E402

FPS = 30
LENTO = 4                     # camara lenta x4 alrededor de la caida
ANTES_S = 1.0                 # video desde 1 s antes de la causa
LENTO_ANTES, LENTO_DESPUES = 0.35, 0.6       # camara lenta alrededor de la deteccion de la caida...
LENTO_PICO = 0.3                              # ...y alrededor del instante de mayor carga (+-0.3 s)
W3D, H, WPANEL = 720, 544, 400
ESCALA_MAX = 3.0              # la barra llega a 3 veces el par de parada
NOMBRES = {
    "LShoulderPitch": "Hombro pitch izq.", "RShoulderPitch": "Hombro pitch der.", "LElbowRoll": "Codo roll izq.",
    "RElbowRoll": "Codo roll der.", "LHipYawPitch": "Cadera yaw izq.", "RHipYawPitch": "Cadera yaw der.",
    "LHipRoll": "Cadera roll izq.", "RHipRoll": "Cadera roll der.", "LHipPitch": "Cadera pitch izq.",
    "RHipPitch": "Cadera pitch der.", "LKneePitch": "Rodilla izq.", "RKneePitch": "Rodilla der.",
    "LAnklePitch": "Tobillo pitch izq.", "RAnklePitch": "Tobillo pitch der.", "LAnkleRoll": "Tobillo roll izq.",
    "RAnkleRoll": "Tobillo roll der.",
}
CAUSA = {"empujon": "Empujón", "golpe_torso": "Golpe en el torso", "tropiezo": "Tropiezo", "resbalon": "Resbalón",
         "motor_muerto": "Motor muerto", "motores_debiles": "Motores débiles", "retraso": "Retraso de la acción"}
PIEZA = {"head": "cabeza", "LWrist": "mano izq.", "RWrist": "mano der.", "LLowerArm": "antebrazo izq.",
         "RLowerArm": "antebrazo der.", "LUpperArm": "brazo izq.", "RUpperArm": "brazo der.", "torso": "torso",
         "LThigh": "muslo izq.", "RThigh": "muslo der.", "LTibia": "tibia izq.", "RTibia": "tibia der.",
         "LHip2": "cadera izq.", "RHip2": "cadera der."}


# ---------------------------------------------------------------------------------------------- eleccion
def puntajes(d, caso):
    """Por caida: carga relativa maxima (todas), severidad de pierna y golpe de la cabeza."""
    arts = d["articulaciones"]
    crit = {c["articulacion"]: c for c in d["criterios"]}
    par = np.array([crit[a]["parada"] or np.nan for a in arts], float)
    pierna = np.array([a[1:] in CA.MOTOR and a[1:] not in ("ShoulderPitch", "ElbowRoll") for a in arts])
    ih = d["piezas"].index("head")
    filas = []
    for pol in d["politicas"]:
        for f in pol["caidas"]:
            ext = np.array(f["casos"][caso]["ext"], float)
            rel = ext / par
            j = int(np.nanargmax(rel))
            filas.append(dict(pol=pol, f=f, todas=float(np.nanmax(rel)), pierna=float(np.nanmax(rel[pierna])),
                              cabeza=float(f["casos"][caso]["piso"][ih]), peor=arts[j]))
    return filas


def elegir(d, n, criterio, caso, por_politica=0, politica=None):
    filas = puntajes(d, caso)
    if politica:
        filas = [x for x in filas if x["pol"]["politica"] in politica]
    filas.sort(key=lambda x: -x[criterio])
    if por_politica:
        out, cuenta = [], {}
        for x in filas:
            p = x["pol"]["politica"]
            if cuenta.get(p, 0) < por_politica:
                out.append(x)
                cuenta[p] = cuenta.get(p, 0) + 1
        return out
    return filas[:n]


# ---------------------------------------------------------------------------------------------- dibujo
def _fuente(tam):
    from PIL import ImageFont
    for nombre in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(nombre, tam)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=tam)
    except TypeError:
        return ImageFont.load_default()


class Grabador(object):
    """Se llama en cada paso de 2 ms; decide cuando sacar un cuadro y lo dibuja."""

    def __init__(self, motor, info, t_causa, t_caida, caso, t_pico=None):
        from PIL import Image, ImageDraw
        self.Image, self.ImageDraw = Image, ImageDraw
        self.mo, self.info, self.caso = motor, info, caso
        m = motor.m
        self.t_ini = max(0.0, t_causa - ANTES_S)
        self.t_causa, self.t_caida = t_causa, t_caida
        self.lentos = [(max(t_causa - 0.1, t_caida - LENTO_ANTES), t_caida + LENTO_DESPUES)]
        if t_pico is not None:
            self.lentos.append((t_pico - LENTO_PICO, t_pico + LENTO_PICO))
        self.t_fin = t_caida + CA.POST
        self.prox = self.t_ini
        self.cuadros = []
        try:
            self.r = mujoco.Renderer(m, H, W3D)
        except Exception as e:
            raise RuntimeError("No se pudo abrir OpenGL para el video (%s). En Linux sin pantalla prueba "
                               "MUJOCO_GL=osmesa (o egl)." % e)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation = 1.35, -14.0
        self.cam.azimuth = info["azimut"]
        self.mira = None
        idx = {n: i for i, n in enumerate(motor.jnombre)}
        self.filas = [(n, idx[n], CA.MOTOR[n[1:]]["parada"]) for n in NOMBRES if n in idx]
        self.sel = np.array([i for _, i, _ in self.filas])
        self.par = np.array([p for _, _, p in self.filas])
        self.ventana = deque(maxlen=CA.VENTANA)
        self.pico = np.zeros(len(self.filas))
        self.f_tit, self.f_txt, self.f_peq = _fuente(20), _fuente(16), _fuente(13)

    def __call__(self, mo):
        d = mo.d
        self.ventana.append(np.abs(d.qfrc_constraint[mo.jv[self.sel]]))
        t = mo.t_sim
        if t < self.t_ini - 1e-9 or t > self.t_fin + 1e-9:
            return
        rel = np.mean(self.ventana, axis=0) / self.par
        if t >= self.t_causa:
            self.pico = np.maximum(self.pico, rel)
        if t + 1e-9 < self.prox:
            return
        lento = any(a <= t <= b for a, b in self.lentos)
        self.prox += (1.0 / FPS) / (LENTO if lento else 1)
        self.cuadros.append(self.cuadro(t, rel, lento))

    def cuadro(self, t, rel, lento):
        d = self.mo.d
        p = d.xpos[self.mo.torso].copy()
        objetivo = np.array([p[0], p[1], max(0.12, min(0.32, p[2]))])
        self.mira = objetivo if self.mira is None else self.mira + 0.25 * (objetivo - self.mira)
        self.cam.lookat[:] = self.mira
        self.r.update_scene(d, camera=self.cam)
        img3d = self.Image.fromarray(self.r.render())
        lienzo = self.Image.new("RGB", (W3D + WPANEL, H), (246, 246, 244))
        lienzo.paste(img3d, (0, 0))
        dib = self.ImageDraw.Draw(lienzo, "RGBA")
        inf = self.info
        dib.rectangle([0, 0, W3D, 96], fill=(15, 20, 30, 175))
        dib.text((12, 8), inf["titulo"], fill=(255, 255, 255), font=self.f_tit)
        dib.text((12, 34), inf["linea2"], fill=(225, 225, 225), font=self.f_txt)
        dib.text((12, 56), inf["linea3"], fill=(200, 200, 200), font=self.f_peq)
        dib.text((12, 75), inf["linea4"], fill=(200, 200, 200), font=self.f_peq)
        fase = "antes de la causa" if t < self.t_causa else ("perturbado" if t < self.t_caida else "caída detectada")
        dib.rectangle([0, H - 34, W3D, H], fill=(15, 20, 30, 150))
        dib.text((12, H - 27), "t = %.2f s desde la causa · %s" % (t - self.t_causa, fase), fill=(255, 255, 255),
                 font=self.f_txt)
        if lento:
            dib.rectangle([W3D - 196, H - 34, W3D, H], fill=(198, 40, 40, 220))
            dib.text((W3D - 184, H - 27), "CÁMARA LENTA ×%d" % LENTO, fill=(255, 255, 255), font=self.f_txt)
        if self.t_causa <= t < self.t_causa + 0.4:
            dib.rectangle([W3D // 2 - 110, 106, W3D // 2 + 110, 138], fill=(232, 116, 47, 230))
            dib.text((W3D // 2 - 98, 112), "CAUSA: " + CAUSA.get(inf["causa"], inf["causa"]), fill=(255, 255, 255),
                     font=self.f_txt)
        self.panel(dib, rel)
        return np.asarray(lienzo)

    def panel(self, dib, rel):
        x0 = W3D + 16
        dib.text((x0, 12), "Carga / par de parada del motor", fill=(30, 30, 30), font=self.f_txt)
        dib.text((x0, 34), "barra: ahora (20 ms) · raya: máximo desde la causa", fill=(100, 100, 100), font=self.f_peq)
        top, alto = 62, (H - 62 - 40) / len(self.filas)
        xl, xr = x0 + 132, W3D + WPANEL - 52
        esc = lambda v: xl + (xr - xl) * min(v, ESCALA_MAX) / ESCALA_MAX   # noqa: E731
        for v in (1, 2, 3):
            xv = esc(v)
            dib.line([xv, top - 4, xv, top + alto * len(self.filas)],
                     fill=(198, 40, 40) if v == 1 else (210, 210, 205), width=2 if v == 1 else 1)
            dib.text((xv - 8, top + alto * len(self.filas) + 4), "%d×" % v, fill=(110, 110, 110), font=self.f_peq)
        for i, (n, _, _) in enumerate(self.filas):
            y = top + i * alto
            v, pk = rel[i], self.pico[i]
            dib.text((x0, y + alto / 2 - 8), NOMBRES[n], fill=(40, 40, 40), font=self.f_peq)
            dib.rectangle([xl, y + 3, xr, y + alto - 3], fill=(232, 232, 228))
            col = (198, 40, 40) if v > 1 else (232, 116, 47) if v > 0.7 else (66, 133, 200)
            dib.rectangle([xl, y + 3, esc(v), y + alto - 3], fill=col)
            if pk > 0:
                xp = esc(pk)
                dib.line([xp, y + 1, xp, y + alto - 1], fill=(20, 20, 20), width=2)
                dib.text((xr + 6, y + alto / 2 - 8), "%.1f" % pk, fill=(198, 40, 40) if pk > 1 else (90, 90, 90),
                         font=self.f_peq)

    def final(self):
        """Un cuadro con el estado del final (el golpe mas fuerte puede llegar despues del ultimo cuadro)."""
        rel = np.mean(self.ventana, axis=0) / self.par
        self.cuadros.append(self.cuadro(self.mo.t_sim, rel, False))

    def cerrar(self):
        self.r.close()


def azimut(motor, direccion):
    """De lado si cae adelante/atras; de frente si cae de lado (grados, convencion de mjvCamera)."""
    R = motor.d.xmat[motor.torso].reshape(3, 3)
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw + (90.0 if direccion in ("adelante", "atras") else 180.0)


# ---------------------------------------------------------------------------------------------- videos
def _politica_desde_json(pol, repo_camilo):
    p = dict(fuente=pol["fuente"], nombre=pol["politica"], vx=pol["vx"])
    if pol["fuente"] == "camilo":
        repo = Path(repo_camilo).resolve()
        p.update(repo=str(repo), ruta=str(repo / pol["ruta"]))
    else:
        p["ruta"] = str(ROOT / pol["ruta"])
    return p


def grabar(motor, f, caso, info):
    """Repite la caida f, comprueba que da los mismos numeros y la graba."""
    fuerza = 1.5 ** f["nivel"]
    marcas = {}

    def al_detectar(mo):
        marcas.update(t_causa=mo.t_causa_sim, t_caida=mo.t_caida_sim, azimut=azimut(mo, f["direccion"]))

    # Primera pasada: instante de mayor carga relativa (promedio de 20 ms) en una articulacion con dato de motor.
    sel = np.array([i for i, n in enumerate(motor.jnombre) if n in NOMBRES])
    par = np.array([CA.MOTOR[motor.jnombre[i][1:]]["parada"] for i in sel])
    ventana, pico = deque(maxlen=CA.VENTANA), [0.0, None]

    def buscar_pico(mo):
        ventana.append(np.abs(mo.d.qfrc_constraint[mo.jv[sel]]))
        if mo.registro is not None:
            v = float(np.max(np.mean(ventana, axis=0) / par))
            if v > pico[0]:
                pico[:] = [v, mo.t_sim]

    motor.grabador = buscar_pico
    try:
        res, _ = motor.simular(f["causa"], fuerza, f["semilla"], casos=(caso,), al_detectar=al_detectar)
    finally:
        motor.grabador = None
    if not res:
        return None, "al repetirla NO se cayo (resultado de otra version o maquina)"
    dif = max(abs(a - b) for k in ("ext", "w", "tope") for a, b in zip(CA.fila_caso(res["casos"][caso])[k],
                                                                       f["casos"][caso][k]))
    aviso = None if dif < 1e-6 else "la repeticion difiere del JSON (dif. max %.3g)" % dif
    info = dict(info, azimut=marcas["azimut"])
    g = Grabador(motor, info, marcas["t_causa"], marcas["t_caida"], caso, t_pico=pico[1])
    motor.grabador = g
    try:
        motor.simular(f["causa"], fuerza, f["semilla"], casos=(caso,))
        g.final()
    finally:
        motor.grabador = None
        g.cerrar()
    cuadros = g.cuadros + [g.cuadros[-1]] * FPS          # 1 s quieto al final
    return cuadros, aviso


def hacer_videos(entrada, n=10, repo_camilo=None, criterio="todas", caso="activo", por_politica=0, politica=None,
                 carpeta=None, log=print):
    import imageio.v2 as imageio
    d = json.load(open(entrada, encoding="utf-8"))
    fuente = d.get("fuente", "daniel")
    repo_camilo = repo_camilo or os.environ.get("REPO_CAMILO", str(ROOT.parent / "nao-v6-rl-race"))
    elegidas = elegir(d, n, criterio, caso, por_politica, politica)
    carpeta = Path(carpeta or ROOT / "videos" / "caidas" / fuente)
    carpeta.mkdir(parents=True, exist_ok=True)
    motores, indice = {}, []
    for i, x in enumerate(elegidas, 1):
        pol, f = x["pol"], x["f"]
        if pol["politica"] not in motores:
            motores[pol["politica"]] = CA.MotorCaidas(CA.crear_adaptador(_politica_desde_json(pol, repo_camilo)))
        motor = motores[pol["politica"]]
        params = ", ".join("%s %s" % (k, "/".join(v) if isinstance(v, list) else v) for k, v in f["params"].items())
        cf = f["casos"][caso]
        info = dict(
            causa=f["causa"],
            titulo="#%d  %s  ·  %s%s" % (i, pol["politica"], CAUSA.get(f["causa"], f["causa"]),
                                         "  (×%g)" % 1.5 ** f["nivel"] if f["nivel"] else ""),
            linea2="Caso %s · cae hacia %s en %.2f s · primero toca: %s" % (
                caso, f["direccion"].replace("atras", "atrás"), f["t_hasta_caer"],
                PIEZA.get(cf["primera"], cf["primera"] or "—")),
            linea3="Peor: %s %.1f× su par de parada · piernas %.2f× · golpe de cabeza %.1f× el peso" % (
                NOMBRES.get(x["peor"], x["peor"]), x["todas"], x["pierna"], x["cabeza"]),
            linea4="%s · semilla %d · caída %d de %s" % (params, f["semilla"], f["id"], pol["politica"]))
        cuadros, aviso = grabar(motor, f, caso, info)
        if cuadros is None:
            log("  #%d %s caida %d: %s; se omite" % (i, pol["politica"], f["id"], aviso))
            continue
        nombre = "%02d_%s_%s_id%d.mp4" % (i, pol["politica"], f["causa"], f["id"])
        ruta = carpeta / nombre
        with imageio.get_writer(str(ruta), fps=FPS, codec="libx264", quality=8, macro_block_size=16) as w:
            for c in cuadros:
                w.append_data(c)
        indice.append(dict(video=nombre, politica=pol["politica"], id=f["id"], causa=f["causa"], nivel=f["nivel"],
                           semilla=f["semilla"], params=f["params"], caso=caso, direccion=f["direccion"],
                           t_hasta_caer=f["t_hasta_caer"], primera=cf["primera"], carga_max_rel=round(x["todas"], 2),
                           articulacion_max=x["peor"], severidad_pierna=round(x["pierna"], 2),
                           cabeza_impacto=round(x["cabeza"], 2), aviso=aviso))
        log("  %s  (%d cuadros, %.1f s)%s" % (ruta, len(cuadros), len(cuadros) / FPS, "  AVISO: " + aviso if aviso else ""))
    with open(carpeta / "indice.json", "w", encoding="utf-8") as fi:
        json.dump(dict(entrada=os.path.relpath(entrada, ROOT).replace("\\", "/"), criterio=criterio, caso=caso,
                       videos=indice), fi, ensure_ascii=False, indent=1)
    log("Indice: %s" % (carpeta / "indice.json"))
    return indice


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default=str(ROOT / "results" / "caidas" / "caidas.json"))
    ap.add_argument("--peores", type=int, default=10)
    ap.add_argument("--criterio", choices=("todas", "pierna", "cabeza"), default="todas")
    ap.add_argument("--caso", choices=CA.CASOS, default="activo")
    ap.add_argument("--por-politica", type=int, default=0, help="N peores de cada politica (en vez de N en total)")
    ap.add_argument("--politicas", nargs="*", default=None, help="solo estas politicas")
    ap.add_argument("--repo-camilo", default=None)
    ap.add_argument("--carpeta", default=None, help="por defecto videos/caidas/<fuente>")
    a = ap.parse_args()
    hacer_videos(a.entrada, a.peores, a.repo_camilo, a.criterio, a.caso, a.por_politica, a.politicas, a.carpeta)


if __name__ == "__main__":
    main()
