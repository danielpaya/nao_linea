# -*- coding: utf-8 -*-
"""Video de "carrera" con las 3 caminatas, cada una en su carril (RL, tipo NAOqi, tipo B-Human).

Cada robot se simula por separado (no se tocan entre si) y luego se dibujan juntos: se renderiza cada robot
con su imagen y su profundidad, y en cada pixel se queda el que esta mas cerca de la camara.

Uso:
    MUJOCO_GL=egl uv run python -m demo.caminatas.video_carrera --out media/carrera_3_caminatas.mp4
    uv run python -m demo.caminatas.video_carrera --sin-vision --gif media/carrera_sin_vision.gif
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CARRILES = {"bhuman": 0.719, "rl": 0.0, "naoqi": -0.719}     # centro de cada carril [m] (izquierda +)
NOMBRES = {"rl": "RL (equipo)", "naoqi": "tipo NAOqi", "bhuman": "tipo B-Human"}
COLORES = {"rl": (26, 115, 232), "naoqi": (232, 113, 10), "bhuman": (11, 128, 67)}


def simular(vision, mu, seed):
    from demo.caminatas.benchmark import correr_clasica, correr_rl
    out = {}
    for c in ("rl", "naoqi", "bhuman"):
        info = correr_rl(mu, seed, vision, grabar=True) if c == "rl" else correr_clasica(c, mu, seed, vision, grabar=True)
        h = info["hist"]
        out[c] = {"t": h[:, 0], "q": info["qpos"], "resultado": info["resultado"], "T": info["t"]}
        print("%-7s %-16s t=%.1f s" % (c, info["resultado"], info["t"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "media" / "carrera_3_caminatas.mp4"))
    ap.add_argument("--gif", default=None, help="ademas guardar un GIF chico")
    ap.add_argument("--sin-vision", action="store_true")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--semilla", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--ancho", type=int, default=960)
    ap.add_argument("--alto", type=int, default=540)
    ap.add_argument("--acelerar", type=float, default=2.0, help="velocidad del video respecto al tiempo real")
    a = ap.parse_args()

    datos = simular(not a.sin_vision, a.mu, a.semilla)
    import mujoco
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "nao" / "scene_carril.xml"))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, a.alto, a.ancho)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 4.6, 20, -30   # desde atras: izquierda del robot = izquierda de la imagen
    t_fin = max(v["t"][-1] for v in datos.values()) + 1.5
    n = int(t_fin / a.acelerar * a.fps)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font_s = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = font_s = ImageFont.load_default()
    frames = []
    for i in range(n):
        t = i * a.acelerar / a.fps
        xs = []
        capas = []
        for c, v in datos.items():
            k = min(len(v["t"]) - 1, int(np.searchsorted(v["t"], t)))
            d.qpos[:] = v["q"][k]
            d.qpos[1] += CARRILES[c]
            mujoco.mj_forward(m, d)
            xs.append(d.qpos[0])
            capas.append(c)
        cam.lookat[:] = [min(4.2, max(1.2, 0.5 * (min(xs) + max(xs)) + 0.3)), 0.0, 0.1]   # entre el ultimo y el primero
        rgb_acc, dep_acc = None, None
        for c in capas:
            v = datos[c]
            k = min(len(v["t"]) - 1, int(np.searchsorted(v["t"], t)))
            d.qpos[:] = v["q"][k]
            d.qpos[1] += CARRILES[c]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False      # las sombras no se pueden combinar entre capas
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            rgb = r.render().copy()
            r.enable_depth_rendering()
            dep = r.render().copy()
            r.disable_depth_rendering()
            if rgb_acc is None:
                rgb_acc, dep_acc = rgb, dep
            else:
                sel = dep < dep_acc - 1e-4
                rgb_acc[sel] = rgb[sel]
                dep_acc = np.minimum(dep_acc, dep)
        img = Image.fromarray(rgb_acc)
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, a.ancho, 40], fill=(255, 255, 255))
        dr.text((14, 8), "Carrera 5 m  -  %s  -  mu = %.1f  -  t = %4.1f s"
                % ("CON vision" if not a.sin_vision else "SIN vision", a.mu, t), fill=(32, 33, 36), font=font)
        y = 50
        for c in ("bhuman", "rl", "naoqi"):
            v = datos[c]
            lado = {"bhuman": "carril izq.", "rl": "carril centro", "naoqi": "carril der."}[c]
            txt = "%s (%s)" % (NOMBRES[c], lado)
            if t >= v["t"][-1]:
                txt += "  ->  %s %.1f s" % (v["resultado"], v["T"])
            dr.rectangle([12, y + 4, 26, y + 18], fill=COLORES[c])
            dr.text((34, y), txt, fill=(32, 33, 36), font=font_s)
            y += 26
        frames.append(np.asarray(img))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    imageio.mimsave(a.out, frames, fps=a.fps, quality=7, macro_block_size=8)
    print("video:", a.out)
    if a.gif:
        paso = 4
        chicos = [np.asarray(Image.fromarray(f).resize((a.ancho // 2, a.alto // 2), Image.LANCZOS)
                             .quantize(colors=96, method=Image.MEDIANCUT).convert("RGB")) for f in frames[::paso]]
        imageio.mimsave(a.gif, chicos, duration=1000.0 * paso / a.fps, loop=0)
        print("gif:", a.gif)


if __name__ == "__main__":
    main()
