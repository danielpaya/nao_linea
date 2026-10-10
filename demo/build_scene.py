# -*- coding: utf-8 -*-
"""Genera la escena MuJoCo de la Carrera Individual (reglamento 3.2.4) con el NAO V6.

  - Duela de madera (textura generada), carril de 70 cm con lineas negras de 1.9 cm, carriles vecinos,
    franja negra de salida (18 cm) y franja roja de meta (30 cm) a 5 m.
  - Camara inferior del NAO en la cabeza, como en el modelo de B-Human: (x 0.05071, z 0.01774) desde
    HeadPitch, inclinada 39.7 grados, 320x240, campo de vision vertical 42.5 grados.

Uso:  python demo/build_scene.py   ->  assets/nao/scene_carril.xml y assets/nao/nao_cam.xml
"""
import math
import re
import zlib
import struct
from pathlib import Path

import numpy as np

NAO_DIR = Path(__file__).resolve().parents[1] / "assets" / "nao"
LANE, LINE, LENGTH = 0.70, 0.019, 5.0
SPACING = LANE + LINE
CAM_PITCH = math.radians(39.7)
CAM_FOVY = 42.5


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def wood_texture(path, size=1024, metres=1.8, seed=0):
    """Tablas de 9 cm a lo largo de x con tonos distintos, veta y juntas. La imagen cubre metres x metres."""
    rng = np.random.RandomState(seed)
    px = metres / size
    v, u = np.mgrid[0:size, 0:size] * px          # v -> y del mundo, u -> x del mundo
    plank = np.floor(v / 0.09)
    tone = rng.uniform(0.85, 1.08, int(plank.max()) + 2)[plank.astype(int)]
    offset = rng.uniform(0, 3, int(plank.max()) + 2)[plank.astype(int)]
    grain = 0.5 + 0.5 * np.sin(v * 420 + 2.5 * np.sin(u * 3.1 + offset) + offset)
    base = np.array([196, 148, 96], dtype=float)  # RGB madera clara
    img = base * (tone[..., None] * (0.93 - 0.08 * grain[..., None]))
    joint = (v % 0.09) < 0.0012
    img[joint] *= 0.72
    write_png(path, np.clip(img, 0, 255).astype(np.uint8))


def build():
    wood_texture(NAO_DIR / "duela.png")
    # 1) NAO con camara inferior (copia de nao.xml con la camara agregada en el cuerpo Head)
    nao = (NAO_DIR / "nao.xml").read_text()
    s, c = math.sin(CAM_PITCH), math.cos(CAM_PITCH)
    cam = ('<camera name="CameraBottom" pos="0.05071 0 0.01774" xyaxes="0 -1 0 %.6f 0 %.6f" fovy="%.2f" />'
           % (s, c, CAM_FOVY))
    nao, n = re.subn(r'(<body name="Head" pos="0 0 0">\s*<joint name="HeadPitch"[^>]*/>)', r"\1\n          " + cam, nao)
    assert n == 1, "no se encontro el cuerpo Head en nao.xml"
    (NAO_DIR / "nao_cam.xml").write_text(nao)

    # 2) Escena de la pista
    geoms = []
    x0, x1 = -0.4, LENGTH + 0.8
    for k in (-2, -1, 0, 1):
        y = (k + 0.5) * SPACING
        geoms.append('<geom type="box" pos="%.4f %.4f 0.0006" size="%.4f %.4f 0.0006" rgba="0.06 0.06 0.07 1" contype="0" conaffinity="0"/>'
                     % ((x0 + x1) / 2, y, (x1 - x0) / 2, LINE / 2))
    geoms.append('<geom name="salida" type="box" pos="0 0 0.0005" size="0.09 %.4f 0.0005" rgba="0.06 0.06 0.07 1" contype="0" conaffinity="0"/>'
                 % (SPACING * 1.5))
    geoms.append('<geom name="meta" type="box" pos="%.3f 0 0.0007" size="0.15 %.4f 0.0007" rgba="0.75 0.08 0.08 1" contype="0" conaffinity="0"/>'
                 % (LENGTH + 0.15, LANE / 2))
    scene = """<mujoco model="nao carrera">
  <include file="nao_cam.xml"/>
  <statistic center="2.5 0 0.25" extent="3"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="160" elevation="-25" offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.88 0.92" rgb2="0.55 0.6 0.68" width="512" height="3072"/>
    <texture type="2d" name="duela" file="duela.png"/>
    <material name="duela" texture="duela" texuniform="true" texrepeat="%.4f %.4f" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="2.5 0 4" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="duela" contype="1" conaffinity="14"/>
    %s
    <camera name="lateral" pos="2.6 -3.4 1.6" xyaxes="1 0 0 0 0.45 0.89"/>
  </worldbody>
</mujoco>
""" % (1 / 1.8, 1 / 1.8, "\n    ".join(geoms))
    (NAO_DIR / "scene_carril.xml").write_text(scene)
    print("Escena:", NAO_DIR / "scene_carril.xml")


if __name__ == "__main__":
    build()
