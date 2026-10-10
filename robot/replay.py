# -*- coding: utf-8 -*-
"""Reprocesa en el PC las imagenes grabadas en el NAO para revisar la vision y ajustar parametros.

Uso (desde la raiz del repo, despues de copiar la carpeta logs/frames_XXXX del robot):
    uv run python robot/replay.py logs/frames_20261012_101500 --out revision/
    uv run python robot/replay.py logs/frames_20261012_101500 --dark-ratio 0.55 --min-contrast 20
    uv run python robot/replay.py logs/frames_20261012_101500 --polaridad clara

Imprime la estimacion de cada imagen y guarda figuras (foto + puntos detectados + rectas en el piso) para
revisar a ojo si la cinta se detecta bien sobre el piso real. Tambien acepta los .npz del formato viejo.
"""
from __future__ import division, print_function

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from carril import CameraModel, LaneEstimator  # noqa: E402


def load(path):
    """Devuelve (lista de imagenes RGB HxWx3, lista de T 4x4, meta)."""
    if os.path.isdir(path):
        meta = json.load(open(os.path.join(path, "meta.json")))
        frames, poses = [], []
        for line in open(os.path.join(path, "poses.jsonl")):
            rec = json.loads(line)
            raw = open(os.path.join(path, rec["archivo"]), "rb").read()
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(rec["alto"], rec["ancho"], 3))
            poses.append(np.array(rec["T"], dtype=float).reshape(4, 4))
        return frames, poses, meta
    data = np.load(path)
    meta = dict(hfov=float(data["hfov"]) if "hfov" in data.files else 56.3,
                vfov=float(data["vfov"]) if "vfov" in data.files else 43.7, color="rgb")
    return list(data["frames"]), list(data["poses"]), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ruta", help="carpeta logs/frames_XXXX (o archivo .npz viejo)")
    ap.add_argument("--out", default=None, help="carpeta para las figuras")
    ap.add_argument("--every", type=int, default=5, help="guardar figura de 1 de cada N imagenes")
    ap.add_argument("--dark-ratio", type=float, default=0.62)
    ap.add_argument("--min-contrast", type=float, default=10.0)
    ap.add_argument("--polaridad", choices=("oscura", "clara"), default=None)
    ap.add_argument("--hfov", type=float, default=None)
    ap.add_argument("--vfov", type=float, default=None)
    a = ap.parse_args()

    frames, poses, meta = load(a.ruta)
    if not frames:
        print("No hay imagenes en", a.ruta)
        return
    h, w = frames[0].shape[:2]
    cam = CameraModel(w, h, a.hfov or meta.get("hfov", 56.3), a.vfov or meta.get("vfov", 43.7))
    est = LaneEstimator(cam, dark_ratio=a.dark_ratio, min_contrast=a.min_contrast, color_order=meta.get("color", "rgb"),
                        polaridad=a.polaridad or meta.get("polaridad", "oscura"))
    plt = None
    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F811
        if not os.path.isdir(a.out):
            os.makedirs(a.out)
    n_valid, n_two, ds = 0, 0, []
    for i in range(len(frames)):
        e = est.process(frames[i], poses[i])
        n_valid += e.valid
        n_two += e.n_lines == 2
        if e.valid:
            ds.append(e.d)
        print("%4d valido=%d lineas=%d puntos=%3d d=%+6.1f cm psi=%+5.1f deg rojo=%s"
              % (i, e.valid, e.n_lines, e.n_points, 100 * e.d, math.degrees(e.psi),
                 "-" if e.red_dist is None else "%.2f m" % e.red_dist))
        if plt is not None and i % a.every == 0:
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
            rgb = frames[i] if meta.get("color", "rgb") == "rgb" else frames[i][:, :, ::-1]
            ax[0].imshow(rgb)
            if len(est._uv):
                ax[0].scatter(est._uv[:, 0], est._uv[:, 1], s=5, c="#00e5ff")
            ax[0].axis("off")
            ax[0].set_title("imagen %d" % i)
            p = est.last_points
            if p.size:
                ax[1].scatter(p[:, 1], p[:, 0], s=4, c="k")
            xs = np.linspace(0.1, 2.0, 20)
            for c in (e.c_left, e.c_right):
                if c is not None:
                    ax[1].plot(c + e.slope * xs, xs, "r-")
            ax[1].plot([0], [0], "b^", ms=10)
            ax[1].invert_xaxis()
            ax[1].set_aspect("equal")
            ax[1].set_xlabel("lateral [m] (izquierda +)")
            ax[1].set_ylabel("adelante [m]")
            ax[1].set_title("d=%+.1f cm psi=%+.0f deg lineas=%d" % (100 * e.d, math.degrees(e.psi), e.n_lines))
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, "frame_%04d.png" % i), dpi=90)
            plt.close(fig)
    n = len(frames)
    print("\nImagenes: %d | validas: %d (%.0f%%) | con dos lineas: %d" % (n, n_valid, 100.0 * n_valid / n, n_two))
    if ds:
        print("d medio %.1f cm, desviacion %.1f cm (si el robot estaba quieto, esto es el ruido de la vision)"
              % (100 * np.mean(ds), 100 * np.std(ds)))


if __name__ == "__main__":
    main()
