# -*- coding: utf-8 -*-
"""Figuras y GIFs del analisis de caidas para el README (media/caidas/).

Uso (desde la raiz del repo):
    uv run python -m demo.caidas_figuras                  # figuras PNG desde results/caidas/*.json
    uv run python -m demo.caidas_figuras --gifs           # ademas, GIFs de los videos elegidos (videos/caidas/...)
    uv run python -m demo.caidas_figuras --gifs 01_v06d_tropiezo_id38 --fuente-gif daniel
Salida: media/caidas/articulaciones.png, causas_y_politicas.png y los GIFs (livianos: 760 px de ancho, 12 fps, 64 colores).
Los videos se hacen antes con demo/caidas_video.py. Las figuras salen de los mismos JSON que el tablero.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "media" / "caidas"
AZUL, NARANJA = "#2a6fdb", "#e8742f"            # activo / protegido (validada: DeltaE >= 27 en CVD)
TINTA, TENUE, REJILLA, FONDO = "#1f2328", "#6b7280", "#e4e4df", "#fcfcfb"
ROJO = "#c62828"
GRUPOS = [("ShoulderPitch", "Hombro pitch"), ("ElbowRoll", "Codo roll"), ("HipPitch", "Cadera pitch"),
          ("HipRoll", "Cadera roll"), ("HipYawPitch", "Cadera yaw"), ("KneePitch", "Rodilla"),
          ("AnklePitch", "Tobillo pitch"), ("AnkleRoll", "Tobillo roll")]
CAUSAS = {"tropiezo": "Tropiezo", "golpe_torso": "Golpe en el torso", "retraso": "Retraso de la acción",
          "resbalon": "Resbalón", "empujon": "Empujón", "motor_muerto": "Motor muerto",
          "motores_debiles": "Motores débiles"}


def cargar(entradas):
    ds = [json.load(open(e, encoding="utf-8")) for e in entradas]
    pols = []
    for d in ds:
        for p in d["politicas"]:
            pols.append((p["politica"], p.get("fuente", d.get("fuente", "daniel")), p["caidas"]))
    return ds[0], pols


def tablas(d, pols):
    """Por articulacion y caso: % de caidas con sobrecarga, sobrevelocidad y golpe al tope (todas las politicas)."""
    crit = {c["articulacion"]: c for c in d["criterios"]}
    arts = d["articulaciones"]
    todas = [c for _, _, cs in pols for c in cs]
    out = {}
    for caso in ("activo", "protegido"):
        ext = np.array([c["casos"][caso]["ext"] for c in todas], float)
        w = np.array([c["casos"][caso]["w"] for c in todas], float)
        tope = np.array([c["casos"][caso]["tope"] for c in todas], float)
        for g, _ in GRUPOS:
            idx = [i for i, a in enumerate(arts) if a[1:] == g or (g == "HipYawPitch" and a == "RHipYawPitch")]
            par, wm = crit[arts[idx[0]]]["parada"], crit[arts[idx[0]]]["w_max"]
            out[(caso, g)] = dict(
                sobrecarga=100 * np.mean(ext[:, idx] > par), sobrevelocidad=100 * np.mean(w[:, idx] > wm),
                tope=100 * np.mean(tope[:, idx] > d["umbral_tope_nm"]))
    return out


def estilo(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": REJILLA, "axes.labelcolor": TENUE,
        "xtick.color": TENUE, "ytick.color": TINTA, "text.color": TINTA, "axes.facecolor": FONDO,
        "figure.facecolor": FONDO, "axes.spines.top": False, "axes.spines.right": False})


def barras_pares(ax, etiquetas, a, b, fmt="{:.0f}"):
    y = np.arange(len(etiquetas))
    h = 0.36
    ax.barh(y - h / 2 - 0.02, a, h, color=AZUL, label="Activo (motores encendidos)")
    ax.barh(y + h / 2 + 0.02, b, h, color=NARANJA, label="Protegido (motores sueltos)")
    for yy, v in zip(y - h / 2 - 0.02, a):
        ax.text(v + 1.2, yy, fmt.format(v), va="center", fontsize=8.5, color=TINTA)
    for yy, v in zip(y + h / 2 + 0.02, b):
        ax.text(v + 1.2, yy, fmt.format(v), va="center", fontsize=8.5, color=TINTA)
    ax.set_yticks(y, etiquetas)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.xaxis.grid(True, color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def figura_articulaciones(d, pols, ruta):
    import matplotlib.pyplot as plt
    estilo(plt)
    t = tablas(d, pols)
    n = sum(len(cs) for _, _, cs in pols)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True)
    marcos = [("sobrecarga", "Sobrecarga", "carga > par de parada del motor"),
              ("sobrevelocidad", "Sobrevelocidad", "velocidad > velocidad máxima del motor"),
              ("tope", "Golpe contra el tope", "más de 2 N·m contra el tope mecánico")]
    nombres = [n_ for _, n_ in GRUPOS]
    for ax, (k, titulo, sub) in zip(axs, marcos):
        barras_pares(ax, nombres, [t[("activo", g)][k] for g, _ in GRUPOS], [t[("protegido", g)][k] for g, _ in GRUPOS])
        ax.set_title(titulo, loc="left", fontsize=12, fontweight="bold", pad=22)
        ax.text(0, 1.03, sub, fontsize=8.8, color=TENUE, transform=ax.transAxes)
        ax.set_xlabel("% de las caídas")
    axs[0].legend(loc="lower right", frameon=False, fontsize=8.8, bbox_to_anchor=(1.0, -0.02))
    fig.suptitle("Qué articulaciones sufren al caer (%d caídas, %d políticas; izquierda y derecha juntas)" % (n, len(pols)),
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def severidad(d, c, caso):
    crit = {x["articulacion"]: x for x in d["criterios"]}
    arts = d["articulaciones"]
    pierna = [i for i, a in enumerate(arts) if a[1:] in ("HipPitch", "HipRoll", "HipYawPitch", "KneePitch", "AnklePitch",
                                                         "AnkleRoll") and crit[a]["parada"]]
    par = np.array([crit[arts[i]]["parada"] for i in pierna])
    return float(np.max(np.array(c["casos"][caso]["ext"])[pierna] / par))


def figura_causas_politicas(d, pols, ruta):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    estilo(plt)
    todas = [c for _, _, cs in pols for c in cs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    # causas: severidad mediana (barra) y percentil 95 (raya), activo y protegido
    filas = []
    for k, nom in CAUSAS.items():
        cs = [c for c in todas if c["causa"] == k]
        filas.append((nom, [(np.median([severidad(d, c, caso) for c in cs]), np.percentile([severidad(d, c, caso) for c in cs], 95))
                            for caso in ("activo", "protegido")], len(cs)))
    filas.sort(key=lambda f: -f[1][0][0])
    y = np.arange(len(filas))
    h = 0.36
    for j, (col, caso) in enumerate(((AZUL, "Activo"), (NARANJA, "Protegido"))):
        yy = y + (j - 0.5) * (h + 0.04)
        med = [f[1][j][0] for f in filas]
        p95 = [f[1][j][1] for f in filas]
        a1.barh(yy, med, h, color=col, label=caso)
        a1.hlines(yy, med, p95, color=TINTA, lw=1.1)
        a1.vlines(p95, yy - h / 2, yy + h / 2, color=TINTA, lw=1.4)
        for q, m_ in zip(yy, med):
            a1.text(0.04, q, "%.2f" % m_, va="center", fontsize=8.5, color="white", fontweight="bold")
    a1.axvline(1, color=ROJO, ls="--", lw=1.2)
    a1.text(1.03, -0.62, "1 = par de parada", color=ROJO, fontsize=8.5, va="center")
    a1.set_yticks(y, ["%s  (%d)" % (f[0], f[2]) for f in filas])
    a1.invert_yaxis()
    a1.set_xlabel("Severidad en las piernas (carga máxima ÷ par de parada del motor)")
    a1.set_title("Causas de caída", loc="left", fontsize=12, fontweight="bold", pad=24)
    a1.text(0, 1.015, "barra: mediana · raya: percentil 95 · entre paréntesis, las caídas", fontsize=8.8, color=TENUE,
            transform=a1.transAxes)
    a1.legend(frameon=False, loc="lower right", fontsize=9)
    # politicas: % de caidas con sobrecarga en alguna pierna, activo / protegido
    pf = []
    for nom, fu, cs in pols:
        pf.append((nom, fu, [100 * np.mean([severidad(d, c, caso) > 1 for c in cs]) for caso in ("activo", "protegido")]))
    pf.sort(key=lambda p: (p[1] != "daniel", p[0]))
    y = np.arange(len(pf))
    for j, col in enumerate((AZUL, NARANJA)):
        yy = y + (j - 0.5) * (h + 0.04)
        v = [p[2][j] for p in pf]
        a2.barh(yy, v, h, color=col)
        for q, m_ in zip(yy, v):
            a2.text(m_ + 1.2, q, "%.0f" % m_, va="center", fontsize=8, color=TINTA)
    a2.set_yticks(y, ["%s" % p[0] for p in pf])
    for lab, p in zip(a2.get_yticklabels(), pf):
        lab.set_color(TINTA)
    a2.invert_yaxis()
    a2.set_xlim(0, 108)
    a2.set_xlabel("% de las caídas con sobrecarga en alguna articulación de pierna")
    a2.set_title("Políticas", loc="left", fontsize=12, fontweight="bold", pad=24)
    a2.text(0, 1.015, "arriba las 7 de Daniel (v05c–v07c), abajo las 6 de Camilo", fontsize=8.8, color=TENUE,
            transform=a2.transAxes)
    nd = sum(1 for p in pf if p[1] == "daniel")
    a2.axhline(nd - 0.5, color=REJILLA, lw=1.2)
    for ax in (a1, a2):
        ax.xaxis.grid(True, color=REJILLA, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def hacer_gif(mp4, gif, ancho=760, fps=12):
    """GIF liviano (~2-3 MB) con ffmpeg: paleta de 64 colores y solo lo que cambia entre cuadros."""
    import subprocess

    import imageio_ffmpeg
    filtro = ("fps=%d,scale=%d:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=none:diff_mode=rectangle" % (fps, ancho))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(mp4), "-vf", filtro, "-loop", "0",
                    str(gif)], check=True)
    return Path(gif).stat().st_size / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", nargs="*", default=None)
    ap.add_argument("--gifs", nargs="*", default=None, metavar="VIDEO",
                    help="sin nombres: los de VIDEOS_README; con nombres (sin .mp4): esos")
    ap.add_argument("--fuente-gif", default=None, help="daniel o camilo (carpeta de videos/caidas)")
    a = ap.parse_args()
    entradas = a.entrada or [str(p) for p in (ROOT / "results" / "caidas" / "caidas.json",
                                              ROOT / "results" / "caidas" / "caidas_camilo.json") if p.exists()]
    SALIDA.mkdir(parents=True, exist_ok=True)
    if a.gifs is None or entradas and not a.gifs:
        d, pols = cargar(entradas)
        figura_articulaciones(d, pols, SALIDA / "articulaciones.png")
        figura_causas_politicas(d, pols, SALIDA / "causas_y_politicas.png")
        print("figuras en", SALIDA)
    if a.gifs is not None:
        nombres = a.gifs or VIDEOS_README
        for n in nombres:
            fuente = a.fuente_gif or ("camilo" if n.split("_")[1].startswith(("rl", "carrera")) else "daniel")
            mp4 = ROOT / "videos" / "caidas" / fuente / (n + ".mp4")
            gif = SALIDA / (n + ".gif")
            print("%s -> %s (%.1f MB)" % (mp4.name, gif.name, hacer_gif(mp4, gif)))


VIDEOS_README = ["01_v06d_tropiezo_id38", "02_v06_golpe_torso_id1", "01_rl_v1.0_empujon_id14"]

if __name__ == "__main__":
    main()
