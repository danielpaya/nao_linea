# -*- coding: utf-8 -*-
"""Resumen para revisar en la manana: evalua lo que falte y compara todas las corridas en una tabla.

Uso (desde ~/nao_rl):  ~/nao_rl/mujoco_playground/.venv/bin/python resumen_manana.py [--horas 24]

1. Busca en runs/ las corridas con politica.pkl de las ultimas --horas (por defecto 24).
2. Evalua (evaluar.py) las que no tengan evaluacion.csv o cuya politica sea mas nueva que su evaluacion.
3. Escribe runs/resumen_<fecha>.md y lo imprime, ordenado por:
   no se cae en nominal > % de metas en condiciones NO vistas > tiempo en 5 m > desvio lateral.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)


def leer_csv(ruta):
    with open(ruta) as f:
        return list(csv.DictReader(f))


def resumen_run(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    ev = leer_csv(os.path.join(run, "evaluacion.csv"))
    prog = leer_csv(os.path.join(run, "progreso.csv")) if os.path.exists(os.path.join(run, "progreso.csv")) else []
    ult = prog[-1] if prog else {}
    estado = open(os.path.join(run, "estado.txt")).read().strip() if os.path.exists(os.path.join(run, "estado.txt")) \
        else "terminado"
    nom = next(r for r in ev if r["condicion"] == "nominal")
    n = int(nom["carreras"])

    def pct(grupo):
        rs = [r for r in ev if r["grupo"] == grupo]
        if not rs:
            return float("nan")
        return 100.0 * sum(int(r["metas"]) for r in rs) / sum(int(r["carreras"]) for r in rs)

    return {
        "corrida": os.path.basename(run.rstrip("/")), "generacion": cfg["generacion"], "semilla": cfg.get("semilla", 0),
        "estado": estado, "pasos_M": float(ult.get("paso", 0)) / 1e6, "minutos": float(ult.get("minutos", 0)),
        "recompensa": float(ult.get("eval/episode_reward", "nan")),
        "nominal_metas": "%d/%d" % (int(nom["metas"]), n), "nominal_caidas": int(nom["caidas"]),
        "t_5m": float(nom["t_meta_medio"]), "desvio_cm": float(nom["desvio_max_cm"]),
        "rumbo_deg": float(nom["rumbo_max_deg"]),
        "metas_vistas_pct": pct("vista"), "metas_no_vistas_pct": pct("no vista"),
        "caidas_total": sum(int(r["caidas"]) for r in ev),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horas", type=float, default=24.0)
    ap.add_argument("--carreras", type=int, default=8)
    a = ap.parse_args()
    import evaluar

    limite = time.time() - a.horas * 3600
    runs = sorted(r for r in glob.glob(os.path.join(AQUI, "runs", "*"))
                  if os.path.exists(os.path.join(r, "politica.pkl")) and os.path.getmtime(r) >= limite)
    if not runs:
        print("No hay corridas con politica.pkl en las ultimas %.0f h." % a.horas)
        return
    for run in runs:
        ev, pol = os.path.join(run, "evaluacion.csv"), os.path.join(run, "politica.pkl")
        if not os.path.exists(ev) or os.path.getmtime(ev) < os.path.getmtime(pol):
            try:
                evaluar.evaluar_run(run, carreras=a.carreras)
            except Exception as exc:  # noqa: BLE001  (una corrida rota no detiene el resumen)
                print("No se pudo evaluar %s: %s" % (run, exc))

    filas = []
    for run in runs:
        try:
            filas.append(resumen_run(run))
        except Exception as exc:  # noqa: BLE001
            print("Sin resumen para %s: %s" % (run, exc))

    def orden(f):
        t = f["t_5m"] if not math.isnan(f["t_5m"]) else 99.0
        nv = f["metas_no_vistas_pct"] if not math.isnan(f["metas_no_vistas_pct"]) else 0.0
        return (f["nominal_caidas"] > 0, -nv, t, f["desvio_cm"])

    filas.sort(key=orden)
    L = ["# Resumen %s" % time.strftime("%Y-%m-%d %H:%M"), "",
         "Orden: no se cae en nominal > %% metas en condiciones NO vistas > tiempo en 5 m > desvio. "
         "Recompensa solo como referencia (no comparable si la recompensa cambio entre generaciones).", "",
         "| # | Corrida | Gen | Semilla | Estado | Pasos | Min | Recomp. | Nominal metas | Caidas nom. | t 5 m | "
         "Desvio cm | Rumbo | % metas vistas | % metas NO vistas | Caidas total |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, f in enumerate(filas, 1):
        L.append("| %d | %s | %s | %s | %s | %.0f M | %.0f | %.1f | %s | %d | %s | %.0f | %.0f | %.0f | %.0f | %d |" % (
            i, f["corrida"], f["generacion"], f["semilla"], f["estado"], f["pasos_M"], f["minutos"], f["recompensa"],
            f["nominal_metas"], f["nominal_caidas"], "-" if math.isnan(f["t_5m"]) else "%.1f s" % f["t_5m"],
            f["desvio_cm"], f["rumbo_deg"], f["metas_vistas_pct"], f["metas_no_vistas_pct"], f["caidas_total"]))
    L += ["", "Detalle por condicion: runs/<corrida>/evaluacion.csv. Videos: ver_politica.py runs/<corrida>."]
    salida = os.path.join(AQUI, "runs", "resumen_%s.md" % time.strftime("%Y%m%d_%H%M"))
    open(salida, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nGuardado en", salida)


if __name__ == "__main__":
    main()
