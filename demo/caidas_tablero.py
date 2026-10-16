# -*- coding: utf-8 -*-
"""Convierte results/caidas/caidas.json (salida de demo/caidas.py) en las tablas que lee la pestana "Caidas" del
tablero de GitHub Pages (docs/tablero/datos/caidas_*.json).

Uso (desde C:\\demo_carril):
    uv run python -m demo.caidas_tablero          # junta results/caidas/caidas.json y caidas_camilo.json (si existe)
    uv run python -m demo.caidas_tablero --entrada results/caidas/caidas.json --salida docs/tablero/datos

Tablas (una fila por politica y caso; "todas" junta las caidas de todas las politicas de todas las entradas; cada
fila de caidas_resumen.json dice de que fuente es la politica, daniel o camilo):
  caidas_resumen.json         cifras generales: caidas, simulaciones, sobrecarga en pierna, severidad, cabeza/manos.
  caidas_articulaciones.json  por articulacion: carga (mediana, p95), % con sobrecarga, sobrevelocidad, golpe al tope.
  caidas_causas.json          por causa: severidad, % con sobrecarga, tiempo hasta caer, peor articulacion.
  caidas_piezas.json          por pieza del cuerpo: % que toca primero, golpe contra el piso (veces el peso).
  caidas_lista.json           una fila por caida.
Definiciones:
  sobrecarga    carga externa (pico de 20 ms) > par de parada del motor.
  sobrevelocidad  velocidad > velocidad maxima del motor.
  golpe_tope    par contra el tope mecanico > umbral_tope_nm (2 N m).
  severidad     max sobre las articulaciones de pierna de carga / par de parada.
  peor articulacion  la de mayor carga / par de parada (cualquier articulacion con dato de motor).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASOS = ("activo", "protegido")
NOMBRE_ART = {
    "HeadYaw": "Cabeza yaw", "HeadPitch": "Cabeza pitch",
    "LShoulderPitch": "Hombro pitch izq.", "LShoulderRoll": "Hombro roll izq.", "LElbowYaw": "Codo yaw izq.",
    "LElbowRoll": "Codo roll izq.", "LWristYaw": "Muñeca izq.",
    "RShoulderPitch": "Hombro pitch der.", "RShoulderRoll": "Hombro roll der.", "RElbowYaw": "Codo yaw der.",
    "RElbowRoll": "Codo roll der.", "RWristYaw": "Muñeca der.",
    "LHipYawPitch": "Cadera yaw izq.", "LHipRoll": "Cadera roll izq.", "LHipPitch": "Cadera pitch izq.",
    "LKneePitch": "Rodilla izq.", "LAnklePitch": "Tobillo pitch izq.", "LAnkleRoll": "Tobillo roll izq.",
    "RHipYawPitch": "Cadera yaw der.", "RHipRoll": "Cadera roll der.", "RHipPitch": "Cadera pitch der.",
    "RKneePitch": "Rodilla der.", "RAnklePitch": "Tobillo pitch der.", "RAnkleRoll": "Tobillo roll der.",
}
NOMBRE_PIEZA = {
    "torso": "Torso", "head": "Cabeza", "LUpperArm": "Brazo izq.", "LLowerArm": "Antebrazo izq.", "LWrist": "Mano izq.",
    "RUpperArm": "Brazo der.", "RLowerArm": "Antebrazo der.", "RWrist": "Mano der.", "LHip2": "Cadera izq.",
    "LThigh": "Muslo izq.", "LTibia": "Tibia izq.", "left_foot": "Pie izq.", "RHip2": "Cadera der.",
    "RThigh": "Muslo der.", "RTibia": "Tibia der.", "right_foot": "Pie der.",
}
NOMBRE_CAUSA = {"empujon": "Empujón", "golpe_torso": "Golpe en el torso", "tropiezo": "Tropiezo", "resbalon": "Resbalón",
                "motor_muerto": "Motor muerto", "motores_debiles": "Motores débiles", "retraso": "Retraso de la acción"}
DIRS = ("adelante", "atras", "izquierda", "derecha")


def grupo(a):
    return "cabeza" if a.startswith("Head") else "brazo" if ("Shoulder" in a or "Elbow" in a or "Wrist" in a) else "pierna"


def r(x, n=2):
    return None if x is None else round(float(x), n)


class Tablas:
    def __init__(self, d):
        self.arts = d["articulaciones"]
        self.piezas = d["piezas"]
        crit = {c["articulacion"]: c for c in d["criterios"]}
        self.parada = np.array([crit[a]["parada"] or np.nan for a in self.arts], float)
        self.wmax = np.array([crit[a]["w_max"] or np.nan for a in self.arts], float)
        self.con_motor = np.isfinite(self.parada)
        self.pierna = np.array([grupo(a) == "pierna" for a in self.arts]) & self.con_motor
        self.umbral = d["umbral_tope_nm"]

    def matrices(self, caidas, caso):
        k = lambda n: np.array([c["casos"][caso][n] for c in caidas], float)  # noqa: E731
        return k("ext"), k("w"), k("tope"), k("piso")

    def severidad(self, ext):
        return np.max(ext[:, self.pierna] / self.parada[self.pierna], axis=1)

    def peor(self, ext_fila):
        rel = np.where(self.con_motor, ext_fila / np.where(self.con_motor, self.parada, 1), -np.inf)
        j = int(np.argmax(rel))
        return self.arts[j], float(rel[j])


def main():
    ap = argparse.ArgumentParser()
    defecto = [str(p) for p in (ROOT / "results" / "caidas" / "caidas.json",
                                ROOT / "results" / "caidas" / "caidas_camilo.json") if p.exists()]
    ap.add_argument("--entrada", nargs="*", default=defecto, help="uno o varios resultados de demo/caidas.py")
    ap.add_argument("--salida", default=str(ROOT / "docs" / "tablero" / "datos"))
    a = ap.parse_args()
    if not a.entrada:
        raise SystemExit("No hay resultados: corre primero uv run python -m demo.caidas")
    ds = [json.load(open(e, encoding="utf-8")) for e in a.entrada]
    d = ds[0]
    for o in ds[1:]:
        if o["articulaciones"] != d["articulaciones"] or o["piezas"] != d["piezas"]:
            raise SystemExit("Las entradas no tienen las mismas articulaciones o piezas")
    T = Tablas(d)
    fuentes = {}
    grupos = []
    for o in ds:
        for p in o["politicas"]:
            fuentes[p["politica"]] = p.get("fuente", o.get("fuente", "daniel"))
            grupos.append((p["politica"], p["caidas"], p.get("intentos", len(p["caidas"])),
                           sum(p.get("resistio", {}).values())))
    todas = [c for _, cs, _, _ in grupos for c in cs]
    grupos.append(("todas", todas, sum(g[2] for g in grupos), sum(g[3] for g in grupos)))
    fuentes["todas"] = "+".join(sorted(set(fuentes.values())))

    resumen, arts, causas, piezas, lista = [], [], [], [], []
    for pol, cs, intentos, resistio in grupos:
        for caso in CASOS:
            ext, w, tope, piso = T.matrices(cs, caso)
            sev = T.severidad(ext)
            prim = [c["casos"][caso]["primera"] for c in cs]
            ih = T.piezas.index("head")
            cabeza = piso[:, ih]                             # golpe de la cabeza contra el piso, en todas las caidas
            dirs = Counter(c["direccion"] for c in cs)
            resumen.append(dict(
                politica=pol, fuente=fuentes[pol], caso=caso, caidas=len(cs), simulaciones=intentos, resistio=resistio,
                sobrecarga_pierna=r(np.mean(sev > 1)), severidad_mediana=r(np.median(sev)),
                severidad_p95=r(np.percentile(sev, 95)), cabeza_primero=r(np.mean(np.array(prim) == "head")),
                manos_primero=r(np.mean(np.isin(prim, ["LWrist", "RWrist"]))),
                cabeza_impacto_mediana=r(np.median(cabeza)),
                **{"dir_" + k: dirs.get(k, 0) for k in DIRS}))
            for j, art in enumerate(T.arts):
                par, wm = T.parada[j], T.wmax[j]
                motor = np.isfinite(par)
                arts.append(dict(
                    politica=pol, caso=caso, articulacion=art, nombre=NOMBRE_ART.get(art, art), grupo=grupo(art),
                    parada=float(par) if motor else None, w_max=float(wm) if motor else None,
                    carga_mediana=r(np.median(ext[:, j])), carga_p95=r(np.percentile(ext[:, j], 95)),
                    carga_rel_mediana=r(np.median(ext[:, j]) / par) if motor else None,
                    sobrecarga=r(np.mean(ext[:, j] > par), 3) if motor else None,
                    sobrevelocidad=r(np.mean(w[:, j] > wm), 3) if motor else None,
                    vel_p95=r(np.percentile(w[:, j], 95)), golpe_tope=r(np.mean(tope[:, j] > T.umbral), 3),
                    tope_p95=r(np.percentile(tope[:, j], 95))))
            for causa, nombre in NOMBRE_CAUSA.items():
                idx = [i for i, c in enumerate(cs) if c["causa"] == causa]
                if not idx:
                    continue
                e, s = ext[idx], sev[idx]
                rel = np.where(T.con_motor, np.median(e, axis=0) / np.where(T.con_motor, T.parada, 1), -np.inf)
                causas.append(dict(
                    politica=pol, caso=caso, causa=causa, nombre=nombre, caidas=len(idx),
                    severidad_mediana=r(np.median(s)), severidad_p95=r(np.percentile(s, 95)),
                    sobrecarga_pierna=r(np.mean(s > 1), 3),
                    t_caer_mediana=r(np.median([cs[i]["t_hasta_caer"] for i in idx])),
                    peor_articulacion=NOMBRE_ART[T.arts[int(np.argmax(rel))]],
                    primera_pieza=NOMBRE_PIEZA.get(Counter(prim[i] for i in idx).most_common(1)[0][0], "—"),
                    direccion=Counter(cs[i]["direccion"] for i in idx).most_common(1)[0][0]))
            for k, pieza in enumerate(T.piezas):
                piezas.append(dict(politica=pol, caso=caso, pieza=pieza, nombre=NOMBRE_PIEZA.get(pieza, pieza),
                                   primera=r(np.mean(np.array(prim) == pieza), 3), impacto_mediana=r(np.median(piso[:, k])),
                                   impacto_p95=r(np.percentile(piso[:, k], 95))))
        if pol == "todas":
            continue
        filas = {caso: T.matrices(cs, caso)[0] for caso in CASOS}
        for i, c in enumerate(cs):
            f = dict(politica=pol, id=c["id"], causa=NOMBRE_CAUSA.get(c["causa"], c["causa"]), nivel=c["nivel"],
                     detalle=", ".join("%s %s" % (k, "/".join(v) if isinstance(v, list) else v) for k, v in c["params"].items()),
                     direccion=c["direccion"], t_caer=r(c["t_hasta_caer"], 3))
            for caso in CASOS:
                e = filas[caso][i]
                art, rel = T.peor(e)
                f.update({"sev_" + caso: r(T.severidad(e[None])[0]), "peor_" + caso: NOMBRE_ART[art], "peor_rel_" + caso: r(rel),
                          "primera_" + caso: NOMBRE_PIEZA.get(c["casos"][caso]["primera"])})
            lista.append(f)

    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    for n, filas in [("resumen", resumen), ("articulaciones", arts), ("causas", causas), ("piezas", piezas), ("lista", lista)]:
        json.dump(filas, open(out / ("caidas_%s.json" % n), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
        print("caidas_%s.json: %d filas" % (n, len(filas)))


if __name__ == "__main__":
    main()
