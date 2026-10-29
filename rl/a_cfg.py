# -*- coding: utf-8 -*-
"""Convierte politica_np.npz (salida de exportar.py) al formato .cfg de B-Human y escribe los datos de paridad.

Por que .cfg y no ONNX: el modulo CopaRLWalkEngine de Camilo (CamiloGomezB/nao-v6-rl-race) ya corrio en el NAO real y
evalua la red en C++ puro leyendo pesos de un .cfg: no depende de ONNX Runtime 1.10 ni de su tiempo de inferencia. Esta
politica es una MLP 4 capas (128x3 + salida), unos 40 mil pesos: cabe de sobra en el ciclo de 24 ms.

Uso (solo numpy; corre en Windows, Ubuntu o aqui):
    python a_cfg.py politicas/nao_v08_s0/politica_np.npz [carpeta_salida]

Salidas en carpeta_salida (por defecto, junto al npz):
    <generacion>.cfg          pesos, normalizador y metadatos (lo lee el modulo C++; copiar a Config/NeuralNets/NaoRL/)
    <generacion>_golden.json  observaciones CRUDAS del actor y la accion que debe dar el C++ (tolerancia 1e-4)

Formato de las capas (igual al de Camilo): pesos filas x columnas POR FILAS, filas = salidas, columnas = entradas, o sea
la traspuesta de W de JAX (W es entradas x salidas). Activacion: swish (x / (1 + exp(-x))) en las ocultas; la salida no
tiene activacion; la accion es tanh de la PRIMERA MITAD de la salida (media de la gaussiana).
"""
import json
import math
import os
import sys

import numpy as np


def fmt(v, dig=9):
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("parametro no finito")
    t = "%.*g" % (dig, v)
    return "0" if t in ("0", "-0") else t


def arreglo(vals, dig=9, por_linea=8):
    vals = [fmt(v, dig) for v in np.asarray(vals).ravel()]
    lin = [", ".join(vals[i:i + por_linea]) for i in range(0, len(vals), por_linea)]
    return "[\n    " + ",\n    ".join(lin) + "\n  ]"


def capa(W, b):
    Wt = np.asarray(W, np.float32).T                       # filas = salidas, columnas = entradas
    return "{\n  filas = %d;\n  columnas = %d;\n  pesos = %s;\n  sesgo = %s;\n}" % (
        Wt.shape[0], Wt.shape[1], arreglo(Wt), arreglo(b))


def red(x, capas, media, desv):
    x = (x - media) / desv
    for i, (W, b) in enumerate(capas):
        x = x @ W + b
        if i < len(capas) - 1:
            x = x / (1.0 + np.exp(-x))
    n = x.shape[-1] // 2
    return np.tanh(x[..., :n])


def convertir(npz, salida=None, n_golden=64):
    d = np.load(npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    media, desv = d["media"].astype(np.float32), d["desviacion"].astype(np.float32)
    capas, i = [], 0
    while "W%d" % i in d:
        capas.append((d["W%d" % i].astype(np.float32), d["b%d" % i].astype(np.float32)))
        i += 1
    gen = str(meta.get("generacion", "nao"))
    salida = salida or os.path.dirname(os.path.abspath(npz))
    os.makedirs(salida, exist_ok=True)
    n_mot = len(meta["motores"])
    assert capas[-1][0].shape[1] == 2 * n_mot, "la salida debe ser media y desviacion de cada motor"
    n_obs = media.size
    paso = n_obs // int(meta["historia"])
    etq = meta["etiquetas"][:n_obs]
    texto = [
        "// Generado por rl/a_cfg.py desde %s (run %s). No editar." % (os.path.basename(npz), meta.get("run")),
        "// Actor determinista: tanh(primera mitad de la salida), MLP swish, normalizador congelado.",
        'nombre = "%s";' % gen,
        'activacion = "swish";',
        "media = %s;" % arreglo(media, 9),
        "desviacion = %s;" % arreglo(desv, 9),
        "ocultas = [\n%s\n];" % ",\n".join(capa(W, b) for W, b in capas[:-1]),
        "salida = %s;" % capa(*capas[-1]),
        "historia = %d;" % int(meta["historia"]),
        "obsPorPaso = %d;" % paso,
        "actionScale = %s;" % fmt(meta["action_scale"]),
        "ctrlDt = %s;" % fmt(meta["ctrl_dt"]),
        "vxMax = %s;" % fmt(meta["vx"]),
        "postura = %s;" % arreglo(meta["postura"], 9),
        'motores = [%s];' % ", ".join('"%s"' % m for m in meta["motores"]),
    ]
    ruta_cfg = os.path.join(salida, gen + ".cfg")
    with open(ruta_cfg, "w", newline="\n") as f:
        f.write("\n".join(texto) + "\n")

    rng = np.random.default_rng(0)
    obs = (media + desv * rng.normal(size=(n_golden, n_obs))).astype(np.float32)
    acc = red(obs, capas, media, desv)
    dorado = {"generacion": gen, "n_obs": n_obs, "n_accion": n_mot, "obs": obs.tolist(), "accion": acc.tolist(),
              "etiquetas": etq, "motores": meta["motores"], "postura": meta["postura"],
              "action_scale": meta["action_scale"], "lo": meta["lo"], "hi": meta["hi"], "vel_max": meta["vel_max"]}
    ruta_g = os.path.join(salida, gen + "_golden.json")
    with open(ruta_g, "w") as f:
        json.dump(dorado, f)
    print("%s (%d bytes)\n%s  (%d casos de paridad)\nobs %d = %d pasos x %d | accion %d | capas %s"
          % (ruta_cfg, os.path.getsize(ruta_cfg), ruta_g, n_golden, n_obs, int(meta["historia"]), paso, n_mot,
             [c[0].shape for c in capas]))
    return ruta_cfg, ruta_g


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    convertir(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
