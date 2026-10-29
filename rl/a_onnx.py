# -*- coding: utf-8 -*-
"""Convierte politica_np.npz (salida de exportar.py) a ONNX para el NAO real (ONNX Runtime 1.10.0, opset <= 15).

Uso (no necesita JAX ni GPU; solo numpy, onnx y onnxruntime):
    python a_onnx.py politicas/nao_v08_s0/politica_np.npz            # -> politicas/nao_v08_s0/politica.onnx

El grafo recibe la observacion CRUDA del actor (n_obs = historia * obs_por_paso, float32, forma [1, n_obs]; el orden
esta en meta['etiquetas']) y devuelve la accion en [-1, 1] (forma [1, n_motores]). La normalizacion va DENTRO del grafo
y la accion es tanh(primera mitad de la salida), igual que make_inference_fn(deterministic=True). Despues de convertir
compara contra la red en numpy (debe dar ~1e-6) y, si hay onnxruntime, contra el modelo cargado.
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 13          # <= 15, que es lo que soporta onnxruntime 1.10.0


def red_numpy(x, capas, media, desv):
    x = (x - media) / desv
    for i, (W, b) in enumerate(capas):
        x = x @ W + b
        if i < len(capas) - 1:
            x = x / (1.0 + np.exp(-x))
    n = x.shape[-1] // 2
    return np.tanh(x[..., :n])


def convertir(npz, salida=None):
    d = np.load(npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    media, desv = d["media"].astype(np.float32), d["desviacion"].astype(np.float32)
    capas, i = [], 0
    while "W%d" % i in d:
        capas.append((d["W%d" % i].astype(np.float32), d["b%d" % i].astype(np.float32)))
        i += 1
    n_obs, n_mot = media.size, len(meta["motores"])
    inits, nodos = [], []

    def const(nombre, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, np.float32), nombre))
        return nombre

    nodos.append(helper.make_node("Sub", ["obs", const("media", media)], ["x_c"]))
    nodos.append(helper.make_node("Div", ["x_c", const("desv", desv)], ["x0"]))
    x = "x0"
    for k, (W, b) in enumerate(capas):
        nodos.append(helper.make_node("MatMul", [x, const("W%d" % k, W)], ["m%d" % k]))
        nodos.append(helper.make_node("Add", ["m%d" % k, const("b%d" % k, b)], ["z%d" % k]))
        x = "z%d" % k
        if k < len(capas) - 1:                      # swish = x * sigmoid(x)
            nodos.append(helper.make_node("Sigmoid", [x], ["s%d" % k]))
            nodos.append(helper.make_node("Mul", [x, "s%d" % k], ["h%d" % k]))
            x = "h%d" % k
    sal = capas[-1][0].shape[1]
    assert sal == 2 * n_mot, "la salida debe ser media y desviacion de cada motor"
    nodos.append(helper.make_node("Slice", [x, const_i("ini", [0], inits), const_i("fin", [n_mot], inits),
                                            const_i("eje", [1], inits)], ["mu"]))
    nodos.append(helper.make_node("Tanh", ["mu"], ["accion"]))
    g = helper.make_graph(
        nodos, "politica_" + str(meta.get("generacion", "nao")),
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, n_obs])],
        [helper.make_tensor_value_info("accion", TensorProto.FLOAT, [1, n_mot])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="demo_carril")
    m.ir_version = 6                                 # IR 6 lo leen todas las versiones de onnxruntime >= 1.2
    for k, v in (("generacion", meta.get("generacion")), ("run", meta.get("run")),
                 ("motores", json.dumps(meta["motores"])), ("action_scale", meta["action_scale"]),
                 ("ctrl_dt", meta["ctrl_dt"]), ("etiquetas_actor", json.dumps(meta["etiquetas"][:meta["n_obs"]])),
                 ("postura", json.dumps(meta["postura"]))):
        p = m.metadata_props.add()
        p.key, p.value = k, str(v)
    onnx.checker.check_model(m)
    salida = salida or os.path.join(os.path.dirname(os.path.abspath(npz)), "politica.onnx")
    onnx.save(m, salida)

    # comprobacion: numpy vs onnxruntime en 256 observaciones
    rng = np.random.default_rng(0)
    obs = (media + desv * rng.normal(size=(256, n_obs))).astype(np.float32)
    ref = red_numpy(obs, capas, media, desv)
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(salida, providers=["CPUExecutionProvider"])
        out = np.concatenate([s.run(None, {"obs": o[None]})[0] for o in obs])
        dif = float(np.max(np.abs(out - ref)))
        print("%s  (onnxruntime %s, opset %d, obs %d, accion %d, %d capas) diferencia vs numpy %.1e %s"
              % (salida, ort.__version__, OPSET, n_obs, n_mot, len(capas), dif, "OK" if dif < 1e-4 else "REVISAR"))
        return dif
    except ImportError:
        print(salida, "(sin onnxruntime: no se pudo comparar)")


def const_i(nombre, valores, inits):
    inits.append(numpy_helper.from_array(np.asarray(valores, np.int64), nombre))
    return nombre


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    convertir(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
