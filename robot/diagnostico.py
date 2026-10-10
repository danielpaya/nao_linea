#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostico rapido del NAO antes de probar la vision (Python 2.7 o 3, sin numpy obligatorio).

Uso en el robot:   python diagnostico.py
Desde el PC:       python diagnostico.py --ip 192.168.x.x   (requiere el SDK de Python de NAOqi)

Revisa: version de Python y NAOqi, si hay numpy, bateria, camara inferior (guarda una foto .ppm),
pose de la camara (getTransform), botones de arranque/paro, inclinacion y cuanto tarda la vision por
imagen con el estimador en Python puro (y con numpy si existe).
"""
from __future__ import division, print_function

import argparse
import math
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def ok(msg):
    print("  [OK]   " + msg)


def warn(msg):
    print("  [OJO]  " + msg)


def fail(msg):
    print("  [FALLA] " + msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9559)
    ap.add_argument("--out", default=os.path.join(HERE, "logs"))
    a = ap.parse_args()
    if not os.path.isdir(a.out):
        os.makedirs(a.out)

    print("1. Python")
    ok("Python %s en %s" % (platform.python_version(), platform.machine()))
    try:
        import numpy
        ok("numpy %s disponible (se usara el estimador rapido)" % numpy.__version__)
        has_np = True
    except ImportError:
        warn("numpy NO disponible: se usara el estimador en Python puro")
        has_np = False
    try:
        from carril import EstimadorCarrilPuro
        ok("paquete carril importado")
    except Exception as exc:  # noqa: BLE001
        fail("no se pudo importar carril: %s (copiaste la carpeta carril/ al robot?)" % exc)
        return

    print("2. NAOqi")
    try:
        from naoqi import ALProxy
    except ImportError:
        fail("no se encontro el modulo naoqi")
        return
    try:
        system = ALProxy("ALSystem", a.ip, a.port)
        ok("NAOqi %s, robot '%s'" % (system.systemVersion(), system.robotName()))
    except Exception as exc:  # noqa: BLE001
        warn("no se pudo leer la version: %s" % exc)
    try:
        bat = ALProxy("ALBattery", a.ip, a.port).getBatteryCharge()
        (ok if bat >= 50 else warn)("bateria %d %%" % bat)
    except Exception as exc:  # noqa: BLE001
        warn("bateria: %s" % exc)

    motion = ALProxy("ALMotion", a.ip, a.port)
    memory = ALProxy("ALMemory", a.ip, a.port)
    video = ALProxy("ALVideoDevice", a.ip, a.port)

    print("3. Pose de la camara inferior (FRAME_ROBOT)")
    T = list(motion.getTransform("CameraBottom", 2, True))
    x, y, z = T[3], T[7], T[11]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -T[8]))))
    (ok if 0.25 < z < 0.6 else warn)("camara en x=%.3f y=%.3f z=%.3f m, inclinada %.1f deg hacia abajo" % (x, y, z, pitch))
    if z < 0.35:
        warn("la camara esta baja: el robot esta sentado o agachado? Ponlo de pie (StandInit) antes de medir")

    print("4. Camara inferior")
    try:
        for sub in video.getSubscribers():
            if sub.startswith("diag"):
                video.unsubscribe(sub)
    except Exception:  # noqa: BLE001
        pass
    h = video.subscribeCamera("diag", 1, 1, 11, 10)
    try:
        res = None
        for _ in range(20):
            res = video.getImageRemote(h)
            if res is not None:
                break
            time.sleep(0.1)
        if res is None:
            fail("no llegan imagenes de la camara inferior")
            return
        w, hh, data = res[0], res[1], bytearray(res[6])
        ok("imagen %dx%d, %d bytes" % (w, hh, len(data)))
        path = os.path.join(a.out, time.strftime("diag_camara_%H%M%S.ppm"))
        with open(path, "wb") as f:
            f.write(("P6\n%d %d\n255\n" % (w, hh)).encode("ascii"))
            f.write(bytes(data))
        ok("foto guardada en %s (se abre con GIMP o IrfanView, o con replay.py en el PC)" % path)
        mean = sum(data[i] for i in range(0, len(data), 97)) / float(len(range(0, len(data), 97)))
        (ok if 40 < mean < 220 else warn)("brillo medio %.0f / 255" % mean)

        print("5. Tiempo de la vision por imagen")
        est = EstimadorCarrilPuro(w, hh, color_order="rgb")
        t0 = time.time()
        for _ in range(10):
            e = est.procesar(data, T)
        t_py = 100.0 * (time.time() - t0)
        (ok if t_py < 100 else warn)("Python puro: %.0f ms por imagen (a 10 FPS hay 100 ms)" % t_py)
        if e.valid:
            ok("lineas detectadas: d = %+.1f cm, psi = %+.1f deg, %d linea(s)" % (100 * e.d, math.degrees(e.psi), e.n_lines))
        else:
            warn("no se ven lineas en esta foto (normal si el robot no esta en la pista)")
        if has_np:
            import numpy as np
            from carril import CameraModel, LaneEstimator, transform_from_list
            est2 = LaneEstimator(CameraModel(w, hh), color_order="rgb")
            img = np.frombuffer(bytes(data), dtype=np.uint8).reshape(hh, w, 3)
            t0 = time.time()
            for _ in range(10):
                est2.process(img, transform_from_list(T))
            ok("numpy: %.0f ms por imagen" % (100.0 * (time.time() - t0)))
    finally:
        video.unsubscribe(h)

    print("6. Botones e inclinacion (tienes 6 s: toca la cabeza adelante y luego el boton del pecho)")
    seen_head = seen_chest = False
    t_end = time.time() + 6
    while time.time() < t_end and not (seen_head and seen_chest):
        seen_head |= float(memory.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value")) > 0.5
        seen_chest |= float(memory.getData("Device/SubDeviceList/ChestBoard/Button/Sensor/Value")) > 0.5
        time.sleep(0.05)
    (ok if seen_head else warn)("tactil frontal de la cabeza " + ("detectado" if seen_head else "no detectado"))
    (ok if seen_chest else warn)("boton del pecho " + ("detectado" if seen_chest else "no detectado"))
    ax = math.degrees(float(memory.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")))
    ay = math.degrees(float(memory.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")))
    ok("inclinacion del torso: roll %.1f deg, pitch %.1f deg" % (ax, ay))
    print("Listo.")


if __name__ == "__main__":
    main()
