#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vision de carril DENTRO del NAO real (NAOqi 2.8, Python 2.7). Funciona con o sin numpy.

Modos (ver robot/LUNES.md):
    python race_nao.py --dry-run             # NO camina: se para y muestra d/psi en vivo (moverlo a mano)
    python race_nao.py --medir               # NO camina: cada toque en la cabeza promedia 2 s y guarda una fila
                                             #   (para comparar con cinta metrica en posiciones conocidas)
    python race_nao.py --dry-run --save-frames 2   # ademas guarda imagenes para revisarlas en el PC
    python race_nao.py --vx 0.4              # CAMINA con la caminata de NAOqi corrigiendo con la vision

  - Camara inferior, QVGA, RGB, 10 FPS -> posicion en el carril (d, psi) -> filtro con odometria
    -> ALMotion.moveToward con correccion de giro y paso lateral.
  - Arranque: tactil FRONTAL de la cabeza sostenido 0.25 s. Paro: boton del PECHO sostenido 0.3 s (o Ctrl+C).
  - Paro de seguridad por inclinacion (caida) cuando camina.
  - No correrlo junto a otro programa que mueva los motores (p. ej. el proceso bhuman).
"""
from __future__ import division, print_function

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # raiz del repo (contiene el paquete carril)
sys.path.insert(0, HERE)                    # en el robot tambien se puede copiar carril/ junto a este archivo

try:
    import numpy as np
except ImportError:
    np = None

from carril import EstimadorCarrilPuro, LaneFilter, LaneController  # noqa: E402  (Python puro)
if np is not None:
    from carril import CameraModel, LaneEstimator, transform_from_list  # noqa: E402

try:
    from naoqi import ALProxy
except ImportError:
    ALProxy = None

FRAME_ROBOT = 2
CAMERA_BOTTOM = 1
RES_QVGA = 1         # 320 x 240
CS_RGB = 11          # kRGBColorSpace
KEY_START = "Device/SubDeviceList/Head/Touch/Front/Sensor/Value"     # tactil frontal
KEY_ABORT = "Device/SubDeviceList/ChestBoard/Button/Sensor/Value"    # boton del pecho
START_HOLD_S, ABORT_HOLD_S = 0.25, 0.3
KEY_ANGLE_X = "Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value"
KEY_ANGLE_Y = "Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value"


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Vision(object):
    """Envuelve el estimador: numpy si existe (mas rapido), si no Python puro."""

    def __init__(self, args):
        self.usa_numpy = np is not None and not args.puro
        kw = dict(dark_ratio=args.dark_ratio, min_contrast=args.min_contrast, color_order="rgb",
                  polaridad=args.polaridad)
        if self.usa_numpy:
            self.est = LaneEstimator(CameraModel(320, 240, args.hfov, args.vfov), **kw)
        else:
            self.est = EstimadorCarrilPuro(320, 240, args.hfov, args.vfov, **kw)

    def procesar(self, w, h, data, T16):
        if self.usa_numpy:
            img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            return self.est.process(img, transform_from_list(T16))
        return self.est.procesar(data, T16)


class NaoIO(object):
    """Todo el acceso a NAOqi en un solo lugar."""

    def __init__(self, ip, port, fps, head_pitch):
        if ALProxy is None:
            raise RuntimeError("No se encontro el modulo naoqi. Ejecuta este script en el robot (o con el SDK de Python).")
        self.motion = ALProxy("ALMotion", ip, port)
        self.posture = ALProxy("ALRobotPosture", ip, port)
        self.video = ALProxy("ALVideoDevice", ip, port)
        self.memory = ALProxy("ALMemory", ip, port)
        self.head_pitch = head_pitch
        try:
            ALProxy("ALAutonomousLife", ip, port).setState("disabled")  # que nada mueva la cabeza
        except Exception as exc:  # noqa: BLE001
            print("Aviso: no se pudo desactivar ALAutonomousLife:", exc)
        try:
            for sub in self.video.getSubscribers():
                if sub.startswith("carril"):
                    self.video.unsubscribe(sub)
        except Exception:  # noqa: BLE001
            pass
        self.handle = self.video.subscribeCamera("carril", CAMERA_BOTTOM, RES_QVGA, CS_RGB, fps)
        self._odom_prev = None

    def stand(self, move_init=True):
        self.motion.wakeUp()
        self.posture.goToPosture("StandInit", 0.6)
        self.look_down()
        if move_init:
            self.motion.moveInit()

    def look_down(self):
        self.motion.setStiffnesses("Head", 1.0)
        self.motion.setAngles(["HeadYaw", "HeadPitch"], [0.0, self.head_pitch], 0.2)

    def image(self):
        """(ancho, alto, bytes RGB, T 16 floats camara->robot) o None."""
        res = self.video.getImageRemote(self.handle)
        if res is None:
            return None
        T16 = list(self.motion.getTransform("CameraBottom", FRAME_ROBOT, True))
        return res[0], res[1], res[6], T16

    def odom_delta(self):
        x, y, th = self.motion.getRobotPosition(True)
        if self._odom_prev is None:
            self._odom_prev = (x, y, th)
            return 0.0, 0.0, 0.0
        x0, y0, th0 = self._odom_prev
        self._odom_prev = (x, y, th)
        dxw, dyw = x - x0, y - y0
        c, s = math.cos(th0), math.sin(th0)
        return c * dxw + s * dyw, -s * dxw + c * dyw, wrap(th - th0)

    def pressed(self, key):
        try:
            return float(self.memory.getData(key)) > 0.5
        except Exception:  # noqa: BLE001
            return False

    def tilt(self):
        ax = float(self.memory.getData(KEY_ANGLE_X))
        ay = float(self.memory.getData(KEY_ANGLE_Y))
        return max(abs(ax), abs(ay))

    def move(self, vx, vy, vth, cfg):
        self.motion.moveToward(vx, vy, vth, cfg)

    def stop(self):
        try:
            self.motion.stopMove()
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        try:
            self.video.unsubscribe(self.handle)
        except Exception:  # noqa: BLE001
            pass


class HoldDetector(object):
    """Un boton cuenta como presionado solo si se mantiene hold_s segundos."""

    def __init__(self, io):
        self.io = io
        self.since = {}

    def held(self, key, hold_s, now):
        if self.io.pressed(key):
            t0 = self.since.setdefault(key, now)
            return now - t0 >= hold_s
        self.since.pop(key, None)
        return False


class FrameSaver(object):
    """Guarda imagenes crudas + pose sin necesitar numpy (las lee robot/replay.py en el PC)."""

    def __init__(self, folder, args):
        self.dir = os.path.join(folder, time.strftime("frames_%Y%m%d_%H%M%S"))
        os.makedirs(self.dir)
        self.poses = open(os.path.join(self.dir, "poses.jsonl"), "w")
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(dict(hfov=args.hfov, vfov=args.vfov, ancho=320, alto=240, color="rgb",
                           polaridad=args.polaridad), f)
        self.n = 0

    def save(self, w, h, data, T16, extra):
        name = "frame_%05d.rgb" % self.n
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(bytes(bytearray(data)))
        rec = dict(archivo=name, ancho=w, alto=h, T=T16)
        rec.update(extra)
        self.poses.write(json.dumps(rec) + "\n")
        self.n += 1

    def close(self):
        self.poses.close()


class Logger(object):
    COLS = ("t", "estado", "valido", "lineas", "d_vis", "psi_vis", "red_dist", "d", "psi", "s", "sigma_d",
            "vx", "vy", "vth", "margen", "ms_vision")

    def __init__(self, folder, prefix="carrera"):
        self.path = os.path.join(folder, time.strftime(prefix + "_%Y%m%d_%H%M%S.csv"))
        self.f = open(self.path, "w")
        self.f.write(",".join(self.COLS) + "\n")

    def row(self, **kw):
        vals = []
        for c in self.COLS:
            v = kw.get(c, "")
            vals.append("%.4f" % v if isinstance(v, float) else str(v))
        self.f.write(",".join(vals) + "\n")

    def close(self):
        self.f.close()


def modo_medir(io, vision, hold, args):
    """Cada toque en el tactil frontal: promedia la vision 2 s y guarda una fila numerada en un CSV.
    El operador anota en paralelo la posicion real (cinta metrica) con el mismo numero."""
    path = os.path.join(args.log_dir, time.strftime("medicion_%Y%m%d_%H%M%S.csv"))
    f = open(path, "w")
    f.write("n,d_cm,psi_deg,d_std_cm,psi_std_deg,validas,lineas_prom,ms_vision\n")
    print("MODO MEDIR. Coloca el robot, toca la cabeza (frontal) y espera 2 s. Pecho 0.3 s = terminar.")
    print("Archivo:", path)
    n = 0
    while True:
        now = time.time()
        if hold.held(KEY_ABORT, ABORT_HOLD_S, now):
            break
        if not hold.held(KEY_START, 0.1, now):
            time.sleep(0.05)
            continue
        print("  midiendo #%d ..." % n)
        ds, ps, ls, ms, total = [], [], [], [], 0
        t_end = time.time() + 2.0
        while time.time() < t_end:
            im = io.image()
            if im is None:
                continue
            total += 1
            t0 = time.time()
            e = vision.procesar(*im)
            ms.append(1000.0 * (time.time() - t0))
            if e.valid:
                ds.append(e.d)
                ps.append(e.psi)
                ls.append(e.n_lines)
        if ds:
            md, mp = sum(ds) / len(ds), sum(ps) / len(ps)
            sd = math.sqrt(sum((x - md) ** 2 for x in ds) / len(ds))
            sp = math.sqrt(sum((x - mp) ** 2 for x in ps) / len(ps))
            f.write("%d,%.2f,%.2f,%.2f,%.2f,%d/%d,%.2f,%.1f\n" % (n, 100 * md, math.degrees(mp), 100 * sd, math.degrees(sp),
                                                                 len(ds), total, sum(ls) / float(len(ls)), sum(ms) / len(ms)))
            f.flush()
            print("  #%d  d = %+5.1f cm (+-%.1f)   psi = %+5.1f deg (+-%.1f)   validas %d/%d   %.0f ms/imagen"
                  % (n, 100 * md, 100 * sd, math.degrees(mp), math.degrees(sp), len(ds), total, sum(ms) / len(ms)))
        else:
            f.write("%d,,,,,0/%d,,\n" % (n, total))
            print("  #%d  SIN LINEAS en %d imagenes (revisar luz, umbrales o polaridad)" % (n, total))
        n += 1
        while io.pressed(KEY_START):   # esperar a que suelte la cabeza
            time.sleep(0.05)
    f.close()
    print("Mediciones guardadas en", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9559)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--head-pitch", type=float, default=0.0, help="rad; + mira mas abajo")
    ap.add_argument("--hfov", type=float, default=56.3, help="campo de vision horizontal de la camara [grados]")
    ap.add_argument("--vfov", type=float, default=43.7)
    ap.add_argument("--dark-ratio", type=float, default=0.62)
    ap.add_argument("--min-contrast", type=float, default=10.0)
    ap.add_argument("--polaridad", choices=("oscura", "clara"), default="oscura",
                    help="oscura = cinta mas oscura que el piso (negra sobre madera); clara = cinta blanca")
    ap.add_argument("--puro", action="store_true", help="forzar el estimador en Python puro aunque haya numpy")
    ap.add_argument("--vx", type=float, default=0.5, help="fraccion de velocidad maxima hacia adelante")
    ap.add_argument("--max-step-x", type=float, default=0.04, help="MaxStepX de la caminata de NAOqi [m]")
    ap.add_argument("--step-freq", type=float, default=0.8, help="MaxStepFrequency normalizada [0, 1]")
    ap.add_argument("--finish-trigger", type=float, default=0.35)
    ap.add_argument("--post-finish", type=float, default=0.45)
    ap.add_argument("--max-distance", type=float, default=6.0, help="tope de seguridad por odometria [m]")
    ap.add_argument("--time-limit", type=float, default=175.0)
    ap.add_argument("--lost-timeout", type=float, default=1.5)
    ap.add_argument("--fall-tilt", type=float, default=0.6, help="rad de inclinacion que detiene todo")
    ap.add_argument("--dry-run", action="store_true", help="no camina: solo estima y muestra")
    ap.add_argument("--medir", action="store_true", help="no camina: mediciones promediadas con un toque en la cabeza")
    ap.add_argument("--sin-pararse", action="store_true", help="no poner al robot de pie (ya esta parado)")
    ap.add_argument("--save-frames", type=int, default=0, help="guardar 1 de cada N imagenes (0 = no)")
    ap.add_argument("--max-saved", type=int, default=400)
    ap.add_argument("--log-dir", default=os.path.join(HERE, "logs"))
    args = ap.parse_args()
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    vision = Vision(args)
    print("Estimador:", "numpy" if vision.usa_numpy else "Python puro (sin numpy)")
    filt = LaneFilter()
    ctrl = LaneController(vx_max=args.vx)
    gait = [["MaxStepX", args.max_step_x], ["MaxStepFrequency", args.step_freq]]
    sin_caminar = args.dry_run or args.medir

    io = NaoIO(args.ip, args.port, args.fps, args.head_pitch)
    hold = HoldDetector(io)
    saver = FrameSaver(args.log_dir, args) if args.save_frames else None
    state = "ESPERA"
    log = None
    try:
        if not args.sin_pararse:
            io.stand(move_init=not sin_caminar)
        else:
            io.look_down()
        if args.medir:
            modo_medir(io, vision, hold, args)
            return
        log = Logger(args.log_dir, "seco" if args.dry_run else "carrera")
        print("Log:", log.path)
        print("Listo. Tactil FRONTAL 0.25 s = arrancar | boton del PECHO 0.3 s (o Ctrl+C) = detener.")
        t_start = None
        last_ok = time.time()
        last_cmd = last_print = 0.0
        finish_s = None
        cmd = (0.0, 0.0, 0.0)
        n_img = 0
        while True:
            now = time.time()
            im = io.image()
            ms = 0.0
            e = None
            if im is not None:
                t0 = time.time()
                e = vision.procesar(*im)
                ms = 1000.0 * (time.time() - t0)
                n_img += 1
                if saver and n_img % args.save_frames == 0 and saver.n < args.max_saved:
                    saver.save(im[0], im[1], im[2], im[3], dict(t=now, valido=bool(e.valid),
                                                               d=e.d if e.valid else None, psi=e.psi if e.valid else None))
            dx, dy, dth = io.odom_delta()
            filt.predict(dx, dy, dth)
            if e is not None and filt.update(e):
                last_ok = now

            if hold.held(KEY_ABORT, ABORT_HOLD_S, now):
                print("Paro manual (boton del pecho).")
                state = "PARO"
                break
            if state == "ESPERA":
                if hold.held(KEY_START, START_HOLD_S, now):
                    state = "CARRERA"
                    t_start = now
                    filt.s = 0.0
                    print("ARRANQUE" + (" (en seco: no camina)" if sin_caminar else ""))
            elif state == "CARRERA":
                t_run = now - t_start
                if not sin_caminar and io.tilt() > args.fall_tilt:
                    print("Inclinacion peligrosa en t=%.1f s: PARO de seguridad" % t_run)
                    state = "CAIDA"
                    break
                if e is not None and e.red_dist is not None and e.red_dist < args.finish_trigger and finish_s is None:
                    finish_s = filt.s + args.post_finish
                    print("Meta a la vista (%.2f m) en t=%.1f s" % (e.red_dist, t_run))
                if not sin_caminar and ((finish_s is not None and filt.s >= finish_s) or filt.s > args.max_distance
                                        or t_run > args.time_limit):
                    state = "META"
                    break
                if now - last_cmd >= 0.1:
                    conf = 1.0 if now - last_ok < args.lost_timeout else 0.0
                    cmd = ctrl.command(filt.d, filt.psi, conf)
                    if not sin_caminar:
                        io.move(cmd[0], cmd[1], cmd[2], gait)
                    last_cmd = now

            log.row(t=(now - t_start) if t_start else 0.0, estado=state,
                    valido=int(bool(e is not None and e.valid)), lineas=(e.n_lines if e is not None else 0),
                    d_vis=(e.d if e is not None and e.valid else ""), psi_vis=(e.psi if e is not None and e.valid else ""),
                    red_dist=(e.red_dist if e is not None and e.red_dist is not None else ""),
                    d=filt.d, psi=filt.psi, s=filt.s, sigma_d=filt.sigma_d,
                    vx=float(cmd[0]), vy=float(cmd[1]), vth=float(cmd[2]), margen=ctrl.margin(filt.d), ms_vision=ms)
            if now - last_print > 0.5:
                last_print = now
                if e is not None and e.valid:
                    vis = "vision d=%+5.1f cm psi=%+5.1f deg (%d linea%s)" % (100 * e.d, math.degrees(e.psi), e.n_lines,
                                                                           "" if e.n_lines == 1 else "s")
                else:
                    vis = "vision: SIN LINEAS"
                print("%-8s %s | filtro d=%+5.1f cm psi=%+5.1f deg | margen %.0f cm | %.0f ms"
                      % (state, vis, 100 * filt.d, math.degrees(filt.psi), 100 * ctrl.margin(filt.d), ms))
    except KeyboardInterrupt:
        print("Interrumpido.")
    finally:
        io.stop()
        io.close()
        if log:
            log.close()
        if saver:
            saver.close()
            print("Imagenes guardadas en", saver.dir, "(%d)" % saver.n)
        print("Fin:", state)


if __name__ == "__main__":
    main()
