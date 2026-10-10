# -*- coding: utf-8 -*-
"""Caminata "tipo B-Human" para el simulador (familia Walk2014 de rUNSWift, que B-Human usa desde 2018).

Idea: pasos muy cortos y rapidos (250 ms), sin plan de ZMP. Los pies se mueven en el marco del torso:
    - pie de balanceo: va de su posicion a +paso/2 con un perfil parabolico, y se levanta footLift en medio;
    - pie de apoyo: retrocede a -paso/2 de forma lineal (esto es lo que empuja al cuerpo hacia adelante);
    - el apoyo cambia cuando el FSR del pie de balanceo recibe mas peso que el de apoyo (o al acabar el periodo);
    - el giroscopio corrige el tobillo del pie de apoyo (balance), y la velocidad sube con aceleracion limitada.
Parametros tomados de los valores publicados de B-Human / rUNSWift: periodo 250 ms, altura de cadera ~230 mm,
footLift 13 mm, velocidad maxima adelante ~0.3 m/s. Es una reimplementacion propia del esquema, no el codigo
de B-Human (que ademas corre en C++ dentro de su framework).

Interfaz: ver base.py (reiniciar / paso(vx, vy, wz, sensores) cada 12 ms).
"""
import math

import numpy as np

from .base import BRAZOS, DT, acercar
from .cinematica import CinematicaPiernas


def parabolic_step(t, T):
    """0 -> 1 acelerando y frenando (como parabolicStep de Walk2014)."""
    s = min(1.0, max(0.0, t / T))
    return 2 * s * s if s < 0.5 else 4 * s - 2 * s * s - 1


def parabolic_return(s):
    """0 -> 1 -> 0 (altura del pie en el periodo)."""
    s = min(1.0, max(0.0, s))
    return math.sin(math.pi * s)


class CaminataBHuman(object):
    nombre = "bhuman_like"

    def __init__(self, model, periodo=0.25, foot_lift=0.013, z_torso=0.31, max_x=0.3, max_y=0.1, max_yaw=1.0,
                 acel=(0.15, 0.1, 1.0), k_gyro_pitch=0.05, k_gyro_roll=0.05, vaiven=0.0, cambio_fsr=True,
                 k_brazos=0.0):
        self.ik = CinematicaPiernas(model)
        self.T = periodo
        self.lift = foot_lift
        self.z_torso = z_torso
        self.max_v = np.array([max_x, max_y, max_yaw])
        self.acel = np.array(acel)               # aceleracion maxima por eje [m/s^2, m/s^2, rad/s^2]
        self.kp, self.kr = k_gyro_pitch, k_gyro_roll
        self.vaiven = vaiven                     # amplitud del vaiven lateral del torso [m] (0 = Walk2014 puro)
        self.cambio_fsr = cambio_fsr
        self.k_brazos = k_brazos
        home = self.ik.pie_home
        self.x0 = float(home["L"][0])
        self.y0 = float(home["L"][1])
        self.reiniciar()

    def reiniciar(self):
        self.t = 0.0                              # tiempo dentro del paso actual
        self.v = np.zeros(3)                      # velocidad actual (con rampa)
        self.apoyo = "R"                          # el primer paso lo da el izquierdo
        # posicion relativa de cada pie: adelante, lateral extra, giro (respecto a la postura home)
        self.f = {"L": [0.0, 0.0, 0.0], "R": [0.0, 0.0, 0.0]}
        self.f0 = {"L": [0.0, 0.0, 0.0], "R": [0.0, 0.0, 0.0]}
        self.gyro_f = np.zeros(2)
        self.pasos_dados = 0
        self.espera = 0.3                         # de pie un momento antes del primer paso
        self.z = {"L": 0.0, "R": 0.0}

    def postura(self):
        return self._angulos(None)

    def _nuevo_paso(self):
        self.pasos_dados += 1
        self.apoyo = "L" if self.apoyo == "R" else "R"
        self.t = 0.0
        self.f0 = {k: list(v) for k, v in self.f.items()}

    def paso(self, vx, vy, wz, sensores=None):
        if self.espera > 0:
            self.espera -= DT
            return self._angulos(sensores)
        # rampa de velocidad (aceleracion limitada) y limites
        cmd = np.clip([vx, vy, wz], -self.max_v, self.max_v)
        for i in range(3):
            self.v[i] = acercar(self.v[i], cmd[i], self.acel[i] * DT)
        # cambio de apoyo: por FSR (el pie de balanceo ya recibe el peso) o al acabar el periodo
        bal = "L" if self.apoyo == "R" else "R"
        if self.t >= self.T or (self.cambio_fsr and sensores is not None and self.t > 0.75 * self.T
                                and sensores["fsr"][bal] > sensores["fsr"][self.apoyo]):
            self._nuevo_paso()
            bal = "L" if self.apoyo == "R" else "R"
        self.t += DT
        # objetivos del paso: el cuerpo avanza v*T; cada pie va de -paso/2 a +paso/2
        paso_x = self.v[0] * self.T
        paso_y = self.v[1] * self.T
        giro = self.v[2] * self.T
        s_par = parabolic_step(self.t, self.T)
        s_lin = min(1.0, self.t / self.T)
        sb = 1.0 if bal == "L" else -1.0
        # adelante
        self.f[bal][0] = self.f0[bal][0] + (paso_x / 2 - self.f0[bal][0]) * s_par
        self.f[self.apoyo][0] = self.f0[self.apoyo][0] + (-paso_x / 2 - self.f0[self.apoyo][0]) * s_lin
        # lateral: solo se abre el pie del lado hacia donde se va; el otro vuelve
        abre = paso_y if paso_y * sb > 0 else 0.0
        self.f[bal][1] = self.f0[bal][1] + (abre - self.f0[bal][1]) * s_par
        self.f[self.apoyo][1] = self.f0[self.apoyo][1] + (0.0 - self.f0[self.apoyo][1]) * s_lin
        # giro (simetrico: HipYawPitch es un solo motor)
        self.f[bal][2] = self.f0[bal][2] + (giro / 2 - self.f0[bal][2]) * s_par
        self.f[self.apoyo][2] = self.f0[self.apoyo][2] + (-giro / 2 - self.f0[self.apoyo][2]) * s_lin
        # altura
        self.z = {bal: self.lift * parabolic_return(self.t / self.T), self.apoyo: 0.0}
        return self._angulos(sensores)

    def _angulos(self, sensores):
        bal = "L" if self.apoyo == "R" else "R"
        # vaiven: el torso se corre hacia el pie de apoyo (opcional)
        sy = (1.0 if self.apoyo == "L" else -1.0) * self.vaiven * math.sin(math.pi * min(1.0, self.t / self.T))
        pies = {}
        for k, sgn in (("L", 1.0), ("R", -1.0)):
            fx, fy, fyaw = self.f[k]
            pies[k] = (np.array([self.x0 + fx, sgn * self.y0 + fy - sy, self.z.get(k, 0.0) - self.z_torso]), fyaw)
        ang = self.ik.resolver(pies["L"][0], pies["L"][1], pies["R"][0], pies["R"][1])
        if sensores is not None:
            g = np.asarray(sensores["gyro"])[:2]
            self.gyro_f = 0.8 * self.gyro_f + 0.2 * g          # filtro pasa-bajas del giroscopio
            ang[self.apoyo + "AnklePitch"] += self.kp * self.gyro_f[1]
            ang[self.apoyo + "AnkleRoll"] += self.kr * self.gyro_f[0]
        ang.update(BRAZOS)
        if self.k_brazos:
            dx = self.f["L"][0] - self.f["R"][0]
            ang["LShoulderPitch"] = BRAZOS["LShoulderPitch"] - self.k_brazos * dx
            ang["RShoulderPitch"] = BRAZOS["RShoulderPitch"] + self.k_brazos * dx
        return ang
