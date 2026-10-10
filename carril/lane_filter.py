# -*- coding: utf-8 -*-
"""Filtro de Kalman sencillo (dos estados desacoplados) que fusiona vision con odometria.

Estado: d (desplazamiento lateral respecto al centro del carril, m, + izquierda) y psi (rumbo respecto al
carril, rad, + izquierda). Ademas acumula el avance s a lo largo del carril.

La odometria (ALMotion.getRobotPosition) predice entre imagenes; la vision corrige. Si la vision se pierde
unos instantes (balanceo, otro robot tapando la linea), el robot sigue con la prediccion.
"""
from __future__ import division

import math


class LaneFilter(object):
    def __init__(self, q_d=2e-4, q_psi=4e-4, q_d_per_m=4e-3, q_psi_per_m=1e-2,
                 r_d=(0.015 ** 2, 0.03 ** 2), r_psi=(math.radians(2.0) ** 2, math.radians(4.0) ** 2),
                 gate_sigma=3.5, max_rejects=6):
        self.q_d, self.q_psi = q_d, q_psi
        self.q_d_per_m, self.q_psi_per_m = q_d_per_m, q_psi_per_m
        self.r_d, self.r_psi = r_d, r_psi          # (dos lineas, una linea)
        self.gate_sigma = gate_sigma
        self.max_rejects = max_rejects
        self.reset()

    def reset(self):
        self.d = 0.0
        self.psi = 0.0
        self.P_d = 1.0
        self.P_psi = 1.0
        self.s = 0.0
        self.initialized = False
        self.rejects = 0

    def predict(self, dx, dy, dth):
        """dx, dy, dth: desplazamiento del robot en SU marco desde la ultima prediccion."""
        psi_mid = self.psi + 0.5 * dth
        self.d += dx * math.sin(psi_mid) + dy * math.cos(psi_mid)
        self.s += dx * math.cos(psi_mid) - dy * math.sin(psi_mid)
        self.psi += dth
        dist = math.hypot(dx, dy)
        self.P_d += self.q_d + self.q_d_per_m * dist
        self.P_psi += self.q_psi + self.q_psi_per_m * dist

    def update(self, est):
        """est: LaneEstimate. Devuelve True si la medicion se acepto."""
        if not est.valid:
            return False
        k = 0 if est.n_lines == 2 else 1
        r_d, r_psi = self.r_d[k], self.r_psi[k]
        if not self.initialized:
            self.d, self.psi = est.d, est.psi
            self.P_d, self.P_psi = r_d, r_psi
            self.initialized = True
            return True
        nd = (est.d - self.d) / math.sqrt(self.P_d + r_d)
        npsi = (est.psi - self.psi) / math.sqrt(self.P_psi + r_psi)
        if abs(nd) > self.gate_sigma or abs(npsi) > self.gate_sigma:
            self.rejects += 1
            if self.rejects >= self.max_rejects:  # la prediccion se fue: reiniciar con la vision
                self.d, self.psi = est.d, est.psi
                self.P_d, self.P_psi = r_d, r_psi
                self.rejects = 0
                return True
            return False
        self.rejects = 0
        kd = self.P_d / (self.P_d + r_d)
        kp = self.P_psi / (self.P_psi + r_psi)
        self.d += kd * (est.d - self.d)
        self.psi += kp * (est.psi - self.psi)
        self.P_d *= (1 - kd)
        self.P_psi *= (1 - kp)
        return True

    @property
    def sigma_d(self):
        return math.sqrt(self.P_d)
