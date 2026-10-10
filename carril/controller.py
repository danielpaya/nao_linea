# -*- coding: utf-8 -*-
"""Controlador de carril: convierte (d, psi) en un comando normalizado para ALMotion.moveToward.

moveToward(x, y, theta) recibe fracciones [-1, 1] de la velocidad maxima de cada eje. La correccion
combina giro (theta) y paso lateral (y), que el NAO puede dar sin girar.

Esta misma salida (velocidad de giro deseada) es la que mas adelante puede recibir una politica RL que
acepte un comando de giro: la capa de vision no cambia.
"""
from __future__ import division

import math


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class LaneController(object):
    def __init__(self, vx_max=1.0, vx_min=0.35, k_psi=1.6, k_d_theta=1.2, k_d_y=2.5,
                 theta_max=0.45, vy_max=0.5, psi_slow=math.radians(20), d_slow=0.15,
                 lane_half=0.35, foot_half_span=0.095):
        self.vx_max, self.vx_min = vx_max, vx_min
        self.k_psi, self.k_d_theta, self.k_d_y = k_psi, k_d_theta, k_d_y
        self.theta_max, self.vy_max = theta_max, vy_max
        self.psi_slow, self.d_slow = psi_slow, d_slow
        self.lane_half = lane_half
        self.foot_half_span = foot_half_span  # mitad de la huella de ambos pies [m] [VERIFICAR] en tu robot

    def margin(self, d):
        """Distancia [m] entre el borde exterior del pie y el borde interior de la linea."""
        return self.lane_half - abs(d) - self.foot_half_span

    def command(self, d, psi, confidence=1.0):
        """Devuelve (vx, vy, vtheta) normalizados para moveToward."""
        theta = -self.k_psi * psi - self.k_d_theta * d
        vy = -self.k_d_y * d
        theta = _clip(theta, -self.theta_max, self.theta_max)
        vy = _clip(vy, -self.vy_max, self.vy_max)
        # Bajar velocidad si el error es grande o la estimacion es dudosa
        slow = max(abs(psi) / self.psi_slow, abs(d) / self.d_slow)
        vx = self.vx_max * (1.0 - 0.6 * min(1.0, slow))
        vx *= 0.6 + 0.4 * _clip(confidence, 0.0, 1.0)
        vx = _clip(vx, self.vx_min, self.vx_max)
        return vx, vy, theta
