# -*- coding: utf-8 -*-
"""Modelo de camara pinhole del NAO y proyeccion de pixeles al suelo.

Compatible con Python 2.7 (NAOqi 2.8 en el robot) y Python 3 (PC). Solo depende de numpy.

Convenciones (las de NAOqi):
  - Marco del robot (FRAME_ROBOT = 2): origen en el suelo entre los pies, x adelante, y izquierda, z arriba.
  - Marco de la camara ("CameraBottom" en ALMotion): x = eje optico (adelante), y izquierda, z arriba.
  - Imagen: u crece hacia la derecha, v hacia abajo.
"""
from __future__ import division

import math

import numpy as np

# Campo de vision de las camaras del NAO V6 (sensor OV5640). [VERIFICAR] con la ficha de tu robot;
# en el V5 son 60.97 x 47.64 grados.
HFOV_V6_DEG = 56.3
VFOV_V6_DEG = 43.7

FRAME_ROBOT = 2  # constante de ALMotion


def transform_from_list(values):
    """Convierte la lista de 16 floats de ALMotion.getTransform en una matriz 4x4."""
    return np.array(values, dtype=float).reshape(4, 4)


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_transform(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    """Transformacion camara -> robot. pitch > 0 inclina el eje optico hacia el suelo."""
    T = np.eye(4)
    T[:3, :3] = rot_z(yaw).dot(rot_y(pitch)).dot(rot_x(roll))
    T[:3, 3] = [x, y, z]
    return T


class CameraModel(object):
    def __init__(self, width=320, height=240, hfov_deg=HFOV_V6_DEG, vfov_deg=VFOV_V6_DEG):
        self.width = int(width)
        self.height = int(height)
        self.fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        self.cx = (width - 1) / 2.0
        self.cy = (height - 1) / 2.0

    def pixels_to_ground(self, u, v, T):
        """Interseca los rayos de los pixeles (u, v) con el plano del suelo z = 0.

        T: matriz 4x4 camara -> robot. Devuelve (x, y, depth, valid): coordenadas en el marco del robot,
        profundidad a lo largo del eje optico y mascara de rayos que si apuntan al suelo.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        rx = np.ones_like(u)
        ry = -(u - self.cx) / self.fx
        rz = -(v - self.cy) / self.fy
        R = T[:3, :3]
        t = T[:3, 3]
        dx = R[0, 0] * rx + R[0, 1] * ry + R[0, 2] * rz
        dy = R[1, 0] * rx + R[1, 1] * ry + R[1, 2] * rz
        dz = R[2, 0] * rx + R[2, 1] * ry + R[2, 2] * rz
        valid = dz < -1e-6
        s = np.where(valid, -t[2] / np.where(valid, dz, -1.0), 0.0)
        x = t[0] + s * dx
        y = t[1] + s * dy
        depth = s  # la componente del rayo sobre el eje optico es 1, asi que s es la profundidad
        return x, y, depth, valid
