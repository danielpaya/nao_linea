# -*- coding: utf-8 -*-
"""Caminata "tipo NAOqi" para el simulador: pasos planeados + pendulo invertido lineal + control preview del ZMP.

Es una reimplementacion propia del ESQUEMA que usa ALMotion (Aldebaran/SoftBank documenta que su caminata
se basa en el control del ZMP con un pendulo invertido lineal, resuelto con un controlador preview), con los
parametros por defecto que publica la documentacion de ALMotion (NAOqi 2.8):
    MaxStepX 0.04 m (por defecto; maximo 0.08), MaxStepY 0.14 m entre pies, MaxStepTheta 0.349 rad,
    StepHeight 0.02 m, frecuencia de paso ~2 Hz, TorsoWy 0 (torso recto).
NO es el codigo de Aldebaran (que es cerrado): es la misma familia de algoritmo con los mismos limites.

Interfaz: ver base.py (reiniciar / paso(vx, vy, wz, sensores) cada 12 ms).
"""
import math

import numpy as np

from .base import BRAZOS, DT, ControlPreviewZMP, Planificador, Pose2D, acercar, pies_en_torso, rot2, suave, wrap
from .cinematica import CinematicaPiernas


class CaminataNaoqi(object):
    nombre = "naoqi_like"

    def __init__(self, model, periodo=0.5, doble_apoyo=0.2, altura_paso=0.02, max_x=0.06, max_y=0.04,
                 max_yaw=0.349, z_torso=0.30, max_acel=0.1, k_tobillo=0.05, k_brazos=0.0):
        self.ik = CinematicaPiernas(model)
        self.T = periodo                      # duracion de un paso [s]
        self.ds = doble_apoyo * periodo       # parte de doble apoyo al inicio de cada paso [s]
        self.h = altura_paso
        self.z_torso = z_torso
        self.espera_inicial = 0.6             # de pie antes del primer paso: el preview "ve venir" el cambio de apoyo
        self.max_acel = max_acel              # cambio maximo de velocidad por paso [m/s]
        self.k_tobillo = k_tobillo
        self.k_brazos = k_brazos              # rad de hombro por metro de separacion adelante-atras de los pies            # realimentacion del giroscopio al tobillo (0 = lazo abierto, como ALMotion)
        home = self.ik.pie_home
        self.sep = float(home["L"][1] - home["R"][1])
        self.x_pie = float(home["L"][0])      # pie adelante del origen del torso en la postura home
        self.plan = Planificador(self.sep, max_x, max_y, max_yaw)
        # desplazamiento torso -> CoM (postura home, brazos abajo), en el marco del torso
        self.off_com = np.array([0.0073, 0.0002])
        zc = 0.2684 - (0.3175 - z_torso)
        self.preview = ControlPreviewZMP(zc)
        self.reiniciar()

    # ------------------------------------------------------------------ estado
    def reiniciar(self):
        self.t = 0.0
        self.pies = {"L": Pose2D(self.x_pie, self.sep / 2), "R": Pose2D(self.x_pie, -self.sep / 2)}
        self.z = {"L": 0.0, "R": 0.0}
        self.v = np.zeros(3)                  # comando actual con rampa (vx, vy, wz)
        self.pasos = []                       # pasos planeados (el primero es el actual)
        self.zmp_inicial = np.array([self.x_pie, 0.0])
        self.cx = self.preview.nuevo_estado(self.x_pie)
        self.cy = self.preview.nuevo_estado(0.0)
        self.siguiente_apoyo = "R"            # el primer paso lo da el pie izquierdo
        self.caminando = False
        self.zmp = self.zmp_inicial.copy()
        self.pasos_dados = 0

    # ------------------------------------------------------------------ planeacion
    def _replanear(self, cmd):
        """Planea los pasos futuros desde la pose actual de los pies con el comando cmd (asumido constante)."""
        apoyo_lado = self.siguiente_apoyo
        poses = {"L": self.pies["L"].copia(), "R": self.pies["R"].copia()}
        t0 = self.t + (self.espera_inicial if self.pasos_dados == 0 else 0.0)
        n = int(math.ceil(self.preview.N * DT / self.T)) + 2
        v = self.v.copy()
        self.pasos = []
        for k in range(n):
            # rampa de velocidad paso a paso (acelera como ALMotion, no de golpe)
            for i, lim in enumerate((self.max_acel, self.max_acel, 0.3)):
                v[i] = acercar(v[i], cmd[i], lim)
            if self.pasos_dados == 0 and k == 0:
                d = (0.0, 0.0, 0.0)            # primer paso en el lugar: solo pasa el peso
                v[:] = 0.0
            else:
                d = (v[0] * self.T, 2 * v[1] * self.T, v[2] * self.T)
            balanceo = "L" if apoyo_lado == "R" else "R"
            hasta = self.plan.siguiente(poses[apoyo_lado], apoyo_lado, *d)
            self.pasos.append({"apoyo": apoyo_lado, "ini": t0, "desde": poses[balanceo].copia(), "hasta": hasta,
                               "pose_apoyo": poses[apoyo_lado].copia(), "v": v.copy()})
            poses[balanceo] = hasta
            apoyo_lado = balanceo
            t0 += self.T

    def _zmp_ref(self, tau, zmp_previo):
        """ZMP de referencia en el instante tau, segun los pasos planeados."""
        prev = zmp_previo
        for p in self.pasos:
            if tau < p["ini"]:
                return prev
            apoyo = self._pose_apoyo(p).xy()
            if tau < p["ini"] + self.T:
                s = (tau - p["ini"]) / self.ds
                return prev + (apoyo - prev) * suave(s) if s < 1 else apoyo
            prev = apoyo
        return prev

    def _pose_apoyo(self, p):
        return p["pose_apoyo"]

    # ------------------------------------------------------------------ ciclo
    def paso(self, vx, vy, wz, sensores=None):
        cmd = np.array([vx, vy, wz], float)
        if not self.pasos:
            self._replanear(cmd)
        # al terminar un paso: fijar el pie que aterrizo y replanear el resto con el comando nuevo
        p = self.pasos[0]
        if self.t >= p["ini"] + self.T - 1e-9:
            bal = "L" if p["apoyo"] == "R" else "R"
            self.pies[bal] = p["hasta"].copia()
            self.z[bal] = 0.0
            self.zmp_inicial = self._pose_apoyo(p).xy()
            self.v = p["v"].copy()
            self.pasos_dados += 1
            self.siguiente_apoyo = bal
            self._replanear(cmd)
            p = self.pasos[0]
        # pie de balanceo
        bal = "L" if p["apoyo"] == "R" else "R"
        s = (self.t - p["ini"] - self.ds) / (self.T - self.ds)
        if s > 0:
            self.pies[bal] = p["desde"].interp(p["hasta"], suave(s))
            self.z[bal] = self.h * math.sin(math.pi * min(1.0, s))
        # control preview: CoM a partir del ZMP de referencia futuro
        refs = np.array([self._zmp_ref(self.t + k * DT, self.zmp_inicial) for k in range(self.preview.N)])
        self.zmp = np.array([self.preview.paso(self.cx, refs[:, 0]), self.preview.paso(self.cy, refs[:, 1])])
        ang = self._angulos(sensores)
        self.t += DT
        return ang

    def postura(self):
        """Angulos de la postura de caminar quieto (lo que haria StandInit antes de arrancar)."""
        return self._angulos(None)

    def _angulos(self, sensores):
        com = np.array([self.cx["x"][0], self.cy["x"][0]])
        # torso: rumbo promedio de los pies, posicion tal que el CoM quede donde dice el preview
        yaw = self.pies["R"].yaw + 0.5 * wrap(self.pies["L"].yaw - self.pies["R"].yaw)
        txy = com - rot2(yaw).dot(self.off_com)
        torso = Pose2D(txy[0], txy[1], yaw)
        pl, yl = pies_en_torso(torso, self.z_torso, self.pies["L"], self.z["L"])
        pr, yr = pies_en_torso(torso, self.z_torso, self.pies["R"], self.z["R"])
        ang = self.ik.resolver(pl, yl, pr, yr)
        if self.k_tobillo and sensores is not None:
            g = sensores["gyro"]
            for lado in "LR":
                ang[lado + "AnklePitch"] += self.k_tobillo * g[1]
                ang[lado + "AnkleRoll"] += self.k_tobillo * g[0]
        ang.update(BRAZOS)
        if self.k_brazos:   # balanceo de brazos opuesto a las piernas (WalkArmsEnable de ALMotion)
            dx = rot2(-yaw).dot(self.pies["L"].xy() - self.pies["R"].xy())[0]
            ang["LShoulderPitch"] = BRAZOS["LShoulderPitch"] + self.k_brazos * dx
            ang["RShoulderPitch"] = BRAZOS["RShoulderPitch"] - self.k_brazos * dx
        return ang
