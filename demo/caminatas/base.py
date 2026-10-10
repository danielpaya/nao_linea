# -*- coding: utf-8 -*-
"""Piezas comunes de las caminatas "clasicas" (sin RL) para el simulador.

Las dos caminatas (tipo NAOqi y tipo B-Human) tienen la misma interfaz que el robot real:
    caminata.reiniciar()
    angulos = caminata.paso(vx, vy, wz, sensores)      # se llama cada 12 ms (ciclo de B-Human)
- vx, vy [m/s] y wz [rad/s]: comando de velocidad (el mismo que daria la vision por moveToward).
- sensores: dict con 'gyro' (3,) [rad/s], 'angulo' (roll, pitch) [rad] y 'fsr' (fuerza vertical de cada pie, kg)
  -> solo sensores que da el NAO real. Ninguna caminata usa la posicion verdadera del simulador.
- angulos: dict nombre_articulacion -> angulo objetivo [rad] (piernas y brazos, convencion NAOqi).

Cada caminata planea en su propio marco de odometria (empieza en el origen); de ahi saca la pose de los
pies relativa al torso y la convierte en angulos con la IK (cinematica.py).
"""
import math

import numpy as np

DT = 0.012                      # ciclo de control [s] (B-Human/LoLA)
BRAZOS = {"LShoulderPitch": 1.57, "LShoulderRoll": 0.2, "LElbowYaw": -1.57, "LElbowRoll": -0.2, "LWristYaw": 0.0,
          "RShoulderPitch": 1.57, "RShoulderRoll": -0.2, "RElbowYaw": 1.57, "RElbowRoll": 0.2, "RWristYaw": 0.0,
          "HeadYaw": 0.0, "HeadPitch": 0.0}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def suave(s):
    """0 -> 1 con velocidad cero en los extremos (perfil coseno)."""
    s = min(1.0, max(0.0, s))
    return 0.5 - 0.5 * math.cos(math.pi * s)


def acercar(actual, objetivo, max_delta):
    return actual + max(-max_delta, min(max_delta, objetivo - actual))


class Pose2D(object):
    """Pose en el plano del suelo: x, y [m], yaw [rad]."""
    __slots__ = ("x", "y", "yaw")

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)

    def copia(self):
        return Pose2D(self.x, self.y, self.yaw)

    def xy(self):
        return np.array([self.x, self.y])

    def mas(self, dx, dy, dyaw):
        """Compone un desplazamiento expresado en el marco de esta pose."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(self.x + c * dx - s * dy, self.y + s * dx + c * dy, self.yaw + dyaw)

    def interp(self, otra, s):
        return Pose2D(self.x + (otra.x - self.x) * s, self.y + (otra.y - self.y) * s,
                      self.yaw + wrap(otra.yaw - self.yaw) * s)


def pies_en_torso(torso, z_torso, pie, z_pie):
    """Pose de un pie (Pose2D + altura sobre el suelo) en el marco del torso (torso a z_torso del suelo)."""
    p = rot2(-torso.yaw).dot(pie.xy() - torso.xy())
    return np.array([p[0], p[1], z_pie - z_torso]), wrap(pie.yaw - torso.yaw)


def riccati_discreta(A, B, Q, R, iters=20000, tol=1e-12):
    """Resuelve la ecuacion de Riccati discreta iterando (asi no hace falta scipy)."""
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T.dot(P)
        Pn = A.T.dot(P).dot(A) - A.T.dot(P).dot(B).dot(np.linalg.solve(R + BtP.dot(B), BtP.dot(A))) + Q
        if np.max(np.abs(Pn - P)) < tol * max(1.0, np.max(np.abs(P))):
            return Pn
        P = Pn
    return P


class ControlPreviewZMP(object):
    """Control por anticipacion (preview control) del ZMP sobre el pendulo invertido lineal (Kajita 2003).

    Estado por eje: [posicion, velocidad, aceleracion] del CoM. Entrada: jerk. Salida: ZMP = x - zc/g * x''.
    Minimiza sum Q*(zmp - zmp_ref)^2 + R*u^2 mirando N muestras al futuro de la referencia.
    """

    def __init__(self, zc, dt=DT, horizonte=1.2, Q=1.0, R=1e-6):
        g = 9.81
        self.dt = dt
        self.N = int(round(horizonte / dt))
        A = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
        B = np.array([[dt ** 3 / 6], [dt * dt / 2], [dt]])
        C = np.array([[1, 0, -zc / g]])
        # sistema aumentado con el error integrado (Katayama)
        Aa = np.zeros((4, 4))
        Aa[0, 0] = 1
        Aa[0, 1:] = C.dot(A)
        Aa[1:, 1:] = A
        Ba = np.vstack((C.dot(B), B))
        Ia = np.array([[1.0], [0], [0], [0]])
        Qa = np.zeros((4, 4))
        Qa[0, 0] = Q
        P = riccati_discreta(Aa, Ba, Qa, R)
        S = R + Ba.T.dot(P).dot(Ba)
        K = np.linalg.solve(S, Ba.T.dot(P).dot(Aa))
        self.Gi = K[0, 0]
        self.Gx = K[0, 1:]
        Ac = Aa - Ba.dot(K)
        G = np.zeros(self.N)
        X = -Ac.T.dot(P).dot(Ia)
        G[0] = -self.Gi
        for j in range(1, self.N):
            G[j] = np.linalg.solve(S, Ba.T.dot(X))[0, 0]
            X = Ac.T.dot(X)
        self.Gd = G
        self.A, self.B, self.C = A, B, C

    def nuevo_estado(self, pos):
        return {"x": np.array([pos, 0.0, 0.0]), "e": 0.0}

    def paso(self, est, ref_futura):
        """ref_futura: N valores de ZMP de referencia (el primero es el actual). Devuelve el ZMP actual."""
        zmp = float(self.C[0].dot(est["x"]))
        est["e"] += zmp - ref_futura[0]
        u = -self.Gi * est["e"] - self.Gx.dot(est["x"]) - self.Gd[1:].dot(ref_futura[1:self.N])
        est["x"] = self.A.dot(est["x"]) + self.B[:, 0] * u
        return zmp


class Planificador(object):
    """Genera pasos a partir del comando de velocidad, con limites por paso (como ALMotion/B-Human).

    Un paso mueve el pie de balanceo desde su pose hasta la nueva pose; el otro pie es el de apoyo.
    La nueva pose se calcula desde el apoyo: adelante dx, a un lado la separacion nominal + dy, y giro.
    """

    def __init__(self, separacion, max_x, max_y, max_yaw, max_x_atras=None):
        self.sep = separacion
        self.max_x, self.max_y, self.max_yaw = max_x, max_y, max_yaw
        self.max_x_atras = max_x if max_x_atras is None else max_x_atras

    def siguiente(self, apoyo, lado_apoyo, dx, dy, dyaw):
        """apoyo: Pose2D del pie de apoyo; lado_apoyo 'L' o 'R'. dx, dy, dyaw: desplazamiento del cuerpo en este paso.
        Devuelve la Pose2D destino del pie de balanceo.

        El "cuerpo" es el punto medio entre los pies con el rumbo del apoyo. El cuerpo avanza (dx, dy, dyaw) y el pie
        de balanceo cae a media separacion del nuevo cuerpo. Los pasos laterales hacia el lado del apoyo se limitan
        para que los pies no se crucen."""
        sgn = -1.0 if lado_apoyo == "L" else 1.0          # +1: el pie de balanceo es el izquierdo
        dx = max(-self.max_x_atras, min(self.max_x, dx))
        dy = max(-self.max_y, min(self.max_y, dy))
        dyaw = max(-self.max_yaw, min(self.max_yaw, dyaw))
        if dy * sgn < 0:            # moverse hacia el lado del apoyo: lo hace el siguiente paso (el del otro pie)
            dy = 0.0
        cuerpo = apoyo.mas(0.0, sgn * self.sep / 2, 0.0).mas(dx, dy, dyaw)
        return cuerpo.mas(0.0, sgn * self.sep / 2, 0.0)
