# -*- coding: utf-8 -*-
"""Cinematica inversa (IK) de las piernas del NAO V6, resuelta numericamente con el propio modelo de MuJoCo.

Se resuelve con el torso fijo en el origen (marco del torso = marco del mundo de una copia cinematica) para
poner cada pie en una pose deseada: posicion (x, y, z) y rumbo (yaw), con la planta paralela al suelo.
Usa 11 grados de libertad: 5 por pierna (HipRoll, HipPitch, KneePitch, AnklePitch, AnkleRoll) mas
HipYawPitch, que en el NAO es UN solo motor para las dos piernas (por eso se resuelven juntas).

Hacerlo con el modelo evita errores de signos y medidas: los angulos salen en la convencion de NAOqi.
"""
import math

import mujoco
import numpy as np

W_ROT_DEF = (0.5, 0.5, 0.1)   # pesos de la orientacion (roll, pitch, yaw) frente a la posicion (1)
PIERNA = ("HipRoll", "HipPitch", "KneePitch", "AnklePitch", "AnkleRoll")


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class CinematicaPiernas(object):
    def __init__(self, model, damping=1e-3, iters=6):
        self.m = model
        self.d = mujoco.MjData(model)
        self.damping = damping
        self.iters = iters
        self.W_ROT = np.array(W_ROT_DEF)
        mujoco.mj_resetDataKeyframe(model, self.d, model.key("home").id)
        self.d.qpos[0:3] = 0.0
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_kinematics(model, self.d)
        self.site = {"L": model.site("left_foot").id, "R": model.site("right_foot").id}
        self.R0 = {k: self.d.site_xmat[i].reshape(3, 3).copy() for k, i in self.site.items()}
        # articulaciones: columnas del jacobiano (dof) y posiciones en qpos
        self.joints = ["LHipYawPitch"] + ["L" + j for j in PIERNA] + ["R" + j for j in PIERNA]
        self.qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in self.joints])
        self.dadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in self.joints])
        self.r_hyp_q = model.jnt_qposadr[model.joint("RHipYawPitch").id]
        self.r_hyp_d = model.jnt_dofadr[model.joint("RHipYawPitch").id]
        self.lo = np.array([model.jnt_range[model.joint(j).id][0] for j in self.joints])
        self.hi = np.array([model.jnt_range[model.joint(j).id][1] for j in self.joints])
        self.q = self.d.qpos[self.qadr].copy()
        # pose de los pies en la postura "home" (referencia para los generadores de marcha)
        self.pie_home = {k: self.d.site_xpos[i].copy() for k, i in self.site.items()}
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def _set(self, q):
        self.d.qpos[self.qadr] = q
        self.d.qpos[self.r_hyp_q] = q[0]          # HipYawPitch derecho = izquierdo (un solo motor)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)          # necesario para que mj_jacSite use ejes actualizados

    def resolver(self, pie_L, yaw_L, pie_R, yaw_R):
        """pie_*: posicion del pie (site en la planta) en el marco del torso [m]; yaw_*: rumbo del pie [rad].
        Devuelve dict nombre_articulacion -> angulo [rad] (incluye RHipYawPitch = LHipYawPitch)."""
        objetivos = {"L": (np.asarray(pie_L, float), rot_z(yaw_L).dot(self.R0["L"])),
                     "R": (np.asarray(pie_R, float), rot_z(yaw_R).dot(self.R0["R"]))}
        q = self.q.copy()
        for _ in range(self.iters):
            self._set(q)
            filas_J, filas_e = [], []
            for k in ("L", "R"):
                sid = self.site[k]
                p = self.d.site_xpos[sid]
                Rc = self.d.site_xmat[sid].reshape(3, 3)
                pt, Rt = objetivos[k]
                e_pos = pt - p
                e_rot = 0.5 * (np.cross(Rc[:, 0], Rt[:, 0]) + np.cross(Rc[:, 1], Rt[:, 1]) + np.cross(Rc[:, 2], Rt[:, 2]))
                mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, sid)
                Jp = self._jp[:, self.dadr].copy()
                Jr = self._jr[:, self.dadr].copy()
                Jp[:, 0] += self._jp[:, self.r_hyp_d]  # columna compartida de HipYawPitch
                Jr[:, 0] += self._jr[:, self.r_hyp_d]
                # pesos: la posicion manda; la planta paralela al suelo (roll/pitch) despues; el rumbo (yaw)
                # pesa poco porque con HipYawPitch compartido no se puede fijar el yaw de cada pie por separado
                filas_J += [Jp, Jr * self.W_ROT[:, None]]
                filas_e += [e_pos, e_rot * self.W_ROT]
            J = np.vstack(filas_J)
            e = np.concatenate(filas_e)
            dq = J.T.dot(np.linalg.solve(J.dot(J.T) + self.damping * np.eye(J.shape[0]), e))
            q = np.clip(q + dq, self.lo, self.hi)
            if np.max(np.abs(np.concatenate([filas_e[0], filas_e[2]]))) < 2e-5:
                break
        self.q = q
        out = dict(zip(self.joints, q))
        out["RHipYawPitch"] = q[0]
        return out

    def error(self, pie_L, pie_R):
        """Error de posicion [m] de la ultima solucion (para pruebas)."""
        self._set(self.q)
        return (np.linalg.norm(self.d.site_xpos[self.site["L"]] - pie_L),
                np.linalg.norm(self.d.site_xpos[self.site["R"]] - pie_R))
