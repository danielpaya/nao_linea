# -*- coding: utf-8 -*-
"""Entorno de entrenamiento del NAO V6 para la Carrera (MuJoCo MJX, GPU).

Estructura copiada de los entornos "Joystick" de MuJoCo Playground (G1 y OP3), adaptada al NAO y a la carrera:
  - El actor solo ve sensores que tiene el NAO real (nada de velocidad lineal ni posicion).
  - Accion cada 24 ms (2 ciclos de 12 ms del NAO), 13 motores: piernas (11) + ShoulderPitch (2).
  - Comando: velocidad adelante (0 a 0.3 m/s) y giro; la vision dara el giro en la carrera.
  - Robustez configurable por generacion: aleatorizacion del mundo, motor debil, retraso y empujones.

Piezas (las mismas de cualquier entorno de RL):
  reset()  -> como empieza cada episodio
  step()   -> que pasa en cada paso de 24 ms
  _obs()   -> que ve el actor ("state") y que ve el critico ("privileged_state")
  _recompensas() -> premios y castigos
  _caida() -> cuando se termina el episodio
  aleatorizar() -> el "mundo distinto" de cada robot (se aplica al modelo, fuera del entorno)
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import gait, mjx_env

# Motores que controla la politica (nombres de B-Human / NAOqi). RHipYawPitch no tiene motor propio:
# esta acoplado a LHipYawPitch por una restriccion de igualdad (en el NAO es un solo motor).
MOTORES = ("LHipYawPitch",
           "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
           "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
           "LShoulderPitch", "RShoulderPitch")
MOTORES_PIERNA = MOTORES[:11]
PIES = ("left_foot", "right_foot")          # sites en la planta de cada pie


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.024,                 # 24 ms por accion (2 ciclos de 12 ms del NAO)
        sim_dt=0.002,                  # fisica a 2 ms -> 12 pasos de fisica por accion
        episode_length=1000,           # 1000 x 24 ms = 24 s
        impl="jax",
        action_scale=0.3,              # rad: objetivo = postura_base + accion * escala
        historia=3,                    # cuantas observaciones pasadas ve el actor (compensa no ver la velocidad)
        # --- comando ---
        vx=[0.0, 0.3],                 # m/s adelante
        wz=[-0.4, 0.4],                # rad/s de giro
        prob_quieto=0.05,              # a veces se le pide quedarse quieto
        # --- rumbo (desde nao_v01b) ---
        # La politica ve el ERROR DE RUMBO: rumbo deseado (integra el giro pedido) - rumbo estimado (integra el
        # giroscopio, como AngleZ del NAO, con una deriva pequena). En la carrera, la vision dara este error (psi).
        rumbo=config_dict.create(
            activo=False,
            deriva_gyro=0.005,         # rad/s de sesgo del giroscopio al integrar (al azar por episodio)
            error_max=0.5,             # rad: el error que ve la politica se recorta a +-error_max
        ),
        frecuencia_paso=[1.6, 2.0],    # Hz del ciclo de marcha (izq + der)
        altura_paso=0.03,              # m que deberia subir el pie en el aire
        # --- ruido de los sensores (uniforme, +- nivel * escala) ---
        ruido=config_dict.create(
            nivel=1.0,
            gyro=0.2, gravedad=0.05, angulos=0.02, velocidades=1.0,
        ),
        # --- robustez (cada generacion activa algo) ---
        latencia_max=0,                # retraso de la accion: 0..latencia_max ciclos de 24 ms (al azar por episodio)
        empujones=config_dict.create(
            activos=False,
            magnitud=[0.05, 0.3],      # m/s que se le suman de golpe al torso
            intervalo=[3.0, 6.0],      # s entre empujones
        ),
        # --- caida ---
        inclinacion_max_deg=45.0,
        altura_min=0.20,
        # --- recompensa ---
        recompensa=config_dict.create(
            sigma_vel=0.025,           # (m/s)^2: que tan estricto es seguir la velocidad
            sigma_giro=0.1,            # (rad/s)^2
            separacion_min=0.08,       # m entre pies (el V6 tiene ~3 cm entre piernas: se pisa)
            sigma_rumbo=0.03,          # rad^2 (~10 grados)
            escalas=config_dict.create(
                seguir_vel=1.5,
                seguir_giro=0.75,
                tiempo_aire=2.0,
                fase_pies=1.0,
                vel_z=-0.5,
                giro_xy=-0.15,
                orientacion=-2.0,
                pies_resbalan=-0.25,
                pies_juntos=-5.0,
                cambio_accion=-0.01,
                energia=-0.001,
                limites=-1.0,
                quieto=-1.0,
                seguir_rumbo=0.0,      # desde nao_v01b (se activa junto con rumbo.activo)
                postura_cadera=0.0,    # desde nao_v01c: HipYawPitch y HipRoll cerca de la postura base
                caida=-100.0,
            ),
        ),
    )


def default_config_entrenamiento() -> config_dict.ConfigDict:
    """Parametros de PPO (los de los humanoides de Playground, con redes mas chicas para el NAO)."""
    return config_dict.create(
        num_timesteps=100_000_000,
        num_evals=10,
        num_envs=4096,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=4,
        unroll_length=20,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0.005,
        clipping_epsilon=0.2,
        max_grad_norm=1.0,
        reward_scaling=1.0,
        normalize_observations=True,
        action_repeat=1,
        num_resets_per_eval=1,
        politica=(128, 128, 128),      # chica: tiene que correr en el NAO (Python 2.7, sin torch)
        critico=(256, 256, 256),
    )


class NaoCarrera(mjx_env.MjxEnv):
    def __init__(self, xml_path: str, config: config_dict.ConfigDict = None,
                 config_overrides: Optional[Dict[str, Union[str, int, list]]] = None):
        super().__init__(config or default_config(), config_overrides)
        self._xml_path = xml_path
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._preparar()

    # ------------------------------------------------------------------ indices y constantes
    def _preparar(self):
        m = self._mj_model
        home = m.keyframe("home")
        self._q_inicial = jp.array(home.qpos)
        self._ctrl_home = jp.array(home.ctrl)
        self._act = np.array([m.actuator(n).id for n in MOTORES])
        jnt = m.actuator_trnid[self._act, 0]
        self._qadr = np.array(m.jnt_qposadr[jnt])
        self._vadr = np.array(m.jnt_dofadr[jnt])
        self._postura = jp.array(home.qpos[self._qadr])            # postura base de los 13 motores
        self._lo = jp.array(m.jnt_range[jnt, 0])
        self._hi = jp.array(m.jnt_range[jnt, 1])
        self._lo_suave = self._postura + 0.95 * (self._lo - self._postura)
        self._hi_suave = self._postura + 0.95 * (self._hi - self._postura)
        self._torso = m.body("torso").id
        self._pies = np.array([m.site(n).id for n in PIES])
        self._piso = m.geom("floor").id
        self._geom_pies = np.array([m.geom(n).id for n in PIES])
        rhyp = m.joint("RHipYawPitch")
        self._q_rhyp = int(rhyp.qposadr[0])
        self._q_lhyp = int(m.joint("LHipYawPitch").qposadr[0])
        self._n_obs = 3 + 3 + 13 + 13 + 13 + 2 + 2                   # 49 por paso
        if self._config.rumbo.activo:
            self._n_obs += 1                                           # + error de rumbo
        self._i_cadera = np.array([MOTORES.index(n) for n in ("LHipYawPitch", "LHipRoll", "RHipRoll")])

    # ------------------------------------------------------------------ propiedades que pide Playground
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return len(MOTORES)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # ------------------------------------------------------------------ sensores (como los da el NAO)
    def _arriba(self, data):
        """Vector 'arriba' del mundo visto desde el torso (lo que da la IMU: AngleX/AngleY)."""
        return data.xmat[self._torso][2, :]

    def _gyro(self, data):
        return data.qvel[3:6]                                          # velocidad angular en el marco del torso

    def _gyro_z_mundo(self, data):
        """Velocidad de giro alrededor de la vertical del mundo (lo que integra el NAO para su AngleZ)."""
        return (data.xmat[self._torso] @ data.qvel[3:6])[2]

    def _error_rumbo(self, info):
        e = jp.arctan2(jp.sin(info["rumbo_ref"] - info["rumbo_est"]), jp.cos(info["rumbo_ref"] - info["rumbo_est"]))
        return e

    def _vel_local(self, data):
        """Velocidad lineal en el marco del torso. SOLO para recompensa y critico: el NAO no la mide."""
        R = data.xmat[self._torso]
        return R.T @ data.qvel[0:3]

    # ------------------------------------------------------------------ comando
    def _comando(self, rng):
        k1, k2, k3 = jax.random.split(rng, 3)
        vx = jax.random.uniform(k1, minval=self._config.vx[0], maxval=self._config.vx[1])
        wz = jax.random.uniform(k2, minval=self._config.wz[0], maxval=self._config.wz[1])
        quieto = jax.random.bernoulli(k3, self._config.prob_quieto)
        return jp.where(quieto, jp.zeros(2), jp.array([vx, wz]))

    # ------------------------------------------------------------------ reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_q, k_yaw, k_cmd, k_f, k_lat, k_e1, k_deriva, k_ruido = jax.random.split(rng, 9)
        q = self._q_inicial
        # postura inicial con ruido y rumbo al azar (no siempre arranca igual)
        dq = jax.random.uniform(k_q, (13,), minval=-0.05, maxval=0.05)
        q = q.at[self._qadr].add(dq)
        q = q.at[self._q_rhyp].set(q[self._q_lhyp])                    # un solo motor HipYawPitch
        yaw = jax.random.uniform(k_yaw, minval=-jp.pi, maxval=jp.pi)
        q = q.at[3:7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
        data = mjx_env.make_data(self.mj_model, qpos=q, qvel=jp.zeros(self.mjx_model.nv),
                                 ctrl=self._ctrl_home, impl=self.mjx_model.impl.value)
        data = mjx.forward(self.mjx_model, data)

        f = jax.random.uniform(k_f, minval=self._config.frecuencia_paso[0], maxval=self._config.frecuencia_paso[1])
        cfg_e = self._config.empujones
        info = {
            "rng": rng,
            "comando": self._comando(k_cmd),
            "paso": 0,
            "fase": jp.array([0.0, jp.pi]),                            # pie izquierdo y derecho, desfasados media vuelta
            "dfase": 2 * jp.pi * f * self.dt,
            "accion_ant": jp.zeros(13),
            "acciones": jp.zeros((self._config.latencia_max + 1, 13)),  # memoria para simular el retraso
            "latencia": jax.random.randint(k_lat, (), 0, self._config.latencia_max + 1),
            "aire": jp.zeros(2),                                        # tiempo en el aire de cada pie
            "rumbo_ref": jp.zeros(()),                                  # rumbo deseado (relativo al de salida)
            "rumbo_est": jp.zeros(()),                                  # rumbo estimado integrando el giroscopio
            "deriva": jax.random.uniform(k_deriva, minval=-self._config.rumbo.deriva_gyro,
                                         maxval=self._config.rumbo.deriva_gyro),
            "contacto_ant": jp.ones(2, dtype=bool),
            "pies_xy_ant": data.site_xpos[self._pies][:, :2],
            "pasos_hasta_empujon": jp.round(jax.random.uniform(
                k_e1, minval=cfg_e.intervalo[0], maxval=cfg_e.intervalo[1]) / self.dt).astype(jp.int32),
            "empujon": jp.zeros(2),
        }
        metrics = {"recompensa/" + k: jp.zeros(()) for k in self._config.recompensa.escalas.keys()}
        historia = jp.zeros(self._config.historia * self._n_obs)
        obs = self._obs(data, info, historia, k_ruido)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ------------------------------------------------------------------ step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = state.info
        rng, k_cmd, k_ruido, k_emp, k_int = jax.random.split(info["rng"], 5)

        # 1) empujon: de vez en cuando se le suma una velocidad horizontal al torso (alguien lo empuja)
        cfg_e = self._config.empujones
        toca = (info["pasos_hasta_empujon"] <= 0) & cfg_e.activos
        ang = jax.random.uniform(k_emp, minval=0, maxval=2 * jp.pi)
        mag = jax.random.uniform(jax.random.fold_in(k_emp, 1), minval=cfg_e.magnitud[0], maxval=cfg_e.magnitud[1])
        empujon = jp.where(toca, mag * jp.array([jp.cos(ang), jp.sin(ang)]), jp.zeros(2))
        data = state.data.replace(qvel=state.data.qvel.at[0:2].add(empujon))
        nuevo_intervalo = jp.round(jax.random.uniform(
            k_int, minval=cfg_e.intervalo[0], maxval=cfg_e.intervalo[1]) / self.dt).astype(jp.int32)
        info["pasos_hasta_empujon"] = jp.where(toca, nuevo_intervalo, info["pasos_hasta_empujon"] - 1)

        # 2) retraso: se aplica la accion de hace 'latencia' ciclos (el NAO real no responde al instante)
        acciones = jp.roll(info["acciones"], 1, axis=0).at[0].set(action)
        aplicada = acciones[info["latencia"]]

        # 3) objetivo de los motores: postura base + correccion; el resto de motores queda en 'home'
        objetivo = jp.clip(self._postura + aplicada * self._config.action_scale, self._lo, self._hi)
        ctrl = self._ctrl_home.at[self._act].set(objetivo)
        data = mjx_env.step(self.mjx_model, data, ctrl, self.n_substeps)

        # 4) pies: contacto (planta a menos de 1 cm del piso) y tiempo en el aire
        pies = data.site_xpos[self._pies]
        contacto = pies[:, 2] < 0.01
        primer_contacto = contacto & ~info["contacto_ant"]
        aire = info["aire"] + self.dt
        vel_pies = (pies[:, :2] - info["pies_xy_ant"]) / self.dt

        # rumbo: el deseado integra el giro pedido; el estimado integra el giroscopio (con deriva)
        info["rumbo_ref"] = info["rumbo_ref"] + info["comando"][1] * self.dt
        info["rumbo_est"] = info["rumbo_est"] + (self._gyro_z_mundo(data) + info["deriva"]) * self.dt

        caida = self._caida(data)
        r = self._recompensas(data, action, info, caida, contacto, primer_contacto, aire, vel_pies)
        r = {k: v * self._config.recompensa.escalas[k] for k, v in r.items()}
        recompensa = sum(r.values()) * self.dt

        # 5) actualizar memoria
        info["aire"] = aire * ~contacto
        info["contacto_ant"] = contacto
        info["pies_xy_ant"] = pies[:, :2]
        info["accion_ant"] = action
        info["acciones"] = acciones
        info["empujon"] = empujon
        fase = info["fase"] + info["dfase"]
        info["fase"] = jp.fmod(fase + jp.pi, 2 * jp.pi) - jp.pi
        info["paso"] += 1
        info["rng"] = rng
        # comando nuevo cada ~10 s (asi aprende a cambiar de velocidad y de giro)
        cambiar = info["paso"] > 400
        info["comando"] = jp.where(cambiar, self._comando(k_cmd), info["comando"])
        info["paso"] = jp.where(caida | cambiar, 0, info["paso"])
        for k, v in r.items():
            state.metrics["recompensa/" + k] = v

        obs = self._obs(data, info, state.obs["state"], k_ruido)
        return state.replace(data=data, obs=obs, reward=recompensa, done=caida.astype(jp.float32))

    # ------------------------------------------------------------------ caida
    def _caida(self, data):
        inclinado = self._arriba(data)[2] < jp.cos(jp.deg2rad(self._config.inclinacion_max_deg))
        bajo = data.xpos[self._torso, 2] < self._config.altura_min
        nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return inclinado | bajo | nan

    # ------------------------------------------------------------------ observacion
    def _obs(self, data, info, historia, rng):
        c = self._config.ruido
        k = jax.random.split(rng, 4)

        def ruido(key, n, escala):
            return c.nivel * escala * jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)

        gyro = self._gyro(data)
        arriba = self._arriba(data)
        ang = data.qpos[self._qadr] - self._postura
        vel = data.qvel[self._vadr]
        fase = jp.concatenate([jp.cos(info["fase"][:1]), jp.sin(info["fase"][:1])])
        actual = jp.concatenate([
            gyro + ruido(k[0], 3, c.gyro),            # 3  giroscopio
            arriba + ruido(k[1], 3, c.gravedad),      # 3  inclinacion
            ang + ruido(k[2], 13, c.angulos),         # 13 angulos de los motores
            vel + ruido(k[3], 13, c.velocidades),     # 13 velocidades de los motores
            info["accion_ant"],                       # 13 accion anterior
            info["comando"],                          # 2  comando (vx, giro)
            fase,                                     # 2  reloj de marcha
        ])
        if self._config.rumbo.activo:                 # 1  error de rumbo (en la carrera: el psi de la vision)
            m = self._config.rumbo.error_max
            actual = jp.concatenate([actual, jp.clip(self._error_rumbo(info), -m, m)[None]])
        # historia: las ultimas N observaciones (la mas nueva primero)
        if isinstance(historia, dict):
            historia = historia["state"]
        estado = jp.roll(historia, self._n_obs).at[: self._n_obs].set(actual)
        # critico: todo lo anterior + cosas que el NAO real no mide (solo para entrenar)
        privilegiado = jp.concatenate([
            estado,
            gyro, arriba, ang, vel,                                   # sin ruido
            self._vel_local(data),                                    # velocidad real
            data.xpos[self._torso, 2:3],                              # altura del torso
            data.site_xpos[self._pies][:, 2],                         # altura de cada pie
            info["aire"],
            self.mjx_model.geom_friction[self._piso, 0:1],            # friccion del piso
            info["empujon"],
            info["latencia"][None].astype(jp.float32),
        ])
        return {"state": estado, "privileged_state": privilegiado}

    # ------------------------------------------------------------------ recompensa
    def _recompensas(self, data, action, info, caida, contacto, primer_contacto, aire, vel_pies):
        cfg = self._config.recompensa
        cmd = info["comando"]
        moverse = jp.linalg.norm(cmd) > 0.01
        v = self._vel_local(data)
        gyro = self._gyro(data)
        arriba = self._arriba(data)
        q = data.qpos[self._qadr]
        # separacion lateral entre pies, en el marco del torso
        R = data.xmat[self._torso]
        pies = data.site_xpos[self._pies]
        sep = jp.abs((R.T @ (pies[0] - pies[1]))[1])
        fuerza = data.actuator_force[self._act]
        vel_mot = data.qvel[self._vadr]
        rz = gait.get_rz(info["fase"], swing_height=self._config.altura_paso)
        return {
            # --- lo que queremos ---
            "seguir_vel": jp.exp(-(jp.square(cmd[0] - v[0]) + jp.square(v[1])) / cfg.sigma_vel),
            "seguir_giro": jp.exp(-jp.square(cmd[1] - gyro[2]) / cfg.sigma_giro),
            "tiempo_aire": jp.sum(jp.clip(aire - 0.1, max=0.2) * primer_contacto) * moverse,
            "fase_pies": jp.exp(-jp.sum(jp.square(pies[:, 2] - rz)) / 0.01) * moverse,
            # --- estabilidad ---
            "vel_z": jp.square(v[2]),
            "giro_xy": jp.sum(jp.square(gyro[:2])),
            "orientacion": jp.sum(jp.square(arriba[:2])),
            "pies_resbalan": jp.sum(jp.linalg.norm(vel_pies, axis=-1) * contacto),
            "pies_juntos": jp.clip(cfg.separacion_min - sep, min=0.0) / cfg.separacion_min,
            # --- suavidad y cuidado de los motores ---
            "cambio_accion": jp.sum(jp.square(action - info["accion_ant"])),
            "energia": jp.sum(jp.abs(fuerza * vel_mot)),
            "limites": jp.sum(jp.clip(self._lo_suave - q, min=0.0) + jp.clip(q - self._hi_suave, min=0.0)),
            "quieto": jp.sum(jp.abs(q - self._postura)) * ~moverse,
            "seguir_rumbo": jp.exp(-jp.square(self._error_rumbo(info)) / cfg.sigma_rumbo),
            "postura_cadera": jp.sum(jp.abs(q[self._i_cadera] - self._postura[self._i_cadera])),
            "caida": caida.astype(jp.float32),
        }


# ---------------------------------------------------------------------- aleatorizacion del mundo
def aleatorizar(model: mjx.Model, rng: jax.Array, friccion=(0.3, 1.2), masa=(0.9, 1.1), kp=(0.8, 1.2),
                motor_debil=None, ids_pie=None, id_piso=0, ids_act_pierna=None):
    """Cada robot (rng es un lote de llaves) recibe un mundo distinto.

    friccion: piso y pies (MuJoCo usa el MAXIMO de los dos, por eso se cambian ambos).
    masa: escala de cada pieza. kp: fuerza (rigidez) de cada motor.
    motor_debil: (min, max) -> UN motor de pierna al azar queda con esa fraccion de su torque maximo (calor).
    """
    nu = model.actuator_gainprm.shape[0]

    @jax.vmap
    def uno(rng):
        k = jax.random.split(rng, 5)
        mu = jax.random.uniform(k[0], minval=friccion[0], maxval=friccion[1])
        geom_friction = model.geom_friction.at[id_piso, 0].set(mu)
        geom_friction = geom_friction.at[ids_pie, 0].set(mu)
        body_mass = model.body_mass * jax.random.uniform(k[1], model.body_mass.shape, minval=masa[0], maxval=masa[1])
        escala_kp = jax.random.uniform(k[2], (nu,), minval=kp[0], maxval=kp[1])
        gainprm = model.actuator_gainprm.at[:, 0].multiply(escala_kp)
        biasprm = model.actuator_biasprm.at[:, 1].multiply(escala_kp)
        forcerange = model.actuator_forcerange
        if motor_debil is not None:
            cual = jax.random.choice(k[3], jp.asarray(ids_act_pierna))
            frac = jax.random.uniform(k[4], minval=motor_debil[0], maxval=motor_debil[1])
            forcerange = forcerange.at[cual].multiply(frac)
        return geom_friction, body_mass, gainprm, biasprm, forcerange

    geom_friction, body_mass, gainprm, biasprm, forcerange = uno(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    cambios = {"geom_friction": geom_friction, "body_mass": body_mass, "actuator_gainprm": gainprm,
               "actuator_biasprm": biasprm, "actuator_forcerange": forcerange}
    in_axes = in_axes.tree_replace({k: 0 for k in cambios})
    return model.tree_replace(cambios), in_axes
