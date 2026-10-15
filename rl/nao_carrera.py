# -*- coding: utf-8 -*-
"""Entorno de entrenamiento del NAO V6 para la Carrera (MuJoCo MJX, GPU).

Estructura copiada de los entornos "Joystick" de MuJoCo Playground (G1 y OP3), adaptada al NAO y a la carrera:
  - El actor solo ve sensores que tiene el NAO real (nada de velocidad lineal ni posicion).
  - Accion cada 24 ms (2 ciclos de 12 ms del NAO), 13 motores: piernas (11) + ShoulderPitch (2); 15 con codos=True (+ ElbowRoll).
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
CODOS = ("LElbowRoll", "RElbowRoll")      # desde nao_v01e (config codos=True): doblar/estirar cada codo
# Motores reales del NAO (desde nao_v05a, config motores_reales=True). Fuente: tabla de motores de NAO de United
# Robotics / Aldebaran (motor de pierna: 8700 rpm sin carga; brazo: 10700 rpm; reducciones por articulacion).
# vel_max = rpm sin carga / reduccion (rad/s): a esa velocidad el motor ya no da fuerza (contra-FEM).
# par_max = torque maximo en la articulacion (N m); en piernas manda el ±5 N m del modelo (el bloqueo real, ~8-13 N m,
# no se usa); en brazos el bloqueo real (≈2.2-2.5 N m) es menor que el ±5 / ±3 del modelo.
#   piernas, reduccion 201.3 (HipYawPitch, HipRoll, AnkleRoll): 8700/201.3 rpm = 4.53 rad/s
#   piernas, reduccion 130.85 (HipPitch, KneePitch, AnklePitch): 8700/130.85 rpm = 6.96 rad/s
#   ShoulderPitch, reduccion 150.27: 10700/150.27 rpm = 7.46 rad/s, 14.3 mN m x 150.27 = 2.15 N m
#   ElbowRoll, reduccion 173.22: 10700/173.22 rpm = 6.47 rad/s, 14.3 mN m x 173.22 = 2.48 N m
VEL_MAX = {"LHipYawPitch": 4.53, "LHipRoll": 4.53, "RHipRoll": 4.53, "LAnkleRoll": 4.53, "RAnkleRoll": 4.53,
           "LHipPitch": 6.96, "RHipPitch": 6.96, "LKneePitch": 6.96, "RKneePitch": 6.96,
           "LAnklePitch": 6.96, "RAnklePitch": 6.96,
           "LShoulderPitch": 7.46, "RShoulderPitch": 7.46, "LElbowRoll": 6.47, "RElbowRoll": 6.47}
PAR_MAX = {"LShoulderPitch": 2.15, "RShoulderPitch": 2.15, "LElbowRoll": 2.48, "RElbowRoll": 2.48}   # el resto: el del modelo
PIES = ("left_foot", "right_foot")          # sites en la planta de cada pie


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.024,                 # 24 ms por accion (2 ciclos de 12 ms del NAO)
        sim_dt=0.002,                  # fisica a 2 ms -> 12 pasos de fisica por accion
        episode_length=1000,           # 1000 x 24 ms = 24 s
        impl="jax",
        action_scale=0.3,              # rad: objetivo = postura_base + accion * escala
        motores_reales=False,          # desde nao_v05a: el torque cae con la velocidad del motor (tope vel_max) y brazos
                                       # con su torque real (VEL_MAX / PAR_MAX)
        codos=False,                   # desde nao_v01e: la politica tambien mueve LElbowRoll y RElbowRoll (15 motores)
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
        # --- vision simulada (desde nao_v06) ---
        # Reemplaza el error de rumbo del giroscopio por lo que mide la camara respecto a las lineas del carril:
        # psi (angulo con el carril) y d (distancia lateral al centro; + = a la izquierda). El carril es virtual: su
        # direccion es el rumbo deseado (integra el giro pedido) y d integra el movimiento lateral respecto a ella.
        # Ruido medido con el estimador real en simulacion (sigma 1.6 cm / 1.2 grados), 10 Hz, 1 cuadro de retraso,
        # cuadros perdidos, y salida chueca (+-10 cm, +-8 grados), como cuando el capitan coloca el robot.
        vision=config_dict.create(
            activa=False,
            sigma_d=0.016,             # m
            sigma_psi=0.021,           # rad (1.2 grados)
            periodo=4,                 # pasos de 24 ms entre cuadros (~10 Hz)
            prob_perdida=0.05,         # cuadros sin medicion (se mantiene la anterior)
            d0_max=0.10,               # m: salida corrida del centro
            psi0_max=0.14,             # rad (8 grados): salida chueca
            d_max=0.5,                 # m: recorte de lo que ve la politica
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
            sigma_carril=0.01,         # m^2 (~10 cm): seguir_carril, desde nao_v06
            sigma_pies=0.01,           # m^2: tolerancia de fase_pies. 0.01 viene del G1 y es muy laxa para pasos de
                                       # 3 cm (arrastrar el pie da 97 % del premio). nao_v01d usa 0.0005 (61 %).
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
                seguir_carril=0.0,     # desde nao_v06: premio por ir por el centro del carril (d real)
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
        self._motores = MOTORES + (CODOS if self._config.codos else ())
        n = len(self._motores)                                      # 13, o 15 con codos
        self._act = np.array([m.actuator(nombre).id for nombre in self._motores])
        jnt = m.actuator_trnid[self._act, 0]
        self._qadr = np.array(m.jnt_qposadr[jnt])
        self._vadr = np.array(m.jnt_dofadr[jnt])
        self._postura = jp.array(home.qpos[self._qadr])            # postura base de los motores que controla
        self._vel_max = jp.array([VEL_MAX[nombre] for nombre in self._motores])
        self._par_max = jp.array([PAR_MAX.get(nombre, 1e6) for nombre in self._motores])
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
        self._n_obs = 3 + 3 + n + n + n + 2 + 2                      # 49 por paso (55 con codos)
        if self._config.vision.activa:
            self._n_obs += 2                                           # + psi y d de la camara
        elif self._config.rumbo.activo:
            self._n_obs += 1                                           # + error de rumbo
        self._i_cadera = np.array([MOTORES.index(c) for c in ("LHipYawPitch", "LHipRoll", "RHipRoll")])

    # ------------------------------------------------------------------ propiedades que pide Playground
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return len(self._motores)

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
        """Sin vision: rumbo deseado - rumbo estimado con el giroscopio. Con vision: psi real (para la recompensa)."""
        est = info["rumbo_real"] if self._config.vision.activa else info["rumbo_est"]
        e = jp.arctan2(jp.sin(info["rumbo_ref"] - est), jp.cos(info["rumbo_ref"] - est))
        return e

    def _yaw(self, data):
        R = data.xmat[self._torso]
        return jp.arctan2(R[1, 0], R[0, 0])

    def _medir_carril(self, info, rng):
        """Lo que daria la camara en este instante: (psi, d) reales + ruido."""
        v = self._config.vision
        k1, k2 = jax.random.split(rng)
        psi = self._error_rumbo(info) + v.sigma_psi * jax.random.normal(k1)
        d = info["carril_d"] + v.sigma_d * jax.random.normal(k2)
        return jp.array([psi, d])

    def etiquetas(self):
        """Nombre de cada entrada de la observacion (actor, critico). Sirve para trasplantar pesos entre versiones."""
        m = self._motores
        blk = (["gyro%d" % i for i in range(3)] + ["arriba%d" % i for i in range(3)] + ["ang_" + n for n in m]
               + ["vel_" + n for n in m] + ["acc_" + n for n in m] + ["cmd_vx", "cmd_wz", "fase_cos", "fase_sin"])
        if self._config.vision.activa:
            blk += ["vis_psi", "vis_d"]
        elif self._config.rumbo.activo:
            blk += ["rumbo"]
        estado = ["h%d_%s" % (h, e) for h in range(self._config.historia) for e in blk]
        extras = (["p_gyro%d" % i for i in range(3)] + ["p_arriba%d" % i for i in range(3)]
                  + ["p_ang_" + n for n in m] + ["p_vel_" + n for n in m] + ["p_vlin%d" % i for i in range(3)]
                  + ["p_altura", "p_pie_z0", "p_pie_z1", "p_aire0", "p_aire1", "p_friccion", "p_empujon0",
                     "p_empujon1", "p_latencia"])
        if self._config.vision.activa:
            extras += ["p_carril_d", "p_carril_psi"]
        return estado, estado + extras

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
        dq = jax.random.uniform(k_q, (self.action_size,), minval=-0.05, maxval=0.05)
        q = q.at[self._qadr].add(dq)
        q = q.at[self._q_rhyp].set(q[self._q_lhyp])                    # un solo motor HipYawPitch
        yaw = jax.random.uniform(k_yaw, minval=-jp.pi, maxval=jp.pi)
        q = q.at[3:7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
        data = mjx_env.make_data(self.mj_model, qpos=q, qvel=jp.zeros(self.mjx_model.nv),
                                 ctrl=self._ctrl_home, impl=self.mjx_model.impl.value)
        data = mjx.forward(self.mjx_model, data)

        f = jax.random.uniform(k_f, minval=self._config.frecuencia_paso[0], maxval=self._config.frecuencia_paso[1])
        v = self._config.vision
        k_d0, k_psi0, k_vis = jax.random.split(jax.random.fold_in(k_yaw, 7), 3)
        d0 = jax.random.uniform(k_d0, minval=-v.d0_max, maxval=v.d0_max) * v.activa
        psi0 = jax.random.uniform(k_psi0, minval=-v.psi0_max, maxval=v.psi0_max) * v.activa
        cfg_e = self._config.empujones
        info = {
            "rng": rng,
            "comando": self._comando(k_cmd),
            "paso": 0,
            "fase": jp.array([0.0, jp.pi]),                            # pie izquierdo y derecho, desfasados media vuelta
            "dfase": 2 * jp.pi * f * self.dt,
            "accion_ant": jp.zeros(self.action_size),
            "acciones": jp.zeros((self._config.latencia_max + 1, self.action_size)),  # memoria para simular el retraso
            "latencia": jax.random.randint(k_lat, (), 0, self._config.latencia_max + 1),
            "aire": jp.zeros(2),                                        # tiempo en el aire de cada pie
            "rumbo_ref": psi0 * jp.ones(()),                            # rumbo deseado (relativo al de salida)
            "yaw0": yaw * jp.ones(()),                                  # rumbo de salida en el mundo
            "rumbo_real": jp.zeros(()),                                 # rumbo real relativo al de salida
            "carril_d": d0 * jp.ones(()),                               # distancia real al centro del carril
            "vis_cuenta": jp.zeros((), jp.int32),                       # pasos hasta el proximo cuadro
            "vis_muestra": jp.array([psi0, d0]),                        # ultimo cuadro tomado (se publica despues)
            "vis_obs": jp.array([psi0, d0]),                            # lo que ve la politica
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
        data = self._fisica(data, ctrl)

        # 4) pies: contacto (planta a menos de 1 cm del piso) y tiempo en el aire
        pies = data.site_xpos[self._pies]
        contacto = pies[:, 2] < 0.01
        primer_contacto = contacto & ~info["contacto_ant"]
        aire = info["aire"] + self.dt
        vel_pies = (pies[:, :2] - info["pies_xy_ant"]) / self.dt

        # rumbo: el deseado integra el giro pedido; el estimado integra el giroscopio (con deriva)
        info["rumbo_ref"] = info["rumbo_ref"] + info["comando"][1] * self.dt
        info["rumbo_est"] = info["rumbo_est"] + (self._gyro_z_mundo(data) + info["deriva"]) * self.dt
        if self._config.vision.activa:
            info = self._paso_vision(data, info, k_cmd)

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

    def _paso_vision(self, data, info, rng):
        """Carril virtual y camara: d integra la velocidad lateral respecto al rumbo deseado; cada 'periodo' pasos
        se publica el cuadro anterior (1 cuadro de retraso) y se toma uno nuevo; a veces el cuadro se pierde."""
        v = self._config.vision
        info["rumbo_real"] = self._yaw(data) - info["yaw0"]
        ref = info["yaw0"] + info["rumbo_ref"]
        normal = jp.array([-jp.sin(ref), jp.cos(ref)])
        info["carril_d"] = info["carril_d"] + jp.dot(data.qvel[0:2], normal) * self.dt
        k_med, k_perd = jax.random.split(jax.random.fold_in(rng, 11))
        toca = info["vis_cuenta"] <= 0
        perdido = jax.random.bernoulli(k_perd, v.prob_perdida)
        info["vis_obs"] = jp.where(toca & ~perdido, info["vis_muestra"], info["vis_obs"])
        info["vis_muestra"] = jp.where(toca, self._medir_carril(info, k_med), info["vis_muestra"])
        info["vis_cuenta"] = jp.where(toca, v.periodo - 1, info["vis_cuenta"] - 1)
        return info

    # ------------------------------------------------------------------ fisica
    def _fisica(self, data, ctrl):
        """n_substeps pasos de MuJoCo. Con motores_reales, antes de cada paso el limite de torque de cada motor
        controlado baja con su velocidad: tope = par_max * (1 - |w| / vel_max) (recta de un motor DC)."""
        if not self._config.motores_reales:
            return mjx_env.step(self.mjx_model, data, ctrl, self.n_substeps)
        base = self.mjx_model.actuator_forcerange               # ya trae el motor debil si se aleatorizo
        tope = jp.minimum(base[self._act, 1], self._par_max)

        def sub(d, _):
            w = d.qvel[self._vadr]
            lim = tope * jp.clip(1.0 - jp.abs(w) / self._vel_max, 0.0, 1.0)
            fr = base.at[self._act, 0].set(-lim).at[self._act, 1].set(lim)
            m = self.mjx_model.tree_replace({"actuator_forcerange": fr})
            return mjx.step(m, d.replace(ctrl=ctrl)), None

        return jax.lax.scan(sub, data, (), self.n_substeps)[0]

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
            ang + ruido(k[2], self.action_size, c.angulos),   # angulos de los motores
            vel + ruido(k[3], self.action_size, c.velocidades),  # velocidades de los motores
            info["accion_ant"],                       # accion anterior
            info["comando"],                          # 2  comando (vx, giro)
            fase,                                     # 2  reloj de marcha
        ])
        if self._config.vision.activa:                # 2  psi y d de la camara (con ruido, 10 Hz, retraso)
            m, dm = self._config.rumbo.error_max, self._config.vision.d_max
            actual = jp.concatenate([actual, jp.array([jp.clip(info["vis_obs"][0], -m, m),
                                                        jp.clip(info["vis_obs"][1], -dm, dm)])])
        elif self._config.rumbo.activo:               # 1  error de rumbo (en la carrera: el psi de la vision)
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
        if self._config.vision.activa:
            privilegiado = jp.concatenate([privilegiado, jp.array([info["carril_d"], self._error_rumbo(info)])])
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
            "fase_pies": jp.exp(-jp.sum(jp.square(pies[:, 2] - rz)) / cfg.sigma_pies) * moverse,
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
            "seguir_carril": jp.exp(-jp.square(info["carril_d"]) / cfg.sigma_carril),
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


def salida_recta(info):
    """Para evaluar: despues de poner el robot mirando a lo largo de la pista (yaw 0), el carril virtual coincide con
    la pista y el robot sale centrado y derecho (sin la salida chueca del entrenamiento)."""
    cero = jp.zeros_like(info["rumbo_ref"])
    return {"yaw0": cero, "rumbo_ref": cero, "rumbo_real": cero, "carril_d": cero,
            "vis_muestra": jp.zeros_like(info["vis_muestra"]), "vis_obs": jp.zeros_like(info["vis_obs"])}


# ---------------------------------------------------------------------- trasplante de pesos entre versiones
ALIAS = {"vis_psi": "rumbo"}       # psi de la camara ocupa el lugar del error de rumbo del giroscopio


def trasplantar(params, etq_vieja, etq_nueva):
    """Adapta (normalizador, politica, valor) a una observacion con otras entradas. Cada entrada nueva copia la fila
    de pesos y la estadistica de la entrada vieja del mismo nombre (o de su alias); las que no existian empiezan con
    peso 0 (al principio no cambian nada) y estadistica media 0 / desviacion 1."""
    import numpy as onp
    norm, pol, val = params

    def mapa(vieja, nueva):
        idx = {e: i for i, e in enumerate(vieja)}
        out = []
        for e in nueva:
            base = e
            for a, b in ALIAS.items():
                if e.endswith("_" + a) or e == a:
                    base = e[: len(e) - len(a)] + b
            out.append(idx.get(e, idx.get(base, -1)))
        return onp.array(out)

    mapas = {"state": mapa(etq_vieja[0], etq_nueva[0]), "privileged_state": mapa(etq_vieja[1], etq_nueva[1])}
    if hasattr(norm.count, "hi"):               # contador de 64 bits partido en dos (brax)
        cuenta = float(onp.asarray(norm.count.hi, dtype=onp.float64) * 2.0 ** 32
                       + onp.asarray(norm.count.lo, dtype=onp.float64))
    else:
        cuenta = float(onp.asarray(norm.count, dtype=onp.float64))

    def pasar(arr, m, relleno):
        arr = onp.asarray(arr)
        nuevo = onp.full((len(m),) + arr.shape[1:], relleno, dtype=arr.dtype)
        ok = m >= 0
        nuevo[ok] = arr[m[ok]]
        return nuevo

    norm = norm.replace(
        mean={k: pasar(norm.mean[k], mapas[k], 0.0) for k in mapas},
        std={k: pasar(norm.std[k], mapas[k], 1.0) for k in mapas},
        summed_variance={k: pasar(norm.summed_variance[k], mapas[k], cuenta) for k in mapas})

    def capa0(red, m):
        red = jax.tree_util.tree_map(lambda x: x, red)
        k = red["params"]["hidden_0"]["kernel"]
        red["params"]["hidden_0"] = dict(red["params"]["hidden_0"], kernel=pasar(k, m, 0.0))
        return red

    return norm, capa0(pol, mapas["state"]), capa0(val, mapas["privileged_state"])
