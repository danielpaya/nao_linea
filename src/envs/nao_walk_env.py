"""Entorno Gymnasium para aprender a caminar con el NAO en MuJoCo.

La politica controla las articulaciones de las piernas a 50 Hz enviando desviaciones respecto a
la pose "home" (de pie, rodillas flexionadas); los servos PD del modelo las siguen a 500 Hz.
"""

import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

SCENE_PATH = Path(__file__).resolve().parents[2] / "assets" / "nao" / "scene.xml"

# Actuadores que controla la politica. HipYawPitch queda fija en 0 (desestabiliza el torso y no
# aporta a caminar recto); cabeza y brazos se mantienen en "home".
LEG_ACTUATORS = (
    "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
)

# Opcional (control_arms=True): hombros para que la politica pueda balancear los brazos.
ARM_ACTUATORS = ("LShoulderPitch", "RShoulderPitch")

N_ACT = len(LEG_ACTUATORS)  # dimension de la accion por defecto (solo piernas)

# Desviacion maxima respecto a "home" [rad] para accion = +-1, por tipo de articulacion.
# Pitch (cadera, rodilla, tobillo) necesita amplitud para dar pasos; roll solo traslada el peso.
# Todos los objetivos resultantes quedan dentro de los limites articulares del NAO.
ACTION_SCALES = {
    "HipRoll": 0.25,
    "HipPitch": 0.5,
    "KneePitch": 0.5,
    "AnklePitch": 0.4,
    "AnkleRoll": 0.25,
    "ShoulderPitch": 0.5,
}


def obs_layout(n_act: int = N_ACT) -> tuple:
    """Componentes de la observacion, en orden, con su dimension. Las escalas dejan cada valor ~O(1)."""
    return (
        ("gravedad_proyectada", 3),   # vector "abajo" en el marco del torso (inclinacion)
        ("vel_angular_torso", 3),     # giroscopio [rad/s] * 0.25
        ("vel_lineal_torso", 3),      # velocidad en el marco del torso [m/s] * 2
        ("q_articular", n_act),       # angulo - angulo en home [rad]
        ("dq_articular", n_act),      # velocidad articular [rad/s] * 0.05
        ("accion_anterior", n_act),   # ultima accion en [-1, 1]
        ("contacto_pies", 2),         # izquierdo, derecho (0/1)
        ("fase_marcha", 2),           # sin, cos del reloj de marcha
    )


OBS_LAYOUT = obs_layout()
# Pesos de la recompensa (por paso de control). Positivos = premios, negativos = penalizaciones.
REWARD_WEIGHTS = {
    "velocidad_avance": 1.0,     # exp(-((vx - v_objetivo) / sigma)^2)
    "patron_marcha": 0.5,        # fraccion de pies en la fase correcta del reloj de marcha
    "vivo": 0.1,                 # constante mientras no cae
    "inclinacion": -2.0,         # gx^2 + gy^2 de la gravedad proyectada
    "altura": -50.0,             # (z_torso - z_objetivo)^2
    "vel_lateral": -10.0,        # vy^2
    "vel_giro": -0.5,            # wz^2
    "cambio_accion": -0.02,      # ||a_t - a_{t-1}||^2
    "torque": -0.001,            # sum(tau^2)
    "caida": -10.0,              # una vez, al terminar por caida
}
TARGET_SPEED = 0.15   # m/s
SPEED_SIGMA = 0.1     # m/s
TARGET_HEIGHT = 0.31  # m (de pie en "home": ~0.316)

ANG_VEL_SCALE = 0.25
LIN_VEL_SCALE = 2.0
JOINT_VEL_SCALE = 0.05


def gait_pattern_reward(contacts: np.ndarray, phase: float, stance_fraction: float = 0.5) -> float:
    """Fraccion de pies (0, 0.5 o 1) cuyo contacto coincide con el reloj de marcha.

    El pie izquierdo apoya en la fase [0, stance_fraction) y el derecho en [0.5, 0.5 + stance_fraction)
    (modulo 1). Con 0.5 nunca hay doble apoyo; con 0.6 hay un 20% de doble apoyo, como al caminar.
    """
    left_stance = phase < stance_fraction
    right_stance = phase >= 0.5 or phase < stance_fraction - 0.5
    expected = np.array([left_stance, right_stance], dtype=float)
    return float(np.mean(contacts == expected))


def obs_slices(n_act: int = N_ACT) -> dict[str, slice]:
    """Posicion de cada componente dentro del vector de observacion."""
    slices, start = {}, 0
    for name, dim in obs_layout(n_act):
        slices[name] = slice(start, start + dim)
        start += dim
    return slices


class NaoWalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        frame_skip: int = 10,
        action_scale: float = 1.0,
        reset_noise: float = 0.02,
        gait_period: float = 0.6,
        fall_height: float = 0.22,
        fall_tilt_deg: float = 40.0,
        target_speed: float = TARGET_SPEED,
        stance_fraction: float = 0.5,
        control_arms: bool = False,
        reward_weights: dict | None = None,
        render_mode: str | None = None,
        render_size: tuple[int, int] = (640, 480),
    ):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.action_scale = action_scale
        self.reset_noise = reset_noise
        self.gait_period = gait_period
        self.fall_height = fall_height
        self.fall_tilt_deg = fall_tilt_deg
        self.target_speed = target_speed
        unknown = set(reward_weights or {}) - set(REWARD_WEIGHTS)
        if unknown:
            raise ValueError(f"Terminos de recompensa desconocidos: {sorted(unknown)}")
        self.reward_weights = {**REWARD_WEIGHTS, **(reward_weights or {})}
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode debe ser uno de {self.metadata['render_modes']}")
        self.render_mode = render_mode
        self.render_size = render_size
        self._renderer = None
        self._viewer = None
        self._last_render_time = None

        home = self.model.key("home").id
        self._home_qpos = self.model.key_qpos[home].copy()
        self._home_ctrl = self.model.key_ctrl[home].copy()
        self.stance_fraction = stance_fraction
        self.controlled_actuators = LEG_ACTUATORS + (ARM_ACTUATORS if control_arms else ())
        self.n_act = len(self.controlled_actuators)
        self._act_ids = np.array([self.model.actuator(name).id for name in self.controlled_actuators])
        # "LHipPitch" -> "HipPitch"; action_scale multiplica todas las escalas (para experimentos).
        self._action_scales = action_scale * np.array([ACTION_SCALES[name[1:]] for name in self.controlled_actuators])
        joint_ids = self.model.actuator_trnid[self._act_ids, 0]
        self._qpos_ids = self.model.jnt_qposadr[joint_ids]
        self._qvel_ids = self.model.jnt_dofadr[joint_ids]
        self._torso_id = self.model.body("torso").id
        self._floor_id = self.model.geom("floor").id
        self._foot_ids = (self.model.geom("left_foot").id, self.model.geom("right_foot").id)

        self._prev_action = np.zeros(self.n_act)
        self._step_count = 0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_act,), dtype=np.float32)
        self._reset_to_home()
        obs_dim = self._get_obs().shape[0]
        assert obs_dim == sum(dim for _, dim in obs_layout(self.n_act))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ API Gymnasium

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_to_home()
        n_joints = self.model.nq - 7
        self.data.qpos[7:] += self.np_random.uniform(-self.reset_noise, self.reset_noise, n_joints)
        mujoco.mj_forward(self.model, self.data)
        self._prev_action[:] = 0.0
        self._step_count = 0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        fall_reason = self.fall_reason()
        terminated = fall_reason is not None
        # La recompensa se calcula antes de avanzar el reloj y guardar la accion (usa a_{t-1}).
        terms = self.reward_terms(action, terminated)
        reward = sum(terms.values())
        self._prev_action[:] = action
        self._step_count += 1
        obs = self._get_obs()
        if fall_reason == "inestabilidad_numerica":
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            reward = self.reward_weights["caida"]
        info = {"fall_reason": fall_reason, "reward_terms": terms}
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                width, height = self.render_size
                self._renderer = mujoco.Renderer(self.model, height, width)
            self._renderer.update_scene(self.data, camera="track")
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._viewer.cam.trackbodyid = self._torso_id
                self._viewer.cam.distance, self._viewer.cam.azimuth, self._viewer.cam.elevation = 1.2, 120, -15
            self._viewer.sync()
            # Mantener tiempo real: un paso de control cada dt segundos.
            now = time.perf_counter()
            if self._last_render_time is not None:
                time.sleep(max(0.0, self.dt - (now - self._last_render_time)))
            self._last_render_time = time.perf_counter()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------ Partes del MDP

    def _get_obs(self) -> np.ndarray:
        torso_rot = self.data.xmat[self._torso_id].reshape(3, 3)  # marco torso -> mundo
        return np.concatenate([
            torso_rot.T @ np.array([0.0, 0.0, -1.0]),
            self.data.sensor("gyro").data * ANG_VEL_SCALE,
            self.data.sensor("local_linvel").data * LIN_VEL_SCALE,
            self.data.qpos[self._qpos_ids] - self._home_qpos[self._qpos_ids],
            self.data.qvel[self._qvel_ids] * JOINT_VEL_SCALE,
            self._prev_action,
            self.foot_contacts(),
            self.gait_phase_features(),
        ]).astype(np.float32)

    def foot_contacts(self) -> np.ndarray:
        """1.0 si el pie (izquierdo, derecho) toca el suelo."""
        contact = np.zeros(2)
        for c in self.data.contact[: self.data.ncon]:
            if self._floor_id in (c.geom1, c.geom2):
                for k, foot in enumerate(self._foot_ids):
                    if foot in (c.geom1, c.geom2):
                        contact[k] = 1.0
        return contact

    def gait_phase(self) -> float:
        """Fase del reloj de marcha en [0, 1)."""
        return (self._step_count * self.dt / self.gait_period) % 1.0

    def gait_phase_features(self) -> np.ndarray:
        angle = 2 * np.pi * self.gait_phase()
        return np.array([np.sin(angle), np.cos(angle)])

    def action_to_targets(self, action: np.ndarray) -> np.ndarray:
        """Angulos objetivo [rad] de las articulaciones controladas para una accion en [-1, 1]."""
        return self._home_ctrl[self._act_ids] + self._action_scales * action

    def _apply_action(self, action: np.ndarray) -> None:
        # Cabeza, brazos y HipYawPitch se quedan en "home"; MuJoCo limita ctrl a ctrlrange.
        self.data.ctrl[:] = self._home_ctrl
        self.data.ctrl[self._act_ids] = self.action_to_targets(action)

    def reward_terms(self, action: np.ndarray, fallen: bool) -> dict[str, float]:
        """Cada termino de la recompensa ya multiplicado por su peso."""
        w = self.reward_weights
        vx, vy = self.data.qvel[0], self.data.qvel[1]  # velocidad del torso en el mundo
        wz = self.data.qvel[5]  # velocidad de giro (eje z)
        torso_rot = self.data.xmat[self._torso_id].reshape(3, 3)
        gravity = torso_rot.T @ np.array([0.0, 0.0, -1.0])
        torque = self.data.actuator_force[self._act_ids]
        raw = {
            "velocidad_avance": np.exp(-(((vx - self.target_speed) / SPEED_SIGMA) ** 2)),
            "patron_marcha": gait_pattern_reward(self.foot_contacts(), self.gait_phase(), self.stance_fraction),
            "vivo": 1.0,
            "inclinacion": gravity[0] ** 2 + gravity[1] ** 2,
            "altura": (self.data.qpos[2] - TARGET_HEIGHT) ** 2,
            "vel_lateral": vy**2,
            "vel_giro": wz**2,
            "cambio_accion": float(np.sum((action - self._prev_action) ** 2)),
            "torque": float(np.sum(torque**2)),
            "caida": float(fallen),
        }
        return {name: float(w[name] * value) for name, value in raw.items()}

    def torso_tilt_deg(self) -> float:
        """Angulo entre el eje vertical del torso y la vertical del mundo."""
        cos_tilt = self.data.xmat[self._torso_id][8]  # componente z del eje z del torso
        return float(np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0))))

    def fall_reason(self) -> str | None:
        """Motivo de la caida, o None si el robot sigue en pie."""
        bad_qacc = self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
        if bad_qacc or not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            return "inestabilidad_numerica"
        if self.data.qpos[2] < self.fall_height:
            return "altura"
        if self.torso_tilt_deg() > self.fall_tilt_deg:
            return "inclinacion"
        for c in self.data.contact[: self.data.ncon]:
            if self._floor_id in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self._floor_id else c.geom1
                if other not in self._foot_ids:
                    return "contacto_" + self.model.geom(other).name
        return None

    # ------------------------------------------------------------------ Utilidades

    def _reset_to_home(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
