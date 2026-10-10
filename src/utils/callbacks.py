"""Callbacks de entrenamiento: progreso en consola, terminos de recompensa y guardado de VecNormalize."""

import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ProgressCallback(BaseCallback):
    """Imprime una linea por actualizacion: pasos, velocidad, retorno y duracion de episodio."""

    def __init__(self, total_timesteps: int, dt: float, print_every_s: float = 30.0):
        super().__init__()
        self.total = total_timesteps
        self.dt = dt
        self.print_every_s = print_every_s
        self._t0 = self._last_print = None

    def _on_training_start(self) -> None:
        self._t0 = self._last_print = time.perf_counter()
        self._start_steps = self.num_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        now = time.perf_counter()
        done = self.num_timesteps >= self.total
        if now - self._last_print < self.print_every_s and not done:
            return
        self._last_print = now
        elapsed = now - self._t0
        fps = (self.num_timesteps - self._start_steps) / max(elapsed, 1e-9)
        eta_min = max(self.total - self.num_timesteps, 0) / max(fps, 1e-9) / 60
        buf = self.model.ep_info_buffer
        rew = np.mean([e["r"] for e in buf]) if buf else float("nan")
        length_s = np.mean([e["l"] for e in buf]) * self.dt if buf else float("nan")
        print(f"[{self.num_timesteps / 1e6:6.2f}M / {self.total / 1e6:.1f}M] {fps:5.0f} pasos/s | "
              f"retorno medio {rew:8.1f} | episodio medio {length_s:5.2f} s | "
              f"transcurrido {elapsed / 60:5.1f} min | faltan ~{eta_min:5.1f} min", flush=True)


class RewardTermsCallback(BaseCallback):
    """Registra en TensorBoard la media de cada termino de recompensa y los motivos de caida."""

    def __init__(self):
        super().__init__()
        self._sums = defaultdict(float)
        self._count = 0
        self._falls = Counter()
        self._episodes = 0

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            for name, value in info.get("reward_terms", {}).items():
                self._sums[name] += value
            self._count += 1
            if done:
                self._episodes += 1
                if info.get("fall_reason"):
                    self._falls[info["fall_reason"]] += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._count:
            for name, total in self._sums.items():
                self.logger.record(f"reward_terms/{name}", total / self._count)
        if self._episodes:
            self.logger.record("episodes/fall_rate", sum(self._falls.values()) / self._episodes)
            for reason, n in self._falls.items():
                self.logger.record(f"episodes/fall_{reason}", n / self._episodes)
        self._sums.clear()
        self._count = 0
        self._falls.clear()
        self._episodes = 0


class SaveVecNormalizeCallback(BaseCallback):
    """Guarda las estadisticas de normalizacion junto al mejor modelo (lo llama EvalCallback)."""

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            vec_normalize.save(str(self.save_path))
        return True
