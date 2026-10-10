from gymnasium.envs.registration import register

from src.envs.nao_walk_env import NaoWalkEnv

MAX_EPISODE_STEPS = 1000  # 20 s a 50 Hz

register(
    id="NaoWalk-v0",
    entry_point="src.envs.nao_walk_env:NaoWalkEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)

__all__ = ["NaoWalkEnv", "MAX_EPISODE_STEPS"]
