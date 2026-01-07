import numpy as np
from typing import Optional, List

from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback


# -----------------------------
# Callback para registrar recompensas (pueden estar normalizadas)
# -----------------------------
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
        return True

# Wrapper para registrar recompensas crudas por episodio (antes de VecNormalize)
class RawRewardTrackerVecWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self._raw_returns = np.zeros(self.num_envs, dtype=np.float32)

    def reset(self):
        self._raw_returns[:] = 0.0
        return self.venv.reset()

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # 'rewards' aquí son crudos porque estamos ANTES de VecNormalize
        self._raw_returns += rewards
        for i, done in enumerate(dones):
            if done:
                if isinstance(infos[i], dict):
                    # Inyecta recompensa REAL del episodio en info
                    infos[i]["raw_episode_reward"] = float(self._raw_returns[i])
                self._raw_returns[i] = 0.0
        return obs, rewards, dones, infos

# Callback robusto: acumula recompensas por env usando dones y rewards
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, n_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self.n_envs = n_envs
        self._ep_returns = None
        self._ep_lengths = None
        self.episode_rewards = []
        self.episode_lengths = []
    def _on_training_start(self) -> None:
        self._ep_returns = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(self.n_envs, dtype=np.int32)
    def _on_step(self) -> bool:      
        dones  = self.locals.get("dones", None)
        rewards = self.locals.get("rewards", None)  # OJO: pueden estar normalizadas si ya pasó VecNormalize
        infos  = self.locals.get("infos", None)
        if dones is None or rewards is None:
            return True
        self._ep_returns += rewards
        self._ep_lengths += 1
        for i, d in enumerate(dones):
            if d:
                # Si el wrapper aportó recompensa REAL se usa
                # si no, usa la acumulada (posible normalizada)
                raw = None
                if isinstance(infos, (list, tuple)) and len(infos) > i and isinstance(infos[i], dict):
                    raw = infos[i].get("raw_episode_reward", None)
                value = float(raw) if raw is not None else float(self._ep_returns[i])
                self.episode_rewards.append(value)
                self.episode_lengths.append(int(self._ep_lengths[i]))
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0
        return True