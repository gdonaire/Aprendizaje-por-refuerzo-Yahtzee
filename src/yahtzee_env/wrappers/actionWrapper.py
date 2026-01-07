
# ActionMaskRepairVecWrapper.py
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

class ActionMaskRepairVecWrapper(VecEnvWrapper):
    """
    Repara acciones inválidas en entornos vectorizados (DummyVecEnv/SubprocVecEnv).
    Intercepta las acciones justo antes de step() y las sustituye por una válida,
    reutilizando la misma lógica que empleas en evaluación.
    """

    # Constantes del entorno Yahtzee (ajústalas si cambian)
    JUGADAS_POSIBLES = 31   # 0..30 = relanzar
    OFFSET_CATEGORIAS = 31  # 31..43 = puntuar
    NUM_CATEGORIAS = 13

    def __init__(self, venv):
        super().__init__(venv)

    def step_async(self, actions):
        # actions: np.ndarray shape (n_envs,)
        repaired = []
        for i, a in enumerate(actions):
            base_env = self.venv.envs[i].unwrapped
            info = base_env._get_info()
            mask = np.asarray(info["action_mask"], dtype=np.int8)
            tiradas_restantes = base_env.tiradas_restantes
            repaired.append(self._repair(int(a), mask, tiradas_restantes, base_env))
        return super().step_async(np.array(repaired))

    # --------- lógica de reparación (idéntica a evaluación) ----------
    def _best_scoring_action(self, base_env):
        libres = [c for c in range(self.NUM_CATEGORIAS) if base_env.puntuaciones[c] == -1]
        if not libres:
            return self.OFFSET_CATEGORIAS + (self.NUM_CATEGORIAS - 1)  # CHANCE
        scores = [(c, base_env._calcular_puntuacion(c)) for c in libres]
        best_cat = max(scores, key=lambda t: t[1])[0]
        return self.OFFSET_CATEGORIAS + best_cat

    def _repair(self, action, mask, tiradas_restantes, base_env):
        if mask[action] == 1:
            return action
        # Acción inválida → decidir reparación
        if tiradas_restantes == 0:
            # Fuerza a puntuar en la mejor categoría
            return self._best_scoring_action(base_env)
        else:
            # Estamos en fase de relanzar: escoge una jugada válida
            valid_roll = np.where(mask[:self.JUGADAS_POSIBLES] == 1)[0]
            if len(valid_roll) > 0:
                # Usa 0 (relanzar todos) si está disponible; si no, uno ale                # Usa 0 (relanzar todos) si está disponible; si no, uno aleatorio válido
                return 0 if 0 in valid_roll else int(np.random.choice(valid_roll))
            # Si no hubiera relanzos válidos (raro), puntúa la mejor categoría
