import gymnasium as gym
from gymnasium.utils import RecordConstructorArgs
import logging

class AllowedActions(gym.ActionWrapper, RecordConstructorArgs):
    def __init__(self, env):
        super().__init__(env)
        # Configurar el logger
        self.logger = logging.getLogger(__name__)
        # Registrar los argumentos del constructor (necesario para check_env)
        self._saved_kwargs = {"env": env}

    def action(self, action):
        # Esto garantiza acceso al YahtzeeEnv real
        env = self.env.unwrapped
        action = env._desnormalizar_accion(action)
        valores_acciones_permitidas = env._acciones_permitidas()
        self.logger.debug(f'Accion: {action}, {type(action).__name__}, Acciones permitidas {valores_acciones_permitidas}')
        if int(action) in valores_acciones_permitidas:
            # La acción está permitida
            return action
        else:
            # La acción NO está permitida
            raise ValueError(f"Acción inválida: {action}. Acciones permitidas: {valores_acciones_permitidas}")

    def getTipoAccion(self, action):
        return self.env.getTipoAccion(action)
    def getMantener(self, action):
        return self.env.getMantener(action)
    def getCategoria(self, action):
        return self.env.getCategoria(action)
