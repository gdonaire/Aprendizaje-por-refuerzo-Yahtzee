import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum, IntEnum
from typing import Dict, Optional, Tuple
import logging

class Categoria(IntEnum):
    UNOS = 0
    DOSES = 1
    TRESES = 2
    CUATROS = 3
    CINCOS = 4
    SEISES = 5
    TRIO = 6
    POKER = 7
    FULL = 8
    ESCALERA_PEQUENA = 9
    ESCALERA_GRANDE = 10
    YAHTZEE = 11
    CHANCE = 12
    BONO_SUPERIOR = 13
    BONO_YAHTZEE = 14

CONSTANT_NUM_CATEGORIAS = 13  # Número de categorías jugables (sin contar bonos)
CONSTANT_JUGADAS_POSIBLES = 31  # Número de combinaciones posibles de mantener dados (2^5 - 1)
CONSTANT_NUM_ACCIONES = CONSTANT_JUGADAS_POSIBLES + CONSTANT_NUM_CATEGORIAS
CONSTANT_OFFSET_CATEGORIAS = 31  # Offset para las categorías jugables
    
class TipoAccion(IntEnum):
    LANZAR = 0
    PUNTUAR = 1

CONSTANT_SCORES: Dict[Categoria, int] = {
    Categoria.FULL: 25,
    Categoria.ESCALERA_PEQUENA: 30,
    Categoria.ESCALERA_GRANDE: 40,
    Categoria.YAHTZEE: 50,
    Categoria.BONO_YAHTZEE: 100,
    Categoria.BONO_SUPERIOR: 35,
}

nombres_categoria = {
    Categoria.UNOS : 'Unos',
    Categoria.DOSES : 'Doses',
    Categoria.TRESES : 'Treses',
    Categoria.CUATROS : 'Cuatros',
    Categoria.CINCOS : 'Cincos',
    Categoria.SEISES : 'Seises',
    Categoria.TRIO: 'Trío',
    Categoria.POKER : 'Póker',
    Categoria.FULL : 'Full',
    Categoria.ESCALERA_PEQUENA : 'Escalera pequeña',
    Categoria.ESCALERA_GRANDE : 'Escalera grande',
    Categoria.YAHTZEE : 'Yahtzee',
    Categoria.CHANCE : 'Chance',
    Categoria.BONO_SUPERIOR : 'Bono superior',
    Categoria.BONO_YAHTZEE : 'Bono Yahtzee'
}

def CategoriaNombre(categoria):
    return nombres_categoria[int(categoria)]

class YahtzeeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        """
        Inicializa el entorno de Yahtzee.
            render_mode (str, opcional): Modo de renderizado del entorno. Puede ser None o "human".
        Configura el estado inicial del juego, el espacio de observación y el espacio de acción para el entorno.
        - max_rondas: Número máximo de rondas (13).
        - ronda_actual: Ronda actual.
        - dado: Vector de 5 enteros que representa los valores actuales de los dados.
        - tiradas_restantes: Número de tiradas restantes en la ronda actual (3 por defecto).
        - puntuaciones: Tarjeta de puntuación con 13 categorías.
        - categorias_usadas: Vector binario que indica si cada categoría ya ha sido utilizada.
        Espacio de observación:
            - "dado": Valores actuales de los dados (5 enteros entre 1 y 6).
            - "tiradas_restantes": Tiradas restantes (0 a 3).
            - "puntuaciones": Puntuaciones por categoría (13 enteros con límites superiores realistas).
            - "categorias_usadas": Categorías ya utilizadas (vector binario de 13 elementos).
            - "ronda_actual": Número de la ronda actual (0 a 13).
        Espacio de acción:
            - "type": Tipo de acción (0 = tirar dados, 1 = puntuar en una categoría).
            - "keep": Dados que se mantienen al tirar (vector binario de longitud 5).
            - "category": Categoría de puntuación seleccionada (valor entre 0 y 12).
        Las categorías de bonificación se calculan automáticamente y no pueden ser seleccionadas directamente.   
        """
        super(YahtzeeEnv, self).__init__()

        self.max_rondas = 13
        self.ronda_actual = 0        
        self.tiradas_restantes = 3
        self.dado = np.zeros(5, dtype=np.int32)
        self.puntuaciones = np.zeros(len(Categoria), dtype=np.int32)
        self.categorias_usadas = np.zeros(len(Categoria), dtype=np.int32)
        # Definir el espacio de estados
        # 31 combinaciones de mantener (0-30), 31 equivale a 11111 mantener todo
        # y adicionalmante 13 acciones mas que se corresponde con puntuar en alguna 
        # de las 13 categorías
        # Discrete action space permite utilizar masking de acciones
        #self.action_space = spaces.Discrete(CONSTANT_NUM_ACCIONES)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Configurar el logger
        self.logger = logging.getLogger(__name__)

        # Establece límites superiores realistas para cada categoría de la tarjeta de puntuación
        puntuaciones_max = np.array([
            5,  # Unos (máximo 6*5)
            10,  # Doses
            15,  # Treses
            20,  # Cuatros
            25,  # Cincos
            30,  # Seises
            30,  # Trío (máxima suma 30)
            30,  # Póker (máxima suma 30)
            25,  # Full (fijo)
            30,  # Escalera pequeña (fijo)
            40,  # Escalera grande (fijo)
            50,  # Yahtzee (fijo)
            30,   # Chance (máxima suma 30)
            35,  # Bono superior (fijo)
            100  # Bono Yahtzee (fijo)
        ], dtype=int)

        # Definición del espacio de observación:
        # - "ronda_actual": número de la ronda actual (0 a 13).
        # - "tiradas_restantes": número de tiradas restantes en la ronda actual (0 a 3).
        # - "dado": vector de 5 enteros entre 1 y 6, representando los valores actuales de los dados.
        # - "puntuaciones": vector de 13 enteros, cada uno con un límite superior realista según la categoría.
        # - "categorias_usadas": vector binario de 13 elementos, indica si cada categoría ya ha sido utilizada.
        
        # Tamaño total: 1 (ronda) + 1 (tiradas) + 5 (dados) + 15 (puntuaciones y bonos) + 15 (categorías usadas)
        obs_size = 1 + 1 + 5 + len(Categoria) + len(Categoria)
        low = np.array([0, 0] + [1]*5 + [0]*len(Categoria) + [0]*len(Categoria), dtype=np.float32)
        high = np.array([self.max_rondas, 3] + [6]*5 + puntuaciones_max.tolist() + [1]*len(Categoria), dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_size,), dtype=np.float32)

        #self.observation_space = spaces.Dict({
        #    "ronda_actual": spaces.Discrete(self.max_rondas + 1),
        #    "tiradas_restantes": spaces.Discrete(4),
        #    "dado": spaces.Box(low=1, high=6, shape=(5,), dtype=np.int32),
        #    "puntuaciones": spaces.Box(low=0, high=puntuaciones_max, shape=(len(Categoria),), dtype=np.int32),
        #    "categorias_usadas": spaces.Box(low=0, high=1, shape=(len(Categoria),), dtype=np.int32)
        #})

        # Espacio de acción aplanado:
        # [tipo_accion, mantener[0], mantener[1], mantener[2], mantener[3], mantener[4], categoria]
        # tipo_accion: 0 = tirar los dados, 1 = puntuar en una categoría
        # mantener: vector binario de longitud 5 (0 -- volver a tirar dado, o 1 = mantener)
        # categoria: valor entre 0 y 12
        # Espacio de acción normalizado: valores en [-1, 1]
        #self.action_space = spaces.Box(
        #    low=-1.0,
        #    high=1.0,
        #    shape=(7,),
        #    dtype=np.float32
        #)
  
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _desnormalizar_accion(self, action):
        #self.logger.debug(f'Accion normalizada {action}, {type(action).__name__}')
        arr = np.clip(action, -1.0, 1.0)               # Limita valores a [-1, 1]
        mapped = np.round((arr + 1) * CONSTANT_NUM_ACCIONES / 2).astype(int).item()
        #self.logger.debug(f'Accion desnormalizada {mapped}, {type(mapped).__name__}')
        return mapped

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno de Yahtzee para comenzar una nueva partida.

        Args:
            seed (int, opcional): Semilla para el generador de números aleatorios.
            options (dict, opcional): Opciones adicionales para el reinicio (no utilizadas).

        Returns:
            observation (dict): Estado inicial del entorno tras el reinicio.
            info (dict): Información adicional (vacía en este caso).
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.ronda_actual = 0
        self.tiradas_restantes = 3
        self.dado = self.np_random.integers(1, 7, size=5, dtype=np.int32)
        self.puntuaciones = np.zeros(len(Categoria), dtype=np.int32)
        self.categorias_usadas = np.zeros(len(Categoria), dtype=np.int32)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _acciones_permitidas(self):
        """
        Devuelve una lista de acciones permitidas en el estado actual del entorno.

        Returns:
            list: Lista de acciones posibles según el estado actual.
            Cada acción se representa como un entero en el rango [-1, 44], 
            donde -1 indica que la acción no es posible.
        """
        acciones_posibles = []

        if self.ronda_actual <  self.max_rondas:
            # Si la ronda actual es menor que el máximo de rondas, se permiten acciones adicionales
            # (re)tirar dados
            if self.tiradas_restantes > 1:
                acciones_posibles.extend([i for i in range(CONSTANT_JUGADAS_POSIBLES)])  # Todas las combinaciones de mantener dados excepto mantener todos (31 combinaciones)
            else:
                # El caso de cero se debe manejar aparte
                acciones_posibles.extend([-1 for _ in range(CONSTANT_JUGADAS_POSIBLES)])  # No se permiten acciones de lanzar dados

            # Puntuar en categorías no usadas
            counts = np.bincount(self.dado, minlength=7)[1:]
            for categoria in range(CONSTANT_NUM_CATEGORIAS):
                # Si la categoría es YAHTZEE, se permite puntuar incluso si ya se ha usado
                if categoria == Categoria.YAHTZEE and self.categorias_usadas[Categoria.YAHTZEE] and self.puntuaciones[Categoria.YAHTZEE] > 0 and np.max(counts) == 5:
                    acciones_posibles.append(CONSTANT_OFFSET_CATEGORIAS + categoria)  # Permitir Yahtzee Bonus
                elif not self.categorias_usadas[categoria]:
                    acciones_posibles.append(CONSTANT_OFFSET_CATEGORIAS + categoria)
                else:
                    acciones_posibles.append(-1)  # Categoría ya usada, no se permite puntuar
        else:
            # Si se han completado todas las rondas, no se permiten más acciones
            acciones_posibles = [-1 for _ in range(CONSTANT_JUGADAS_POSIBLES + CONSTANT_OFFSET_CATEGORIAS)]

        self.logger.debug(f'Acciones posibles: {acciones_posibles}')
        return acciones_posibles

    def _terminated(self):
        """
        Comprueba si el juego ha terminado.

        Returns:
            bool: True si el juego ha terminado (se han completado todas las rondas), False en caso contrario.
        """
        return self.ronda_actual >= self.max_rondas
        # Comprobar que la acción es válida
        if len(action) != 7:
            return False

        # Desnormalizar acción
        action_desnormalizada = desnormalizar_accion(action)
        #print(f'WRAPPER Acción desnormalizada: {action_desnormalizada}')

        tipo_accion = action_desnormalizada[0]
        mantener = action_desnormalizada[1:6]
        categoria = action_desnormalizada[6]
        
        if tipo_accion not in [TipoAccion.LANZAR, TipoAccion.PUNTUAR]:
            return False
        # Comprobar si se puede LANZAR
        if tipo_accion == TipoAccion.LANZAR and self.tiradas_restantes <= 1:
            return False
        # Comprobar si se mantienen todos los dados
        if tipo_accion == TipoAccion.LANZAR and np.sum(mantener) == 5:
            return False
        # Comprobar si se puede puntuar
        if tipo_accion == TipoAccion.PUNTUAR and not self.categorias_usadas[categoria]:
            return False

        return True

    def getTipoAccion(self, action):
        if not isinstance(action, (int, np.int32)):      
            action = self._desnormalizar_accion(action)
        return TipoAccion.LANZAR if action < CONSTANT_JUGADAS_POSIBLES else TipoAccion.PUNTUAR

    def getMantener(self, action):
        if not isinstance(action, (int, np.int32)):      
            action = self._desnormalizar_accion(action)
        if action < CONSTANT_JUGADAS_POSIBLES:
            mantener = [int((action >> i) & 1) for i in range(5)]
            return mantener
        else:
            return [1]*5  # Si la acción es puntuar, se consideran todos los dados como mantenidos
    
    def getCategoria(self, action):
        if not isinstance(action, (int, np.int32)):      
            action = self._desnormalizar_accion(action)
        if action >= CONSTANT_OFFSET_CATEGORIAS:
            categoria = action - CONSTANT_OFFSET_CATEGORIAS
            return categoria
        else:
            return None  # Si la acción es lanzar, no hay categoría asociada    

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()  # convierte array(…) → float
        if isinstance(action, float):
            action = self._desnormalizar_accion(action)

        if (action != None) and not isinstance(action, (int, np.int32)):
            self.logger.error(f'Formato action no soportado, Action {action}, {type(action).__name__}')

        done = False
        recompensa = 0

        if action == None:
            recompensa = -1  # Penalización por acción inválida
        elif action < CONSTANT_OFFSET_CATEGORIAS:
            # LANZAR DADOS
            mantener = [(action >> i) & 1 for i in range(5)]
            for i in range(5):
                if mantener[i] == 0:
                    # Volver a tirar dado i (aleatorio)
                    self.dado[i] = self.np_random.integers(1, 7, dtype=np.int32)
            self.tiradas_restantes -= 1
        else:
            # PUNTUAR
            categoria = action - CONSTANT_OFFSET_CATEGORIAS
            counts = np.bincount(self.dado, minlength=7)[1:]
            # Comprobar Yahtzee adicional
            if categoria == Categoria.YAHTZEE and np.max(counts) == 5:
                # Yahtzee adicional
                if self.categorias_usadas[Categoria.YAHTZEE] and self.puntuaciones[Categoria.YAHTZEE] > 0:
                    self.puntuaciones[Categoria.BONO_YAHTZEE] += CONSTANT_SCORES[Categoria.BONO_YAHTZEE]
                    recompensa = CONSTANT_SCORES[Categoria.BONO_YAHTZEE]

            # Comprobar puntuación
            puntuacion = self._calculate_score(categoria)
            self.puntuaciones[categoria] = puntuacion
            self.categorias_usadas[categoria] = 1
            recompensa += puntuacion

            # Comprobar bono superior
            if not self.categorias_usadas[Categoria.BONO_SUPERIOR]:
                upper_score = np.sum(self.puntuaciones[:6])
                if upper_score >= 63:
                    self.puntuaciones[Categoria.BONO_SUPERIOR] = CONSTANT_SCORES[Categoria.BONO_SUPERIOR]
                    self.categorias_usadas[Categoria.BONO_SUPERIOR] = 1
                    recompensa += CONSTANT_SCORES[Categoria.BONO_SUPERIOR]
            
            # Despues de puntuar siempre incrementamos ronda
            # Reiniciamos los dados y las tiradas
            self.ronda_actual += 1
            self.tiradas_restantes = 3
            self.dado = self.np_random.integers(1, 7, size=5, dtype=np.int32)

        done = self._terminated()
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        truncated = False
        return observation, float(recompensa), done, truncated, info

    def _get_info(self):
        """
        Devuelve un diccionario vacío con información adicional sobre el estado actual del entorno.

        Returns:
            dict: Un diccionario vacío. Puede ampliarse para incluir información específica del entorno.
        """
        return {}  

    def _get_obs(self):
        """Devuelve la observacion como un vector aplanado (Box)."""
        return np.concatenate([
            [self.ronda_actual],
            [self.tiradas_restantes],
            self.dado.astype(np.float32),
            self.puntuaciones.astype(np.float32),
            self.categorias_usadas.astype(np.float32)
        ]).astype(np.float32)

    def _get_obs_v0(self):
        """
        Devuelve una observación actual del estado del juego de Yahtzee.

        Retorna un diccionario que contiene:
            - "dado": una copia de la lista de dados actuales.
            - "ronda_actual": el número de la ronda actual.
            - "tiradas_restantes": la cantidad de lanzamientos restantes en la ronda actual.
            - "puntuaciones": una copia de la tarjeta de puntuación actual.
            - "categorias_usadas": una copia de las categorías ya utilizadas.

        Returns:
            dict: Estado actual del entorno de Yahtzee.
        """
        obs = {
            "ronda_actual": self.ronda_actual,
            "tiradas_restantes": self.tiradas_restantes,
            "dado": self.dado.copy(),
            "puntuaciones": self.puntuaciones.copy(),
            "categorias_usadas": self.categorias_usadas.copy()
        }
        
        for key, val in obs.items():
            if np.any(val < 0):
                self.logger.error(f"Valor negativo en {key}: {val}")

        self.logger.debug(f'Observación: {obs}')

        return obs

    def _calculate_score(self, categoria):
        """
        Calcula la puntuación para una categoría dada según los dados actuales.

        Args:
            categoria (Categoria): La categoría de puntuación a evaluar.

        Returns:
            int: La puntuación obtenida para la categoría especificada según las reglas de Yahtzee.
        """
        # Parte Inferior
        counts = np.bincount(self.dado, minlength=7)[1:]
        score = 0
        match categoria:
            case Categoria.UNOS | Categoria.DOSES | Categoria.TRESES | Categoria.CUATROS | Categoria.CINCOS | Categoria.SEISES:
                score = counts[categoria] * (categoria + 1)
            case Categoria.TRIO:
                # Trío: Al menos tres dados iguales. Se suma el total de todos los dados.
                score = sum(self.dado) if np.max(counts) >= 3 else 0
            case Categoria.POKER:
                # Póker: Al menos cuatro dados iguales. Se suma el total de todos los dados.
                score = sum(self.dado) if np.max(counts) >= 4 else 0
            case Categoria.FULL:
                # Full: Tres dados de un número y dos de otro. Vale 25 puntos.
                score = CONSTANT_SCORES[Categoria.FULL] if 3 in counts and 2 in counts else 0
            case Categoria.ESCALERA_PEQUENA:
                # Escalera pequeña: Secuencia de 4 números consecutivos. Vale 30 puntos.
                score = CONSTANT_SCORES[Categoria.ESCALERA_PEQUENA] if self._has_straight(4) else 0
            case Categoria.ESCALERA_GRANDE:
                # Escalera grande: Secuencia de 5 números consecutivos. Vale 40 puntos.
                score = CONSTANT_SCORES[Categoria.ESCALERA_GRANDE] if self._has_straight(5) else 0
            case Categoria.YAHTZEE:
                # Yahtzee: Cinco dados iguales. Vale 50 puntos.
                score = CONSTANT_SCORES[Categoria.YAHTZEE] if np.max(counts) == 5 else 0
            case Categoria.CHANCE:
                # Chance: Se puede anotar cualquier combinación. Se suma el total de los dados.
                score = sum(self.dado)
            case _:
                raise ValueError(f"Categoría inválida pasada a _calculate_score: {categoria}")
        self.logger.debug(f'Puntuación calculada para {CategoriaNombre(categoria)}: {score}')
        return score

    def _has_straight(self, length):
        """
        Comprueba si los dados contienen una secuencia (escalera) de longitud especificada.

        Args:
            length (int): Longitud de la escalera a buscar.

        Returns:
            bool: True si existe una escalera de la longitud dada, False en caso contrario.
        """
        unique = sorted(set(self.dado))
        for i in range(len(unique) - length + 1):
            if all(unique[i + j] == unique[i] + j for j in range(length)):
                return True
        return False

    def _calculate_final_score(self):
        """
        Calcula la puntuación final de la partida de Yahtzee.

        - Suma la sección superior de la tarjeta de puntuación (Unos, Doses, Treses, Cuatros, Cincos, Seises).
        - Aplica el bono superior si la suma de la sección superior es mayor o igual a 63 puntos.
        - Incluye el bono de Yahtzee según el número de Yahtzees extra obtenidos.
        - Devuelve la suma total de la tarjeta de puntuación, el bono superior y el bono de Yahtzee.

        Returns:
            int: Puntuación final de la partida.
        """
        return np.sum(self.puntuaciones)

    def _render_frame(self):
        if self.render_mode == "human":
            print(f'Puntuaciones: {self.puntuaciones}')
            #self.render()

    def render(self):
        dado_str = " ".join([f"🎲{d}" for d in self.dado])
        print("\n" + "="*40)
        print(f"🕒 Ronda: {self.ronda_actual + 1} / {self.max_rondas}")
        print(f"🎯 Dados: {dado_str}")
        print(f"🔁 Tiradas restantes: {self.tiradas_restantes}")
        print("📋 Tarjeta de puntuación:")
        # Fallback for environments without Unicode support
        print("="*40 + "")
        # Display puntuaciones categories and scores
        for i in range(13):
            nombre = nombres_categoria[Categoria(i)]
            usado = "✅" if self.categorias_usadas[i] else "❌"
            puntuacion = self.puntuaciones[i]
            print(f"  {nombre:<18} | {puntuacion:>3} pts | {usado}")
        print(f"🎁 Bono superior: {self.puntuaciones[Categoria.BONO_SUPERIOR]} pts")
        print(f"🔥 Bono Yahtzee: {self.puntuaciones[Categoria.YAHTZEE]} pts")
        print(f"🏁 Puntuación total: {self._calculate_final_score()} pts")
        print("="*40 + "")

class YahtzeeEnvV1(gym.Env):
    """
    Gymnasium environment that simulates a single-player Yahtzee game.


    Overview:
    - 5 six-sided dice
    - 13 rounds (one score category filled per round)
    - On each round the agent can roll up to 3 times (initial roll + up to 2 rerolls).
    - Between rolls the agent selects which dice to keep via a 5-bit mask action (0..30).
    - When rolls_left == 0 (or the agent chooses to score by selecting a category action), the agent must choose a scoring category (13 categories).


    Action space:
    - Discrete(44):
    0..30 -> keep mask for next roll (bit i==1 => keep die i). If rolls_left==0 these actions are ignored.
    31..43 -> select scoring category (31 = ones, 32 = twos, ..., 43 = chance)


    Observation space:
    - Box(low=0, high=..., shape=(1 + 1 + 5 + 13,), dtype=np.float32)
    Order: [round, rolls_left, dice(5), scorecard(13)]
    - dice: values 1..6
    - rolls_left: 0..2
    - scorecard: -1 means empty, otherwise score
    - round: 0..12


    Reward:
    - 0 for keep actions / rerolls
    - When choosing a category, reward equals the points awarded for that category.


    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        
        super(YahtzeeEnvV1, self).__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # configurar parámetros del juego
        self.max_rondas = 13
        
        # Definir el espacio de estados
        # 31 combinaciones de mantener (0-30), 31 equivale a 11111 mantener todo
        # y adicionalmante 13 acciones mas que se corresponde con puntuar en alguna 
        # de las 13 categorías
        # Discrete action space permite utilizar masking de acciones
        self.action_space = spaces.Discrete(CONSTANT_NUM_ACCIONES)

        # Definición del espacio de observación:
        # - "ronda_actual": número de la ronda actual (0 a 13).
        # - "tiradas_restantes": número de tiradas restantes en la ronda actual (0 a 3).
        # - "dado": vector de 5 enteros entre 1 y 6, representando los valores actuales de los dados.
        # - "puntuaciones": vector de 13 enteros, cada uno con un límite superior realista según la categoría.
        # Tamaño total: 1 (ronda) + 1 (tiradas) + 5 (dados) + 15 (puntuaciones y bonos)        
        obs_size = 1 + 1 + 5 + len(Categoria)
        low = np.array([0, 0] + [1]*5 + [-1]*len(Categoria), dtype=np.float32)
        # Establece límites superiores realistas para cada categoría de la tarjeta de puntuación
        puntuaciones_max = np.array([
            5,  # Unos (máximo 6*5)
            10,  # Doses
            15,  # Treses
            20,  # Cuatros
            25,  # Cincos
            30,  # Seises
            30,  # Trío (máxima suma 30)
            30,  # Póker (máxima suma 30)
            25,  # Full (fijo)
            30,  # Escalera pequeña (fijo)
            40,  # Escalera grande (fijo)
            50,  # Yahtzee (fijo)
            30,   # Chance (máxima suma 30)
            35,  # Bono superior (fijo)
            100  # Bono Yahtzee (fijo)
        ], dtype=int)        
        high = np.array([self.max_rondas, 3] + [6]*5 + puntuaciones_max.tolist(), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_size,), dtype=np.float32)

        # Configurar el logger
        self.logger = logging.getLogger(__name__)
  
        # Inicializar estado del juego
        self.ronda_actual = 0        
        self.tiradas_restantes = 2
        self.dado = np.zeros(5, dtype=np.int32)
        self.puntuaciones = np.zeros(len(Categoria), dtype=np.int32)
        self.categorias_usadas = np.zeros(len(Categoria), dtype=np.int32)
        self.recompensa = 0
        self.done = False
                                   
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reinicia el entorno de Yahtzee para comenzar una nueva partida.

        Args:
            seed (int, opcional): Semilla para el generador de números aleatorios.
            options (dict, opcional): Opciones adicionales para el reinicio (no utilizadas).

        Returns:
            observation (dict): Estado inicial del entorno tras el reinicio.
            info (dict): Información adicional (vacía en este caso).
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.ronda_actual = 0
        self.tiradas_restantes = 2
        self.dado = self.np_random.integers(1, 7, size=5, dtype=np.int32)
        self.puntuaciones = np.zeros(len(Categoria), dtype=np.int32)
        self.categorias_usadas = np.zeros(len(Categoria), dtype=np.int32)
        self.recompensa = 0
        self.done = False
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self):
        """Devuelve la observacion como un vector aplanado (Box)."""
        obs_vec = np.concatenate([
            np.array([self.ronda_actual], dtype=np.float32),
            np.array([self.tiradas_restantes], dtype=np.float32),
            self.dado.astype(np.float32),
            self.puntuaciones.astype(np.float32)
        ])
        return obs_vec

    def _get_info(self):
        """Devuelve información adicional sobre el estado del juego."""
        return {
            "categorias_usadas": self.categorias_usadas.copy(),
            "recompensa": self.recompensa
        }

    def getTipoAccion(self, action):
        return TipoAccion.LANZAR if action < CONSTANT_JUGADAS_POSIBLES else TipoAccion.PUNTUAR

    def getMantener(self, action):
        if action < CONSTANT_JUGADAS_POSIBLES:
            mantener = [int((action >> i) & 1) for i in range(5)]
            return mantener
        else:
            return [1]*5  # Si la acción es puntuar, se consideran todos los dados como mantenidos
    
    def getCategoria(self, action):
        if action >= CONSTANT_OFFSET_CATEGORIAS:
            categoria = action - CONSTANT_OFFSET_CATEGORIAS
            return categoria
        else:
            return None  # Si la acción es lanzar, no hay categoría asociada 

    def _mascara_categorias_permitidas(self):
        return (self.puntuaciones == -1).astype(np.int8)

    def _terminated(self):
        """
        Comprueba si el juego ha terminado.

        Returns:
            bool: True si el juego ha terminado (se han completado todas las rondas), False en caso contrario.
        """
        return self.ronda_actual >= self.max_rondas

    def _contar_frecuencia(self, dados):
        """
        Calcula la frecuencia de cada cara en los dados actuales.

        Args:
            dados (np.ndarray): Array de enteros que representan las caras de los dados.

        Returns:
            np.ndarray: Array de longitud 6 donde el índice i representa la cantidad de veces que la cara (i+1) aparece en los dados.
        """
        conteo_frecuencia = np.bincount(dados, minlength=6)
        return conteo_frecuencia

    def _calcular_puntuacion(self, categoria):
        """
        Calcula la puntuación para una categoría dada según los dados actuales.

        Args:
            categoria (Categoria): La categoría de puntuación a evaluar.

        Returns:
            int: La puntuación obtenida para la categoría especificada según las reglas de Yahtzee.
        """
        # Parte Inferior
        conteo_frecuencia = self._contar_frecuencia(self.dado)
        puntuacion = 0
        match categoria:
            case Categoria.UNOS | Categoria.DOSES | Categoria.TRESES | Categoria.CUATROS | Categoria.CINCOS | Categoria.SEISES:
                puntuacion = conteo_frecuencia[categoria] * (categoria + 1)
            case Categoria.TRIO:
                # Trío: Al menos tres dados iguales. Se suma el total de todos los dados.
                puntuacion = sum(self.dado) if np.max(conteo_frecuencia) >= 3 else 0
            case Categoria.POKER:
                # Póker: Al menos cuatro dados iguales. Se suma el total de todos los dados.
                puntuacion = sum(self.dado) if np.max(conteo_frecuencia) >= 4 else 0
            case Categoria.FULL:
                # Full: Tres dados de un número y dos de otro. Vale 25 puntos.
                puntuacion = CONSTANT_SCORES[Categoria.FULL] if 3 in conteo_frecuencia and 2 in conteo_frecuencia else 0
            case Categoria.ESCALERA_PEQUENA:
                # Escalera pequeña: Secuencia de 4 números consecutivos. Vale 30 puntos.
                puntuacion = CONSTANT_SCORES[Categoria.ESCALERA_PEQUENA] if self._escalera(4) else 0
            case Categoria.ESCALERA_GRANDE:
                # Escalera grande: Secuencia de 5 números consecutivos. Vale 40 puntos.
                puntuacion = CONSTANT_SCORES[Categoria.ESCALERA_GRANDE] if self._escalera(5) else 0
            case Categoria.YAHTZEE:
                # Yahtzee: Cinco dados iguales. Vale 50 puntos.
                puntuacion = CONSTANT_SCORES[Categoria.YAHTZEE] if np.max(conteo_frecuencia) == 5 else 0
            case Categoria.CHANCE:
                # Chance: Se puede anotar cualquier combinación. Se suma el total de los dados.
                puntuacion = sum(self.dado)
            case _:
                raise ValueError(f"Categoría inválida pasada a _calculate_score: {categoria}")
        self.logger.debug(f'Puntuación calculada para {CategoriaNombre(categoria)}: {puntuacion}')
        return puntuacion

    def _escalera(self, length):
        """
        Comprueba si los dados contienen una secuencia (escalera) de longitud especificada.

        Args:
            length (int): Longitud de la escalera a buscar.

        Returns:
            bool: True si existe una escalera de la longitud dada, False en caso contrario.
        """
        unique = sorted(set(self.dado))
        for i in range(len(unique) - length + 1):
            if all(unique[i + j] == unique[i] + j for j in range(length)):
                return True
        return False
    
    def _calculate_puntuacion_final(self):
        """
        Calcula la puntuación final de la partida de Yahtzee.

        - Suma la sección superior de la tarjeta de puntuación (Unos, Doses, Treses, Cuatros, Cincos, Seises).
        - Aplica el bono superior si la suma de la sección superior es mayor o igual a 63 puntos.
        - Incluye el bono de Yahtzee según el número de Yahtzees extra obtenidos.
        - Devuelve la suma total de la tarjeta de puntuación, el bono superior y el bono de Yahtzee.

        Returns:
            int: Puntuación final de la partida.
        """
        return np.sum(self.puntuaciones)

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, dict]:
        if self.done:
            raise RuntimeError("Step called after environment is done. Call reset().")

        self.recompensa = 0
                
        if action < CONSTANT_OFFSET_CATEGORIAS:
            # LANZAR DADOS
            if self.tiradas_restantes > 0:
                mantener = np.array([(action >> i) & 1 for i in range(5)], dtype=bool)
                for i in range(5):
                    if not mantener[i]:
                        # Volver a tirar dado i (aleatorio)
                        self.dado[i] = self.np_random.integers(1, 7, dtype=np.int32)
                self.tiradas_restantes -= 1
        else:
            # PUNTUAR
            categoria = action - CONSTANT_OFFSET_CATEGORIAS
            if categoria < 0 or categoria >= len(Categoria):
                raise ValueError(f"Categoría inválida seleccionada: {categoria} en step()")

            avanzar_ronda = False
            counts = self._contar_frecuencia(self.dado)
            
            # Comprobar Yahtzee adicional
            if categoria == Categoria.YAHTZEE and np.max(counts) == 5 and self.categorias_usadas[Categoria.YAHTZEE] and self.puntuaciones[Categoria.YAHTZEE] > 0:
                # calcular puntuación de yahtzee adicional
                puntuacion = self._calcular_puntuacion(categoria)
                self.puntuaciones[categoria] = puntuacion
                self.categorias_usadas[categoria] = 1
                self.recompensa += puntuacion
                # agregar bono yahtzee
                self.puntuaciones[Categoria.BONO_YAHTZEE] += CONSTANT_SCORES[Categoria.BONO_YAHTZEE]
                self.recompensa += CONSTANT_SCORES[Categoria.BONO_YAHTZEE]
                avanzar_ronda = True
            # Resto de categorías
            elif not self.categorias_usadas[categoria]:
                # puntuar categoría
                puntuacion = self._calcular_puntuacion(categoria)
                self.puntuaciones[categoria] = puntuacion
                self.categorias_usadas[categoria] = 1
                self.recompensa += puntuacion                
                # Comprobar bono superior
                if not self.categorias_usadas[Categoria.BONO_SUPERIOR] and categoria in range(6):
                    upper_score = np.sum(self.puntuaciones[:6])
                    if upper_score >= 63:
                        self.puntuaciones[Categoria.BONO_SUPERIOR] = CONSTANT_SCORES[Categoria.BONO_SUPERIOR]
                        self.categorias_usadas[Categoria.BONO_SUPERIOR] = 1
                        self.recompensa += CONSTANT_SCORES[Categoria.BONO_SUPERIOR]
                avanzar_ronda = True
            else:
                #recompensa = -5  # Penalización por categoría ya usada
                self.recompensa = 0  # Penalización por categoría ya usada
                print(f"Categoría ya usada seleccionada: {CategoriaNombre(categoria)} en step()")

            if avanzar_ronda:
                # Despues de puntuar siempre incrementamos ronda
                # Reiniciamos los dados y las tiradas
                self.ronda_actual += 1
                self.tiradas_restantes = 2
                self.dado = self.np_random.integers(1, 7, size=5, dtype=np.int32)

        # actualizar done, obs, info
        info = self._get_info()
        self.done = self._terminated()
        obs = self._get_obs()

        return obs, float(self.recompensa), self.done, False, info
        
    def render(self):
        dado_str = " ".join([f"🎲{d}" for d in self.dado])
        print("\n" + "="*40)
        print(f"🕒 Ronda: {self.ronda_actual + 1} / {self.max_rondas}")
        print(f"🎯 Dados: {dado_str}")
        print(f"🔁 Tiradas restantes: {self.tiradas_restantes}")
        print("📋 Tarjeta de puntuación:")
        # Fallback for environments without Unicode support
        print("="*40 + "")
        # Display puntuaciones categories and scores
        for i in range(13):
            nombre = nombres_categoria[Categoria(i)]
            usado = "✅" if self.categorias_usadas[i] else "❌"
            puntuacion = self.puntuaciones[i]
            print(f"  {nombre:<18} | {puntuacion:>3} pts | {usado}")
        print(f"🎁 Bono superior: {self.puntuaciones[Categoria.BONO_SUPERIOR]} pts")
        print(f"🔥 Bono Yahtzee: {self.puntuaciones[Categoria.YAHTZEE]} pts")
        print(f"🏁 Puntuación total: {self._calculate_puntuacion_final()} pts")
        print("="*40 + "")

    def close(self):
        pass

