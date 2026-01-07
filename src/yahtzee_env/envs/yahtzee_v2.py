import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum
from typing import Dict, Optional, Tuple
import logging

# ==============================================================
# ENUMERACIONES Y CONSTANTES
# ==============================================================
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

CONSTANT_NUM_CATEGORIAS = 13               # categorías jugables (sin contar bonos)
CONSTANT_JUGADAS_POSIBLES = 31             # 2^5 - 1
CONSTANT_NUM_ACCIONES = CONSTANT_JUGADAS_POSIBLES + CONSTANT_NUM_CATEGORIAS
CONSTANT_OFFSET_CATEGORIAS = 31            # offset para acciones de puntuar

# Penalizaciones (ajustables)
PENALIZACION_RELANZAR_SIN_TIRADAS = 0.5    # penalización suave al relanzar sin tiradas
PENALIZACION_CATEGORIA_USADA = 0.5         # penalización suave al puntuar en casilla ocupada
PENALIZACION_MASCARA_TODO = 0.5            # penalización al usar máscara de relanzar que no cambia nada

# Shaping potencial (recomendado para A2C)
ENABLE_POTENTIAL_SHAPING = True            # activar shaping denso durante relanzos
ALPHA_POTENTIAL = 0.10                     # intensidad del shaping
GAMMA_SHAPING = 0.995                      # usa el mismo gamma que el agente (aprox)

# No-op estricto al relanzar inválidamente (no cambia estado)
STRICT_NOOP_ON_INVALID_REROLL = True

CONSTANT_SCORES: Dict[Categoria, int] = {
    Categoria.FULL: 25,
    Categoria.ESCALERA_PEQUENA: 30,
    Categoria.ESCALERA_GRANDE: 40,
    Categoria.YAHTZEE: 50,
    Categoria.BONO_YAHTZEE: 100,
    Categoria.BONO_SUPERIOR: 35,
}

nombres_categoria = {
    Categoria.UNOS: 'Unos',
    Categoria.DOSES: 'Doses',
    Categoria.TRESES: 'Treses',
    Categoria.CUATROS: 'Cuatros',
    Categoria.CINCOS: 'Cincos',
    Categoria.SEISES: 'Seises',
    Categoria.TRIO: 'Trío',
    Categoria.POKER: 'Póker',
    Categoria.FULL: 'Full',
    Categoria.ESCALERA_PEQUENA: 'Escalera pequeña',
    Categoria.ESCALERA_GRANDE: 'Escalera grande',
    Categoria.YAHTZEE: 'Yahtzee',
    Categoria.CHANCE: 'Chance',
    Categoria.BONO_SUPERIOR: 'Bono superior',
    Categoria.BONO_YAHTZEE: 'Bono Yahtzee',
}

def CategoriaNombre(categoria: int) -> str:
    """
    Devuelve el nombre de la categoría correspondiente al índice dado.

    Args:
        categoria (int): Índice de la categoría.
    Returns:
        str: Nombre de la categoría.
    """
    return nombres_categoria[Categoria(int(categoria))]


class YahtzeeEnvV2(gym.Env):
    """
    Entorno Gymnasium: Yahtzee 1 jugador.

    Acciones (Discrete(44)):
      0..30  -> máscara de mantener para relanzar (5 bits; bit i==1 => mantener dado i)
      31..43 -> puntuar en una de las 13 categorías (31=UNOS ... 43=CHANCE)

    Observación (Box):
      [ronda_actual, tiradas_restantes, dado(5), frecuencias, puntuaciones(15)]
      # Tamaño total: 1 (ronda) + 1 (tiradas) + 5 (dados) + 6 (frecuencias 1..6) + len(Categoria)
      - puntuaciones[i] = -1 si vacío, si no, score en esa categoría/bono.

    Recompensas:
      - Relanzar: 0, salvo penalización si no quedan tiradas.
      - Puntuar: puntos otorgados (+ posibles bonos).
      - Shaping potencial (opcional): α * (γ * Φ(s') - Φ(s)), donde Φ(s) es el
        mejor score posible con los dados actuales sobre categorías libres.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        """
        Inicializa el entorno Yahtzee para un solo jugador.

        Args:
            render_mode (Optional[str], opcional): Modo de renderizado ('human' o 'rgb_array').
        """
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Juego
        self.max_rondas = 13
        self.rng = np.random.default_rng()

        # Espacios
        self.action_space = spaces.Discrete(CONSTANT_NUM_ACCIONES)

        # Observación
        # 1 (ronda) + 1 (tiradas) + 5 (dados) + 6 (frecuencias 1..6) + len(Categoria)  
        obs_size = 1 + 1 + 5 + 6 + len(Categoria)
        low = np.array([0, 0] + [1]*5 + [0]*6 + [-1]*len(Categoria), dtype=np.float32)
        puntuaciones_max = np.array([
            5, 10, 15, 20, 25, 30,      # Unos..Seises
            30, 30, 25, 30, 40, 50,     # Trío, Póker, Full, Esc. peq., Esc. gran., Yahtzee
            30, 35, 300                 # Chance, Bono sup., Bono Yahtzee
        ], dtype=np.float32)
        high = np.array([self.max_rondas, 3] + [6]*5 + [5]*6 + puntuaciones_max.tolist(), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_size,), dtype=np.float32)

        # Logger
        self.logger = logging.getLogger(__name__)

        # Estado inicial
        self.reset()

    # -----------------------------
    # Utilidades de estado/observación
    # -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Restablece el entorno a un estado inicial y devuelve la observación inicial y la información adicional.

        Args:
            seed (Optional[int], opcional): Semilla para el generador de números aleatorios. Por defecto es None.
            options (Optional[dict], opcional): Opciones adicionales para el reinicio (actualmente no se usan). Por defecto es None.

            observation: La observación inicial del entorno tras el reinicio.
            info: Información adicional sobre el entorno.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.ronda_actual = 0
        self.tiradas_restantes = 2
        self.dado = self.rng.integers(1, 7, size=5, dtype=np.int32)
        # -1 indica categoria no usada
        self.puntuaciones = np.full(len(Categoria), -1, dtype=np.int32)
        self.done = False

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self) -> np.ndarray:
        """
        Construye y retorna la observación actual del entorno Yahtzee como un array de NumPy.

        La observación consiste en la concatenación de las siguientes características:
            - Número de ronda actual (float32)
            - Número de tiradas restantes en la ronda actual (float32)
            - Valores actuales de los dados (array de float32)
            - Frecuencia de aparición de cada cara del dado (array de float32, longitud 6)
            - Puntuación actual para cada categoría (array de float32)

        Returns:
            np.ndarray: Vector de observación concatenado que representa el estado actual del entorno.
        """
        freqs = self._contar_frecuencia(self.dado).astype(np.float32)  # 6 valores [0..5]
        return np.concatenate([
            np.array([self.ronda_actual], dtype=np.float32),
            np.array([self.tiradas_restantes], dtype=np.float32),
            self.dado.astype(np.float32),
            freqs,
            self.puntuaciones.astype(np.float32),
        ])

    def _get_info(self) -> dict:
        """
        Genera y retorna un diccionario con información sobre el estado actual del juego.

        Returns:
            dict: Un diccionario con las siguientes claves:
                - "categorias_usadas" (list): Lista que indica qué categorías ya han sido utilizadas (1 si usada, 0 en caso contrario).
                - "action_mask" (list): Lista máscara que indica qué acciones están actualmente disponibles (1 si disponible, 0 en caso contrario).
                    - Si no quedan tiradas, las acciones de relanzar se deshabilitan.
                    - Las categorías ya utilizadas también se deshabilitan.
        """
        categorias_usadas = np.where(self.puntuaciones != -1, 1, 0).tolist()
        mask = [1] * CONSTANT_NUM_ACCIONES

        # Si no quedan tiradas, deshabilitar relanzos (0..30)
        if self.tiradas_restantes == 0:
            for i in range(CONSTANT_JUGADAS_POSIBLES):
                mask[i] = 0

        # Deshabilitar categorías ya usadas
        for i in range(CONSTANT_NUM_CATEGORIAS):
            if categorias_usadas[i] == 1:
                mask[CONSTANT_OFFSET_CATEGORIAS + i] = 0

        return {
            "categorias_usadas": categorias_usadas.copy(),
            "action_mask": mask,
        }

    def _terminated(self) -> bool:
        """
        Verifica si el juego actual ha alcanzado su condición de terminación.

        Returns:
            bool: True si el número de ronda actual es mayor o igual al número máximo de rondas, lo que indica que el juego ha terminado; False en caso contrario.
        """
        return self.ronda_actual >= self.max_rondas

    def _contar_frecuencia(self, dados: np.ndarray) -> np.ndarray:
        """
        Cuenta la frecuencia de aparición de cada valor de dado (1 a 6).

        Args:
            dados (np.ndarray): Array de valores de los dados.
        Returns:
            np.ndarray: Array de frecuencias para los valores 1 a 6.
        """
        return np.bincount(dados, minlength=7)[1:]

    def _escalera(self, length: int) -> bool:
        """
        Verifica si los dados actuales contienen una escalera de la longitud dada.

        Args:
            length (int): Longitud de la escalera a buscar (4 o 5).
        Returns:
            bool: True si existe la escalera, False en caso contrario.
        """
        unique = sorted(set(self.dado))
        for i in range(len(unique) - length + 1):
            if unique[i + length - 1] - unique[i] == length - 1:
                return True
        return False

    def _calcular_puntuacion(self, categoria: int) -> int:
        """
        Calcula la puntuación obtenida en la categoría indicada según los dados actuales.

        Args:
            categoria (int): Índice de la categoría a puntuar.
        Returns:
            int: Puntuación obtenida en la categoría.
        """
        conteo = self._contar_frecuencia(self.dado)
        puntuacion = 0

        match categoria:
            case Categoria.UNOS | Categoria.DOSES | Categoria.TRESES | Categoria.CUATROS | Categoria.CINCOS | Categoria.SEISES:
                puntuacion = conteo[categoria] * (categoria + 1)
            case Categoria.TRIO:
                puntuacion = sum(self.dado) if np.max(conteo) >= 3 else 0
            case Categoria.POKER:
                puntuacion = sum(self.dado) if np.max(conteo) >= 4 else 0
            case Categoria.FULL:
                es_full = (np.any(conteo == 3) and np.any(conteo == 2)) or np.any(conteo == 5)
                puntuacion = CONSTANT_SCORES[Categoria.FULL] if es_full else 0
            case Categoria.ESCALERA_PEQUENA:
                puntuacion = CONSTANT_SCORES[Categoria.ESCALERA_PEQUENA] if self._escalera(4) else 0
            case Categoria.ESCALERA_GRANDE:
                puntuacion = CONSTANT_SCORES[Categoria.ESCALERA_GRANDE] if self._escalera(5) else 0
            case Categoria.YAHTZEE:
                puntuacion = CONSTANT_SCORES[Categoria.YAHTZEE] if np.max(conteo) == 5 else 0
            case Categoria.CHANCE:
                puntuacion = sum(self.dado)
            case _:
                raise ValueError(f"Categoría inválida pasada a _calcular_puntuacion: {categoria}")

        self.logger.debug(f'Conteo {conteo.tolist()}: {self.dado.tolist()}')
        self.logger.debug(f'Puntuación calculada para {Categoria(categoria).name}: {puntuacion}')
        return int(puntuacion)

    def _calcular_puntuacion_final(self) -> int:
        """
        Calcula la puntuación total final sumando todas las categorías puntuadas.

        Returns:
            int: Puntuación total final del jugador.
        """
        return int(np.sum(self.puntuaciones[self.puntuaciones != -1]))

    def _categoria_usada(self, categoria: int) -> bool:
        """
        Indica si la categoría dada ya ha sido utilizada.

        Args:
            categoria (int): Índice de la categoría.
        Returns:
            bool: True si la categoría ya fue usada, False en caso contrario.
        """
        return self.puntuaciones[categoria] != -1

    def _phi(self) -> float:
        """
        Calcula el potencial Φ(s): la mejor puntuación alcanzable con los dados actuales en categorías libres.

        Returns:
            float: Mejor puntuación posible en las categorías libres.
        """
        libres = [c for c in range(int(Categoria.UNOS), int(Categoria.CHANCE) + 1) if not self._categoria_usada(c)]
        if not libres:
            return 0.0
        return float(max(self._calcular_puntuacion(c) for c in libres))

    # -----------------------------
    # Dinámica principal
    # -----------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Ejecuta un paso en el entorno Yahtzee dado una acción.

        La acción puede ser:
            - Relanzar dados (acciones 0..30): especifica qué dados mantener y cuáles relanzar.
            - Puntuar en una categoría (acciones 31..43): asigna la puntuación de los dados actuales a la categoría elegida.

        Args:
            action (int): Acción a ejecutar (0..43).

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                - obs (np.ndarray): Observación resultante tras la acción.
                - recompensa (float): Recompensa obtenida por la acción.
                - terminated (bool): True si la partida ha terminado.
                - truncated (bool): Siempre False (no se usa truncamiento).
                - info (dict): Información adicional sobre el estado del entorno.
        """
        if self.done:
            raise RuntimeError("La partida ha terminado. Llama a reset().")

        recompensa = 0.0
        phi_before = self._phi() if ENABLE_POTENTIAL_SHAPING else 0.0

        # --- RELANZAR (0..30)
        if action < CONSTANT_OFFSET_CATEGORIAS:
            if self.tiradas_restantes > 0:
                mantener = np.array([(action >> i) & 1 for i in range(5)], dtype=bool)
                for i in range(5):
                    if not mantener[i]:
                        self.dado[i] = self.rng.integers(1, 7, dtype=np.int32)
                self.tiradas_restantes -= 1
            else:
                # Relanzar inválido -> penalización y no-op estricto
                recompensa -= PENALIZACION_RELANZAR_SIN_TIRADAS
                if STRICT_NOOP_ON_INVALID_REROLL:
                    # NO cambiamos estado (dados/tiradas/ronda)
                    pass

        # --- PUNTUAR (31..43)
        else:
            categoria = action - CONSTANT_OFFSET_CATEGORIAS
            if (categoria < int(Categoria.UNOS) or categoria > int(Categoria.CHANCE)):
                raise ValueError(f"Categoría inválida seleccionada: {categoria} en step()")

            if self._categoria_usada(categoria):
                recompensa -= PENALIZACION_CATEGORIA_USADA
                self.logger.debug(f"Categoría ya usada: {CategoriaNombre(categoria)}")
            else:
                counts = self._contar_frecuencia(self.dado)
                yahtzee_actual = (np.max(counts) == 5)
                yahtzee_previo = (self.puntuaciones[Categoria.YAHTZEE] == CONSTANT_SCORES[Categoria.YAHTZEE])

                # Bono Yahtzee adicional
                if yahtzee_actual and yahtzee_previo:
                    if self.puntuaciones[Categoria.BONO_YAHTZEE] == -1:
                        self.puntuaciones[Categoria.BONO_YAHTZEE] = 0
                    self.puntuaciones[Categoria.BONO_YAHTZEE] += CONSTANT_SCORES[Categoria.BONO_YAHTZEE]
                    recompensa += CONSTANT_SCORES[Categoria.BONO_YAHTZEE]

                # Puntuación en la categoría seleccionada
                puntuacion = self._calcular_puntuacion(categoria)
                self.puntuaciones[categoria] = puntuacion
                recompensa += float(puntuacion)

                # Bono superior
                if not self._categoria_usada(Categoria.BONO_SUPERIOR) and categoria in range(6):
                    upper_score = int(np.sum(self.puntuaciones[:6]))
                    if upper_score >= 63:
                        if self.puntuaciones[Categoria.BONO_SUPERIOR] == -1:
                            self.puntuaciones[Categoria.BONO_SUPERIOR] = 0
                        self.puntuaciones[Categoria.BONO_SUPERIOR] = CONSTANT_SCORES[Categoria.BONO_SUPERIOR]
                        recompensa += float(CONSTANT_SCORES[Categoria.BONO_SUPERIOR])

                # Avanza ronda tras puntuar
                self.ronda_actual += 1
                self.tiradas_restantes = 2
                self.dado = self.rng.integers(1, 7, size=5, dtype=np.int32)

        # Observación e info
        obs = self._get_obs()
        terminated = self._terminated()
        truncated = False
        info = self._get_info()

        if terminated:
            info['final_score'] = self._calcular_puntuacion_final()

        # Shaping potencial (denso durante relanzos, mantiene óptimo con descuento gamma)
        if ENABLE_POTENTIAL_SHAPING:
            phi_after = self._phi()
            recompensa += ALPHA_POTENTIAL * (GAMMA_SHAPING * phi_after - phi_before)

        return obs, float(recompensa), terminated, truncated, info

    # -----------------------------
    # Renderización
    # -----------------------------
    def render(self):
        """
        Renderiza el entorno según el modo seleccionado.

        - 'human': imprime por consola el estado actual del juego.
        - 'rgb_array': devuelve un frame NumPy (H×W×3, uint8) representando el estado.
        """
        if self.render_mode == "human":
            dado_str = " ".join([f"🎲{d}" for d in self.dado])
            print("\n" + "="*40)
            print(f"🕒 Ronda: {self.ronda_actual + 1} / {self.max_rondas}")
            print(f"🎯 Dados: {dado_str}")
            print(f"🔁 Tiradas restantes: {self.tiradas_restantes}")
            print("📋 Tarjeta de puntuación:")
            print("="*40 + "")
            for i in range(13):
                nombre = nombres_categoria[Categoria(i)]
                usado = "✅" if self._categoria_usada(i) else "❌"
                puntuacion = self.puntuaciones[i]
                print(f" {nombre:<18} {puntuacion:>3} pts {usado}")
            print(f"🎁 Bono superior: {self.puntuaciones[Categoria.BONO_SUPERIOR]} pts")
            print(f"🔥 Bono Yahtzee: {self.puntuaciones[Categoria.BONO_YAHTZEE]} pts")
            print(f"🏁 Puntuación total: {self._calcular_puntuacion_final()} pts")
            print("="*40 + "")
            return None

        elif self.render_mode == "rgb_array":
            H, W = 128, 128
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            # Barra superior: estado de tiradas (verde si >0, rojo si 0)
            color = (0, 200, 0) if self.tiradas_restantes > 0 else (200, 0, 0)
            bar_h = 10
            frame[0:bar_h, :, 0] = color[0]
            frame[0:bar_h, :, 1] = color[1]
            frame[0:bar_h, :, 2] = color[2]
            # Barra inferior: marca de ronda (azul)
            frame[H - bar_h:H, :, 2] = 200
            return frame

        else:
            return None

    def close(self):
        """
        Cierra el entorno y libera recursos si es necesario.
        """
        pass