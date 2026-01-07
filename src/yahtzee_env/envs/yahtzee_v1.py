import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum, IntEnum
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

CONSTANT_NUM_CATEGORIAS = 13  # Número de categorías jugables (sin contar bonos)
CONSTANT_JUGADAS_POSIBLES = 31  # Número de combinaciones posibles de mantener dados (2^5 - 1)
CONSTANT_NUM_ACCIONES = CONSTANT_JUGADAS_POSIBLES + CONSTANT_NUM_CATEGORIAS
CONSTANT_OFFSET_CATEGORIAS = 31  # Offset para las categorías jugables
PENALIZACION_CATEGORIA_USADA = 0.5  # Penalización por seleccionar una categoría ya usada
PENALIZACION_RELANZAR_SIN_TIRADAS = 0.5  # Penalización por intentar relanzar sin tiradas restantes
    
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

class YahtzeeEnvV1(gym.Env):
    """
     Gymnasium environment que simula una partida de Yahtzee para un jugador.

    - 5 dados (valores 1..6)
    - 13 rondas (una categoria puntuable por ronda)
    - En cada ronda el agente puede lanzar hasta 3 veces (lanzamiento inicial + hasta 2 relanzamientos).
    - Entre lanzamientos el agente selecciona qué dados mantener vía una máscara de 5 bits (acciones 0..30).
    - Cuando rolls_left == 0 (o el agente decide puntuar), debe escoger una categoría (13 categorías).

    Espacio de acciones:
    - Discrete(44):
      0..30 -> mascara de dados a mantener para el relanzamiento (bit i==1 => mantener dado i).
      31..43 -> seleccionar categoria de puntuacion (31 = UNOS, ..., 43 = CHANCE)

    Espacio de observacion (Box, vector aplanado):
    - [ronda_actual, tiradas_restantes, dado(5), puntuaciones(15)]
      * ronda_actual: 0..12
      * tiradas_restantes: 0..2
      * dado: valores 1..6
      * puntuaciones: -1 si vacio, o puntuacion actual por categoria (incluye bonos)

    Recompensa:
    - 0 para acciones de mantener/relanzar (salvo penalizacion suave si se relanza sin tiradas).
    - Al puntuar, la recompensa es la puntuacion obtenida por la categoria seleccionada,
      mas posibles bonos (superior y yahtzee adicional).
    """
    # Añadimos 'rgb_array' para compatibilidad con VecEnv/SB3
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        
        super(YahtzeeEnvV1, self).__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # configurar parámetros del juego
        self.max_rondas = 13
        # PRIMERO se crea rng
        self.rng = np.random.default_rng()
        
        # Definir el espacio de estados
        # 31 combinaciones de mantener (0-30),
        # y adicionalmante 13 acciones mas que se corresponde con puntuar en alguna 
        # de las 13 categorías
        # Discrete action space permite utilizar masking de acciones
        self.action_space = spaces.Discrete(CONSTANT_NUM_ACCIONES)

        # Definición del espacio de observación:
        # - "ronda_actual": número de la ronda actual (0 a 13).
        # - "tiradas_restantes": número de tiradas restantes en la ronda actual (0 a 3).
        # - "dado": vector de 5 enteros entre 1 y 6, representando los valores actuales de los dados.
        # - "puntuaciones": vector de 15 enteros, cada uno con un límite superior realista según la categoría.
        # Tamaño total: 1 (ronda) + 1 (tiradas) + 5 (dados) + 15 (puntuaciones y bonos)        
        obs_size = 1 + 1 + 5 + len(Categoria)
        low = np.array([0, 0] + [1]*5 + [-1]*len(Categoria), dtype=np.float32)
        # Establece límites superiores realistas para cada categoría de la tarjeta de puntuación
        puntuaciones_max = np.array([
            5,  # Unos (max 5*1)
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
            300  # Bono Yahtzee (fijo)
        ], dtype=np.float32)        
        high = np.array([self.max_rondas, 3] + [6]*5 + puntuaciones_max.tolist(), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_size,), dtype=np.float32)

        # Configurar el logger
        self.logger = logging.getLogger(__name__)
  
        # Inicializar estado del juego
        self.reset()
                                   
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reinicia el entorno de Yahtzee para comenzar una nueva partida.

        Args:
            seed (int, opcional): Semilla para el generador de números aleatorios.
            options (dict, opcional): Opciones adicionales para el reinicio (no utilizadas).

        Returns:
            observation (np.ndarray): Estado inicial del entorno tras el reinicio.
            info (dict): Información adicional (incluye action_mask).
        """
        # Semilla
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.ronda_actual = 0
        self.tiradas_restantes = 2
        self.dado = self.rng.integers(1, 7, size=5, dtype=np.int32)
        self.puntuaciones = np.full(len(Categoria), -1, dtype=np.int32)
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
        """Devuelve información adicional sobre el estado del juego, incluye action_mask."""
        categorias_usadas = np.where(self.puntuaciones != -1, 1, 0).tolist()
        mask = [1]*CONSTANT_NUM_ACCIONES
         # Si no quedan tiradas, deshabilitar acciones 0..30 (mantener/relanzar)
        if self.tiradas_restantes == 0:
            for i in range(CONSTANT_JUGADAS_POSIBLES):
                mask[i] = 0
        # Deshabilitar categori­as ya usadas
        for i in range(CONSTANT_NUM_CATEGORIAS):
            if categorias_usadas[i] == 1:
                mask[CONSTANT_OFFSET_CATEGORIAS + i] = 0
        return {
            "categorias_usadas": categorias_usadas.copy(),
            "action_mask": mask
        }

    def getTipoAccion(self, action):
        return TipoAccion.LANZAR if action < CONSTANT_JUGADAS_POSIBLES else TipoAccion.PUNTUAR

    def getMantener(self, action):
        if action < CONSTANT_JUGADAS_POSIBLES:
            mantener = [int((action >> i) & 1) for i in range(5)]
            return mantener
        else:
            # Si la accion es puntuar, se consideran todos los dados como mantenidos
            return [1]*5  
    
    def getCategoria(self, action):
        if action >= CONSTANT_OFFSET_CATEGORIAS:
            categoria = action - CONSTANT_OFFSET_CATEGORIAS
            return categoria
        else:
            return None  # Si la acción es lanzar, no hay categoría asociada 

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
        conteo_frecuencia = np.bincount(dados, minlength=7)[1:]  # Ignorar el índice 0
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
                # En las reglas oficiales, un Yahtzee (5 iguales) también cuenta como Full si el jugador decide anotarlo ahí.
                # Full: Tres dados de un número y dos de otro. Vale 25 puntos.
                puntuacion = CONSTANT_SCORES[Categoria.FULL] if (np.any(conteo_frecuencia == 3) and np.any(conteo_frecuencia == 2)) or np.any(conteo_frecuencia == 5) else 0
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
                raise ValueError(f"Categoría inválida pasada a _calcular_puntuacion: {categoria}")
        self.logger.debug(f'Conteo {conteo_frecuencia.tolist()}: {self.dado.tolist()}')
        self.logger.debug(f'Puntuación calculada para {Categoria(categoria).name}: {puntuacion}')
        return puntuacion

    def _escalera(self, length):
        """
        Comprueba si los dados contienen una secuencia (escalera) de longitud especificada.
        Considera duplicados correctamente en casos complejos (ej. [1,2,2,3,4] debería contar como escalera pequeña).

        Args:
            length (int): Longitud de la escalera a buscar.

        Returns:
            bool: True si existe una escalera de la longitud dada, False en caso contrario.
        """
        unique = sorted(set(self.dado))
        for i in range(len(unique) - length + 1):
            if unique[i + length - 1] - unique[i] == length - 1:
                return True
        return False
            
    def _calcular_puntuacion_final(self):
        """
        Calcula la puntuación final de la partida de Yahtzee.

        Returns:
            int: Puntuación final de la partida.
        """
        return np.sum(self.puntuaciones[self.puntuaciones != -1])

    def _categoria_usada(self, categoria):
        return not self.puntuaciones[categoria] == -1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("La partida ha terminado. Llama a reset().")

        recompensa = 0.0
        
        # --------------------------------------------------------------
        # ACCIÓN DE RELANZAMIENTO DE DADOS
        # --------------------------------------------------------------
        if action < CONSTANT_OFFSET_CATEGORIAS:
            # LANZAR DADOS
            if self.tiradas_restantes > 0:
                mantener = np.array([(action >> i) & 1 for i in range(5)], dtype=bool)
                for i in range(5):
                    if not mantener[i]:
                        # Volver a tirar dado i (aleatorio)
                        self.dado[i] = self.rng.integers(1, 7, dtype=np.int32)
                self.tiradas_restantes -= 1
            else:
                self.logger.debug("No quedan tiradas restantes, la acción de lanzar dados se ignora.")
                # Sin tiradas disponibles → penalización leve
                recompensa -= PENALIZACION_RELANZAR_SIN_TIRADAS
        # --------------------------------------------------------------
        # ACCIÓN DE PUNTUACIÓN EN UNA CATEGORÍA
        # --------------------------------------------------------------
        else:
            # PUNTUAR
            categoria = action - CONSTANT_OFFSET_CATEGORIAS

            # validad categoría
            if (categoria < Categoria.UNOS or categoria > Categoria.CHANCE):
                raise ValueError(f"Categoría inválida seleccionada: {categoria} en step()")

            # solo permitir categorías no usadas
            if self._categoria_usada(categoria):                
                recompensa -= PENALIZACION_CATEGORIA_USADA
                self.logger.debug(f"Categoría ya usada seleccionada: {CategoriaNombre(categoria)} en step()")

            else:
                # --- Categoria no usada ---
                counts = self._contar_frecuencia(self.dado)
                yahtzee_actual = (np.max(counts) == 5)
                yahtzee_previo = (self.puntuaciones[Categoria.YAHTZEE] == CONSTANT_SCORES[Categoria.YAHTZEE])
            
                # --- Yahtzee adicional (Bonus Yahtzee) ---
                if yahtzee_actual and yahtzee_previo:
                    # agregar bono yahtzee
                    if self.puntuaciones[Categoria.BONO_YAHTZEE] == -1:
                        self.puntuaciones[Categoria.BONO_YAHTZEE] = 0  # inicializar bono yahtzee si no se ha hecho antes
                    self.puntuaciones[Categoria.BONO_YAHTZEE] += CONSTANT_SCORES[Categoria.BONO_YAHTZEE]
                    recompensa += CONSTANT_SCORES[Categoria.BONO_YAHTZEE]
                    # y anotar en la categoria seleccionada
                    puntuacion = self._calcular_puntuacion(categoria)
                    self.puntuaciones[categoria] = puntuacion
                    recompensa += puntuacion

                # --- Primer Yahtzee y Resto de categorías ---
                else:
                    puntuacion = self._calcular_puntuacion(categoria)
                    self.puntuaciones[categoria] = puntuacion
                    recompensa += puntuacion
                
                # Recalcular bono superior
                if not self._categoria_usada(Categoria.BONO_SUPERIOR) and categoria in range(6):
                    upper_score = int(np.sum(self.puntuaciones[:6]))
                    if upper_score >= 63:
                        if self.puntuaciones[Categoria.BONO_SUPERIOR] == -1:
                            self.puntuaciones[Categoria.BONO_SUPERIOR] = 0  # inicializar bono superior si no se ha hecho antes
                        self.puntuaciones[Categoria.BONO_SUPERIOR] = CONSTANT_SCORES[Categoria.BONO_SUPERIOR]
                        recompensa += CONSTANT_SCORES[Categoria.BONO_SUPERIOR]
                
                # Avanzar ronda
                # Despues de puntuar siempre incrementamos ronda
                # Reiniciamos los dados y las tiradas
                self.ronda_actual += 1
                self.tiradas_restantes = 2
                self.dado = self.rng.integers(1, 7, size=5, dtype=np.int32)
        # --------------------------------------------------------------

        # actualizar done, obs, info
        obs = self._get_obs()
        terminated = self._terminated()
        truncated = False
        info = self._get_info() 

        return obs, float(recompensa), terminated, truncated, info
    
    def render(self):
        """
        Render:
        - 'human': imprime por consola el estado actual.
        - 'rgb_array': devuelve un frame NumPy (HÃ—WÃ—3, uint8).
        """
        if self.render_mode == "human":
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
                usado = "✅" if self._categoria_usada(i) else "❌"
                puntuacion = self.puntuaciones[i]
                print(f"  {nombre:<18} | {puntuacion:>3} pts | {usado}")
            print(f"🎁 Bono superior: {self.puntuaciones[Categoria.BONO_SUPERIOR]} pts")
            print(f"🔥 Bono Yahtzee: {self.puntuaciones[Categoria.BONO_YAHTZEE]} pts")
            print(f"🏁 Puntuación total: {self._calcular_puntuacion_final()} pts")
            print("="*40 + "")
            return None
        elif self.render_mode == "rgb_array":
            # Frame placeholder sencillo para cumplir la interfaz
            H, W = 128, 128
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            # Barra superior de estado: verde si quedan tiradas, roja si no
            color = (0, 200, 0) if self.tiradas_restantes > 0 else (200, 0, 0)
            bar_h = 10
            frame[0:bar_h, :, 0] = color[0]
            frame[0:bar_h, :, 1] = color[1]
            frame[0:bar_h, :, 2] = color[2]
            # Barra inferior para indicar nÃºmero de ronda (azul)
            frame[H-bar_h:H, :, 2] = 200
            return frame
        else:
            return None

    def close(self):
        pass