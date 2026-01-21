# -*- coding: utf-8 -*-
"""
play_yahtzee_agent_vs_human.py

Simulador CLI para jugar una partida de Yahtzee (versión V2) entre:
  - Un **agente entrenado** (cargando un modelo SB3 y las stats de VecNormalize)
  - Un **jugador humano** (interactivo por consola)

Características:
- Alterna turnos: primero Agente, luego Humano, hasta completar 13 rondas por jugador.
- Cada jugador tiene su **propio entorno** y tarjeta de puntuación (como en el juego estándar).
- Para el agente, se normalizan **solo las observaciones** según VecNormalize cargado;
  las recompensas se dejan **sin normalizar** (training=False, norm_reward=False).
- Valida/ayuda con acciones: muestra los dados, tiradas restantes y categorías libres.

Uso típico:
  python play_yahtzee_agent_vs_human.py \
    --model_file best/dqn_variants/.../best/dqn_yahtzee_vecnorm_final.zip \
    --stats_file best/dqn_variants/.../best/dqn_vecnorm_stats.pkl \
    --algo dqn \
    --seed 42

Requisitos:
- Tener disponible la clase del entorno YahtzeeEnvV2 (yahtzee_v2.py) en el PYTHONPATH o mismo directorio.
- Tener instalado Stable-Baselines3 (y sb3-contrib si usas QRDQN/MaskablePPO).
"""
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

# SB3 / SB3-contrib
try:
    from stable_baselines3 import A2C, PPO, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, VecNormalize
except Exception as e:
    print("ERROR: No se puede importar stable_baselines3. ¿Está instalado?", e)
    sys.exit(1)

try:
    from sb3_contrib import QRDQN, MaskablePPO
except Exception:
    QRDQN = None
    MaskablePPO = None

# Entorno (clase local)
try:
    from yahtzee_env.envs.yahtzee_v2 import YahtzeeEnvV2, Categoria, CONSTANT_NUM_ACCIONES, CONSTANT_JUGADAS_POSIBLES, CONSTANT_OFFSET_CATEGORIAS, CONSTANT_NUM_CATEGORIAS
except Exception as e:
    print("ERROR: No se encuentra yahtzee_v2.YahtzeeEnvV2 en el entorno.", e)
    sys.exit(1)

from final_evaluate_rl import ActionMaskRepairVecWrapper, repair_action_if_invalid

# --------- Utilidades human-friendly ---------
CAT_NAMES = {
    0:"Unos",1:"Doses",2:"Treses",3:"Cuatros",4:"Cincos",5:"Seises",
    6:"Trío",7:"Póker",8:"Full",9:"Esc. pequeña",10:"Esc. grande",11:"Yahtzee",12:"Chance"
}

def print_state_player(title: str, env: YahtzeeEnvV2) -> None:
    dados = " ".join([str(d) for d in env.dado])
    libres = [i for i in range(CONSTANT_NUM_CATEGORIAS) if env.puntuaciones[i] == -1]
    libres_names = ", ".join([f"{i}:{CAT_NAMES[i]}" for i in libres])
    print(f"\n=== {title} ===")
    print(f"Ronda: {env.ronda_actual+1}/{env.max_rondas} | Tiradas restantes: {env.tiradas_restantes}")
    print(f"Dados: {dados}")
    print(f"Categorías libres: {libres_names if libres else 'Ninguna'}")
    print(f"Puntuación parcial: {env._calcular_puntuacion_final()}\n")

def ask_human_action(env: YahtzeeEnvV2) -> int:
    mask = np.array(env._get_info()["action_mask"], dtype=np.int8)
    while True:
        if env.tiradas_restantes > 0:
            print("Opciones:")
            print("  - Relanzar: escribe 'r' seguido de los índices de dados a MANTENER (0..4), p.ej.: r 0 2 4")
            print("  - Puntuar:  escribe 'p' seguido del índice de categoría (0..12), p.ej.: p 12  # 12=Chance")
        else:
            print("No quedan tiradas: debes puntuar. Ejemplo: p 12")
        s = input("Tu jugada> ").strip().lower()
        parts = s.split()
        if not parts:
            continue
        if parts[0] == 'r' and env.tiradas_restantes > 0:
            keep = set()
            for x in parts[1:]:
                try:
                    idx = int(x)
                    if 0 <= idx <= 4:
                        keep.add(idx)
                except Exception:
                    pass
            # construir máscara de mantener: bit i=1 si mantener dado i
            action = 0
            for i in keep:
                action |= (1 << i)
            if mask[action] == 1:
                return action
            else:
                print("Relanzar inválido según action_mask; intenta otra combinación.")
        elif parts[0] == 'p':
            if len(parts) < 2:
                print("Falta índice de categoría (0..12)")
                continue
            try:
                c = int(parts[1])
            except Exception:
                print("Índice de categoría no válido.")
                continue
            if 0 <= c <= 12:
                action = CONSTANT_OFFSET_CATEGORIAS + c
                if mask[action] == 1:
                    return action
                else:
                    print("Esa categoría no está permitida ahora (ocupada o inválida).")
            else:
                print("Categoría fuera de rango (0..12)")
        else:
            print("Entrada no reconocida. Usa 'r ...' o 'p <cat>'.")

# --------- Carga de modelo ---------
ALGO_LOADERS = {
    'a2c': lambda p: A2C.load(str(p)),
    'ppo': lambda p: PPO.load(str(p)),
    'mppo': (lambda p: MaskablePPO.load(str(p))) if MaskablePPO else None,
    'dqn': lambda p: DQN.load(str(p)),
    'qrdqn': (lambda p: QRDQN.load(str(p))) if QRDQN else None,
}

def autodetect_algo(model_path: Path) -> str:
    name = model_path.name.lower()
    for k in ('a2c','ppo','mppo','qrdqn','dqn'):
        if k in name:
            return k
    return 'dqn'

# --------- Main ---------

def main():
    ap = argparse.ArgumentParser(description="Agente vs humano en Yahtzee V2 (carga modelo + VecNormalize, sin re-entrenar)")
    ap.add_argument('--model_file', type=str, required=True, help='Ruta al .zip del modelo SB3')
    ap.add_argument('--stats_file', type=str, default='', help='Ruta al .pkl de VecNormalize (opcional, muy recomendado)')
    ap.add_argument('--algo', type=str, default='', help="Algoritmo: dqn|qrdqn|ppo|a2c|mppo (autodetecta por nombre si vacío)")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--debug', action='store_true', help='Imprime información detallada de acciones/máscaras')
    args = ap.parse_args()

    model_path = Path(args.model_file)
    stats_path = Path(args.stats_file) if args.stats_file else None
    algo = args.algo.strip().lower() or autodetect_algo(model_path)

    if algo not in ALGO_LOADERS or ALGO_LOADERS.get(algo) is None:
        print(f"ERROR: algoritmo '{algo}' no soportado o sb3-contrib no instalado.")
        sys.exit(1)

    print(f"Cargando modelo ({algo.upper()}): {model_path}")
    model = ALGO_LOADERS[algo](model_path)

    # Entorno del AGENTE: DummyVecEnv -> ActionMaskRepair -> VecNormalize (cargado o nuevo)
    def _make_env():
        return YahtzeeEnvV2(render_mode=None)

    agent_env_vec = DummyVecEnv([_make_env])
    #agent_env_vec = ActionMaskRepairVecWrapper(agent_env_vec)
    if stats_path and stats_path.exists():
        agent_env_vec = VecNormalize.load(str(stats_path), agent_env_vec)
        agent_env_vec.training = False
        agent_env_vec.norm_reward = False
        print(f"Stats VecNormalize cargadas: {stats_path}")
    else:
        print("[AVISO] No se proporcionaron stats VecNormalize; se creará wrapper default (training=False, norm_reward=False)")
        agent_env_vec = VecNormalize(agent_env_vec, training=False, norm_reward=False)

    # Entorno del HUMANO: instancia directa (no vectorizado)
    human_env = YahtzeeEnvV2(render_mode='human')
    # Semillas
    _seed = args.seed
    agent_env_vec.seed(_seed)
    human_env.reset(seed=_seed)
    agent_obs = agent_env_vec.reset()  # obs normalizadas según stats

    # Bucle de juego
    print("\n=== Comienza la partida: Agente vs Humano ===")
    turn = 'agent'  # alterna 'agent' / 'human'
    agent_done = False
    human_done = False
    agent_final_score = None
    human_final_score = None
    ronda_actual = 0
    ronda_actual_agente = agent_env_vec.venv.envs[0].unwrapped.ronda_actual
    ronda_actual_humano = 0

    while not (agent_done and human_done):
        if turn == 'agent':
            while (ronda_actual_agente <= ronda_actual and not agent_done):
            #if not agent_done:
                # Mostrar estado del agente (desde env base)
                base_env = agent_env_vec.venv.envs[0].unwrapped
                print_state_player("Turno AGENTE", base_env)
                # Predicción de acción
                action, _ = model.predict(agent_obs, deterministic=True)
                action_scalar = int(np.asarray(action).item())

                info = base_env._get_info()
                mask = np.asarray(info["action_mask"], dtype=np.int8)
                if args.debug:
                    if action_scalar < CONSTANT_OFFSET_CATEGORIAS:
                        keep = [i for i in range(5) if (action_scalar >> i) & 1]
                        vals = [int(base_env.dado[i]) for i in keep] if keep else []
                        print(f"[DEBUG] AGENTE relanzar: mask={action_scalar} keep_idx={keep} keep_vals={vals} tiradas={base_env.tiradas_restantes}")
                    else:
                        cat = action_scalar - CONSTANT_OFFSET_CATEGORIAS
                        print(f"[DEBUG] AGENTE puntuar: categoría={cat} ({CAT_NAMES.get(cat,'?')}) tiradas={base_env.tiradas_restantes}")
                repaired = repair_action_if_invalid(action_scalar, mask, base_env.tiradas_restantes, base_env)
                if repaired != action_scalar:
                    print(f"[AGENTE] Acción predicha: {action_scalar} -> reparada: {repaired}")
                else:
                    print(f"[AGENTE] Acción predicha (válida): {action_scalar}")
                # Paso
                agent_obs, agent_rewards, agent_dones, agent_infos = agent_env_vec.step([repaired])
                if args.debug:
                    # Estado tras el paso del agente
                    print_state_player("Turno AGENTE (post-step)", base_env)
                if agent_dones[0]:
                    agent_done = True
                    info0 = agent_infos[0] if isinstance(agent_infos, (list, tuple)) and agent_infos else {}
                    agent_final_score = float(info0.get('final_score', base_env._calcular_puntuacion_final()))
                    print(f"[AGENTE] Ronda completada. Puntuación final: {agent_final_score}\n")
                ronda_actual_agente = base_env.ronda_actual

            # alternar
            turn = 'human'
            continue

        if turn == 'human':
            while (ronda_actual_humano <= ronda_actual and not human_done):
            #if not human_done:
                print_state_player("Turno HUMANO", human_env)
                action_h = ask_human_action(human_env)
                obs_h, reward_h, terminated_h, truncated_h, info_h = human_env.step(action_h)
                if args.debug:
                    # Estado tras el paso del humano
                    print_state_player("Turno HUMANO (post-step)", human_env)
                if terminated_h:
                    human_done = True
                    human_final_score = float(info_h.get('final_score', human_env._calcular_puntuacion_final()))
                    print(f"[HUMANO] Ronda completada. Puntuación final: {human_final_score}\n")
                ronda_actual_humano = human_env.ronda_actual

            # alternar
            turn = 'agent'
            ronda_actual += 1
            continue

    # Resultado final
    print("\n=== Resultado Final ===")
    print(f"AGENTE: {agent_final_score:.2f} pts")
    print(f"HUMANO: {human_final_score:.2f} pts")
    if agent_final_score > human_final_score:
        print("Ganador: AGENTE 🧠")
    elif human_final_score > agent_final_score:
        print("Ganador: HUMANO 🧑")
    else:
        print("Empate 🤝")

if __name__ == '__main__':
    main()
