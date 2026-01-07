# -*- coding: utf-8 -*-
"""
final_evaluate_rl.py

Evaluación final para modelos entrenados con VecNormalize
sobre `yahtzee_env/Yahtzee-v2`. Carga el modelo y las estadísticas
( VecNormalize ), evalúa N episodios con recompensas SIN normalizar,
repara acciones inválidas y guarda métricas y ficheros de resultados.

Salidas por defecto (en `--best_dir`):
  - final_eval_summary.txt         -> resumen (media reward/score)
  - final_episode_rewards.csv      -> recompensas por episodio
  - final_episode_scores.csv       -> puntuaciones finales por episodio

Uso típico:
    python final_evaluate_rl.py \
      --best_dir ./best/qrdqn_abl_51/qrdqn_nq25_batch96/seed_304/best \
      --episodes 100 --seed 304 --algo qrdqn

Opcional:
    --stats_file   ruta explícita a stats VecNormalize (.pkl)
    --model_file   ruta explícita al modelo (.zip)
    --max_steps    límite de steps por episodio (default: 1000)
    --algo         'dqn' | 'qrdqn' (si no se especifica, se autodetecta)

Notas:
 - Requiere que el entorno `yahtzee_env/Yahtzee-v2` esté registrado.
 - La carga de stats y la evaluación con `training=False`, `norm_reward=False`
   sigue el patrón recomendado por Stable-Baselines3.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import yahtzee_env  # asegura el registro del entorno

from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import MaskablePPO, QRDQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper

# Constantes
MAX_STEPS_PER_EPISODE = 1000

# Configuracion logger
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger("final_evaluate_rl")

# -----------------------------
# Helpers de evaluacion SIN normalizar
# -----------------------------
# Constante de evaluacion
JUGADAS_POSIBLES = 31               # 0..30 = relanzar
OFFSET_CATEGORIAS = 31              # 31..43 = puntuar
NUM_CATEGORIAS = 13

def get_action_mask_and_state(eval_env):
    # Accede al env subyacente (DummyVecEnv) y saca la mascara y estado util
    base_env = eval_env.venv.envs[0].unwrapped
    info = base_env._get_info()
    mask = np.array(info["action_mask"], dtype=np.int8)
    tiradas_restantes = base_env.tiradas_restantes
    return mask, tiradas_restantes, base_env

def best_scoring_action(base_env):
    # Devuelve la accion de "puntuar" que mas recompensa da en el estado actual
    libres = [c for c in range(NUM_CATEGORIAS) if base_env.puntuaciones[c] == -1]
    if not libres:
        return OFFSET_CATEGORIAS + (NUM_CATEGORIAS - 1)  # CHANCE por defecto, si no quedase nada (case raro)
    scores = [(c, base_env._calcular_puntuacion(c)) for c in libres]
    best_cat = max(scores, key=lambda t: t[1])[0]
    return OFFSET_CATEGORIAS + best_cat
    
def repair_action_if_invalid(action, mask, tiradas_restantes, base_env):
    if mask[action] == 1:
        return action  # valida

    # Accion invalida: decide reparacion
    if tiradas_restantes == 0:
        # Fuerza a puntuar en la mejor categoria
        return best_scoring_action(base_env)
    else:
        # Estamos en fase de relanzar: escoge una jugada valida
        valid_roll_actions = np.where(mask[:JUGADAS_POSIBLES] == 1)[0]
        if len(valid_roll_actions) > 0:
            # Usa un relanzo que cambie algo: p.ej., 0 (relanzar todos) o aleatorio
            if 0 in valid_roll_actions:
                return 0
            return int(np.random.choice(valid_roll_actions))
        # Si no hubiera relanzos validos (case raro), puntua la mejor categoria
        return best_scoring_action(base_env)
        
# -----------------------------
# Wrapper: repara acciones invalidas antes de VecNormalize
# -----------------------------
class ActionMaskRepairVecWrapper(VecEnvWrapper):

    def __init__(self, venv):
        # Llama al constructor de VecEnvWrapper
        super().__init__(venv)

    # ---- Metodos requeridos por la interfaz VecEnv ----
    def reset(self):
        # Simplemente delega en el env envuelto
        return self.venv.reset()

    def step_async(self, actions):
        repaired = []
        for i, a in enumerate(actions):
            base_env = self.venv.envs[i].unwrapped
            info = base_env._get_info()
            mask = np.asarray(info["action_mask"], dtype=np.int8)
            tiradas_restantes = base_env.tiradas_restantes
            repaired.append(repair_action_if_invalid(int(a), mask, tiradas_restantes, base_env))
        return super().step_async(np.array(repaired))
    
    def step_wait(self):
        # Delegacion directa
        return self.venv.step_wait()


def autodetect_files(best_dir: Path, explicit_model: Optional[Path], explicit_stats: Optional[Path]) -> Tuple[Optional[Path], Optional[Path], str]:
    model_file = explicit_model if explicit_model else None
    stats_file = explicit_stats if explicit_stats else None
    algo = 'unknown'
    # Detecta modelo
    if model_file is None:
        #candidates_model = sorted(list(best_dir.glob("*_yahtzee_vecnorm_final.zip")))
        candidates_model = sorted(list(best_dir.glob("*_model.zip")))
        if not candidates_model:
            candidates_model = sorted(list(best_dir.glob("*.zip")))
        if candidates_model:
            model_file = candidates_model[0]
    # Detecta stats
    if stats_file is None:
        candidates_stats = sorted(list(best_dir.glob("*_stats.pkl")))
        if not candidates_stats:
            candidates_stats = sorted(list(best_dir.parent.glob("*_stats.pkl")))
        if candidates_stats:
            stats_file = candidates_stats[0]
    # Algoritmo por nombre
    if model_file is not None:
        name = model_file.name.lower().split("_")[0]
        if "a2c" in name:
            algo = 'a2c'
        elif "mppo" in name:
            algo = 'mppo'
        elif "ppo" in name:
            algo = 'ppo'
        elif "qrdqn" in name:
            algo = 'qrdqn'
        elif "dqn" in name:
            algo = 'dqn'
    logger.info(f"Autodeteccion: modelo={model_file}, stats={stats_file}, algo={algo}")
    return model_file, stats_file, algo


def load_model(model_path: Path, algo_hint: str):
    if algo_hint == 'a2c':
        return A2C.load(str(model_path))
    if algo_hint == 'ppo':
        return PPO.load(str(model_path))
    if algo_hint == 'mppo':
        return MaskablePPO.load(str(model_path))
    if algo_hint == 'dqn':
        return DQN.load(str(model_path))
    if algo_hint == 'qrdqn':
        return QRDQN.load(str(model_path))
    try:
        return DQN.load(str(model_path))
    except Exception:
        return QRDQN.load(str(model_path))


def evaluate(best_dir: Path, episodes: int, max_steps: int, seed: int, algo_force: Optional[str], model_file: Optional[Path], stats_file: Optional[Path]) -> Tuple[float, float]:
    # Asegurar que best_dir es Path aunque se reciba como str desde otros scripts
    try:
        from pathlib import Path as _Path
        best_dir = _Path(best_dir)
    except Exception:
        pass
    model_path, stats_path, algo_auto = autodetect_files(best_dir, model_file, stats_file)
    algo = (algo_force.lower().strip() if algo_force else algo_auto)
    if model_path is None:
        raise FileNotFoundError(f"No se encontró archivo de modelo .zip en {best_dir}")
    logger.info(f"Cargando modelo: {model_path}")
    logger.info(f"Algo detectado/forzado: {algo}")
    model = load_model(model_path, algo)
    # Entorno + stats
    final_env = make_vec_env("yahtzee_env/Yahtzee-v2", n_envs=1, seed=seed)
    final_env = ActionMaskRepairVecWrapper(final_env)
    if stats_path and stats_path.exists():
         # Cargar estadisticas (patron recomendado en SB3)
        final_env = VecNormalize.load(str(stats_path), final_env)
        final_env.training = False      # no actualizar estadisticas
        final_env.norm_reward = False   # obtener recompensas reales
    else:
        final_env = VecNormalize(final_env, training=False, norm_reward=False)
    
    # -----------------------------
    # Evaluacion SIN normalizar (metricas reales) con reparacion de acciones invalidas
    # -----------------------------
    logger.info("Evaluando el modelo con recompensas SIN normalizar(evaluacion segura)...")

    # Evaluacion
    num_episodes = episodes
    all_rewards, final_scores = [], []
    max_steps_per_episode = max_steps

    for ep in range(num_episodes):  
        obs = final_env.reset()
        ep_reward = 0.0
        steps = 0

        while steps < max_steps_per_episode:
            action, _ = model.predict(obs, deterministic=True)          
            action_scalar = int(np.asarray(action).item())
            logger.debug(f"Episodio {ep + 1}, Step {steps + 1}: Accion elegida: {action_scalar}")
            obs, rewards_step, dones, infos = final_env.step([action_scalar])
            ep_reward += float(rewards_step[0])
            steps += 1

            if dones[0]:
                # Prioriza final_score en info; fallback al env
                if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                    if "final_score" in infos[0]:
                        final_score = float(infos[0]["final_score"])
                    else:
                        # intenta leer del env subyacente
                        try:
                            final_score = base_env._calcular_puntuacion_final()
                        except Exception:
                            final_score = np.nan
                else:
                    final_score = np.nan

                final_scores.append(final_score)
                break

        # si agota max steps sin done
        if steps >= max_steps_per_episode:
            logger.warning(f"Episodio {ep + 1} alcanzo el max_steps={max_steps_per_episode} sin done=True; forzando reset.")
            all_rewards.append(ep_reward)
            final_scores.append(np.nan)
            continue

        logger.info(f"Episode {ep + 1}: Recompensa del episodio: {ep_reward}, Puntuacion final: {final_scores[-1]}")
        all_rewards.append(ep_reward)

    mean_reward = float(np.nanmean(all_rewards)) if all_rewards else float('nan')
    mean_score = float(np.nanmean(final_scores)) if final_scores else float('nan')
    # Guardar resultados
    np.savetxt(best_dir / f"{algo}_vecnorm_final_episode_rewards.txt", np.array(all_rewards))
    with open(best_dir / f"{algo}_vecnorm_final_evaluation_metrics.txt", "w") as f:
        f.write(f"Recompensa media (sin normalizar) en {num_episodes} episodios: {mean_reward}\n")
        f.write(f"Puntuacion final promedio: {mean_score}\n")
    logger.info(f"Metricas guardadas en {algo}_vecnorm_final_evaluation_metrics.txt")
    # Guardar resultados detallados
    np.savetxt(best_dir / f"{algo}_final_episode_rewards.csv", np.array(all_rewards), fmt="%.6f", delimiter=",")
    np.savetxt(best_dir / f"{algo}_final_episode_scores.csv", np.array(final_scores), fmt="%.6f", delimiter=",")
    with open(best_dir / f"{algo}_final_eval_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Episodios: {num_episodes}\n")
        f.write(f"Semilla: {seed}\n")
        f.write(f"Modelo: {model_path.name}\n")
        f.write(f"Stats: {stats_path.name if stats_path else 'N/A'}\n")
        f.write(f"Recompensa media (sin normalizar): {mean_reward}\n")
        f.write(f"Puntuacion final promedio: {mean_score}\n")
    print("\nResumen:")
    print(f"  Recompensa media (sin normalizar): {mean_reward:.4f}")
    print(f"  Puntuacion final promedio:        {mean_score:.4f}")
    return mean_reward, mean_score

def build_argparser():
    ap = argparse.ArgumentParser(description="Evaluación final modelo con VecNormalize (Yahtzee-v2)")
    ap.add_argument("--best_dir", type=str, required=True, help="Directorio 'best' del seed a evaluar")
    ap.add_argument("--episodes", type=int, default=100, help="Número de episodios de evaluación")
    ap.add_argument("--max_steps", type=int, default=1000, help="Máximo de steps por episodio")
    ap.add_argument("--seed", type=int, default=42, help="Semilla")
    ap.add_argument("--algo", type=str, default="", help="Forzar algoritmo: 'dqn' | 'qrdqn' (opcional)")
    ap.add_argument("--model_file", type=str, default="", help="Ruta explícita al .zip del modelo (opcional)")
    ap.add_argument("--stats_file", type=str, default="", help="Ruta explícita al .pkl de VecNormalize (opcional)")
    return ap


def main():
    args = build_argparser().parse_args()

    # Mostrar argumentos
    print("Argumentos de evaluacion:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    best_dir = Path(args.best_dir)
    if not best_dir.exists():
        print(f"ERROR: no existe --best_dir: {best_dir}")
        sys.exit(1)

    model_path = Path(args.model_file) if args.model_file else None
    stats_path = Path(args.stats_file) if args.stats_file else None
    algo_force = args.algo if args.algo else None

    # Ejecutar evaluacion
    evaluate(best_dir=best_dir, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
             algo_force=algo_force, model_file=model_path, stats_file=stats_path)


if __name__ == "__main__":
    main()
