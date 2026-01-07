"""
Entrenamiento de un agente PPO para el entorno Yahtzee-v2 usando Stable Baselines3 y VecNormalize.

Este script permite configurar el entrenamiento en varias fases, aplicar wrappers de reparación y normalización,
guardar el mejor modelo y evaluar el desempeño final. Incluye opciones avanzadas para ajustar hiperparámetros,
reproducibilidad y arquitectura de la red.
"""
import os
import logging
import argparse
import json
import ast
import re
import yahtzee_env
from pathlib import Path
from final_evaluate_rl import evaluate, ActionMaskRepairVecWrapper, MAX_STEPS_PER_EPISODE

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from callbacks import EpisodeRewardCallback , RewardLoggerCallback, RawRewardTrackerVecWrapper
from helper import plot_reward_evolution

MODEL_NAME = "ppo"

def parse_policy_arch(s: str):
    """
    Devuelve un diccionario con las arquitecturas de las redes de política y valor a partir de una cadena en formato JSON,
    literal de Python o un formato alternativo.

    Args:
        s (str): Cadena que representa la arquitectura.
    Returns:
        dict: Diccionario con claves 'pi' y 'vf' para las capas de cada red.
    """
    if not s:
        return {"pi": [256, 256], "vf": [256, 256]}
    # 1) JSON
    try:
        d = json.loads(s)
        if isinstance(d, dict) and "pi" in d and "vf" in d:
            return d
    except Exception:
        pass
    # 2) Literal Python (por si pasas dict como texto)
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict) and "pi" in d and "vf" in d:
            return d
    except Exception:
        pass
    # 3) Fallback: formato tipo "pi=[256, 256], vf=[256, 256]"
    m_pi = re.search(r"pi\s*=\s*\[([^\]]+)\]", s)
    m_vf = re.search(r"vf\s*=\s*\[([^\]]+)\]", s)
    def to_list(txt): return [int(x) for x in re.findall(r"\d+", txt or "256,256")]
    pi = to_list(m_pi.group(1) if m_pi else "")
    vf = to_list(m_vf.group(1) if m_vf else "")
    return {"pi": pi or [256, 256], "vf": vf or [256, 256]}
    
# -----------------------------
# Argumentos de linea de comandos
# -----------------------------

def build_argparser():
    """
    Construye el parser de argumentos para la línea de comandos.

    Returns:
        argparse.ArgumentParser: Parser configurado con los argumentos necesarios.
    """
    parser = argparse.ArgumentParser(description=f"Entrenar {MODEL_NAME.upper()} con VecNormalize (Yahtzee-v2)")
    # Fases
    parser.add_argument("--timesteps", type=int, default=400_000, help="Timesteps Fase 1")
    parser.add_argument("--timesteps2", type=int, default=400_000, help="Timesteps Fase 2")
    parser.add_argument("--timesteps3", type=int, default=300_000, help="Timesteps Fase 3")
    # Entornos / reproducibilidad
    parser.add_argument("--n_envs", type=int, default=8, help="Numero de entornos paralelos")
    parser.add_argument("--seed", type=int, default=42, help="Semilla")
    # Hiperparametros
    parser.add_argument("--lr", type=float, default=3e-4, help="learning_rate")
    parser.add_argument("--n_steps", type=int, default=2048, help="n_steps por entorno")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamaño de batch")
    parser.add_argument("--n_epochs", type=int, default=10, help="Numero de epochs por update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Factor de descuento")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Lambda de GAE")
    parser.add_argument("--ent_coef", type=float, default=0.02, help="Coeficiente de entropía")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clip range")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Coeficiente de loss de valor")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximo gradiente norm")
    parser.add_argument("--target_kl", type=float, default=None, help="KL objetivo para early stopping (None para desactivar)")
    parser.add_argument("--use_sde", type=lambda s: s.lower() in ["true", "1", "yes"], default=False,
                        help="Usar exploracion SDE (por defecto: True)")
    parser.add_argument("--sde_sample_freq", type=int, default=-1, help="Frecuencia de muestreo SDE")
    parser.add_argument("--policy_arch", type=str, default='{"pi":[256, 256], "vf":[256, 256]}',
                        help="Arquitectura de la red en formato JSON")
    # Evaluacion
    parser.add_argument("--eval_episodes", type=int, default=50, help="Episodios por evaluacion")
    parser.add_argument("--best_dir", type=str, default="./best", help="Carpeta para guardar el mejor modelo")
    # opcional: un suavizado largo para la grafica
    parser.add_argument("--plot_ma_long", type=int, default=200, help="Ventana larga para media movil en la grafica")
    parser.add_argument("--repair_actions_training", type=lambda s: s.lower() in ["true", "1", "yes"], default=True,
                        help="Aplica reparacion de acciones invalidas durante el entrenamiento (por defecto: True)")

    return parser

def main():
    """
    Función principal de entrenamiento y evaluación del agente PPO para Yahtzee-v2.

    - Configura los entornos paralelos y wrappers de reparación y normalización.
    - Realiza el entrenamiento en tres fases con diferentes coeficientes de entropía.
    - Guarda el modelo y las estadísticas de normalización.
    - Grafica la evolución de la recompensa.
    - Evalúa el modelo final sin normalización para obtener métricas reales.
    """
    args = build_argparser().parse_args()

    # Mostrar argumentos
    print(f"Argumentos de entrenamiento {MODEL_NAME.upper()}:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    logger = logging.getLogger(f"train_{MODEL_NAME.lower()}_vecnorm")

    # -----------------------------
    # VecEnv (+ reparacion opcional) + VecNormalize
    # -----------------------------
    logger.info(f"Creando {args.n_envs} entornos paralelos...")
    vec_env = make_vec_env("yahtzee_env/Yahtzee-v2", n_envs=args.n_envs, seed=args.seed)

    if args.repair_actions_training:
        logger.info("Activando wrapper de reparacion de acciones invalidas durante TRAINING...")
        vec_env = ActionMaskRepairVecWrapper(vec_env)  # reparar primero

    logger.info("Activando wrapper de recompensa REAL por episodio (antes de VecNormalize)...")
    vec_env = RawRewardTrackerVecWrapper(vec_env)
    logger.info("Aplicando VecNormalize (obs y reward normalizados)...")
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # -----------------------------
    # Fase 1
    # -----------------------------
    arch = parse_policy_arch(args.policy_arch)
    policy_kwargs = dict(net_arch=dict(pi=arch["pi"], vf=arch["vf"]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        seed=args.seed,
        policy_kwargs=policy_kwargs, 
    )

    # --- Callbacks ---
    # --- EpisodeRewardCallback (no depende de info["episode"]) ---
    train_cb = EpisodeRewardCallback(n_envs=args.n_envs)

    try:
        logger.info(f"Entrenando {MODEL_NAME.upper()} Fase 1 por {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps, callback=train_cb)
    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido manualmente en Fase 1.")

    # -----------------------------
    # Fase 2, ajuste fino, menor entropia -> mas explotacion
    # -----------------------------
    model2 = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps*2,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef/2,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        seed=args.seed,
        policy_kwargs=policy_kwargs, 
    )
    model2.set_parameters(model.get_parameters())
    try:
        logger.info(f"Entrenando {MODEL_NAME.upper()} Fase 2 por {args.timesteps2} timesteps...")
        model2.learn(total_timesteps=args.timesteps2,callback=train_cb)
    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido manualmente en Fase 2.")
    model = model2

    # -----------------------------
    # Fase 3, entropia muy baja -> casi pura explotacion
    # -----------------------------
    model3 = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps*3,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef/4,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        seed=args.seed,
        policy_kwargs=policy_kwargs, 
    )
    model3.set_parameters(model.get_parameters())
    try:
        logger.info(f"Entrenando {MODEL_NAME.upper()} Fase 3 por {args.timesteps3} timesteps...")
        model3.learn(total_timesteps=args.timesteps3,callback=train_cb)
    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido manualmente en Fase 3.")
    model = model3

    # Guardar modelo y estadisticas VecNormalize
    os.makedirs(args.best_dir, exist_ok=True)
    model_path = os.path.join(args.best_dir, f"{MODEL_NAME.lower()}_yahtzee_vecnorm_model")
    stats_path = os.path.join(args.best_dir, f"{MODEL_NAME.lower()}_yahtzee_vecnorm_stats.pkl")
    model.save(model_path)
    vec_env.save(stats_path)
    logger.info(f"Modelo guardado en {model_path} y estadisticas de normalizacion en {stats_path}")

    # -----------------------------
    # Grafica (recompensas del callback, potencialmente normalizadas)    
    # -----------------------------
    plot_reward_evolution(
        rewards=train_cb.episode_rewards,
        best_dir=args.best_dir,
        algo_name=MODEL_NAME,
        n_envs=args.n_envs,
        window_short=10,
        window_long=args.plot_ma_long,
        y_label="Recompensa",
        alpha_series=0.6
    )

    # -----------------------------
    # Evaluacion SIN normalizar (metricas reales) con reparacion de acciones invalidas
    # -----------------------------

    evaluate(best_dir=Path(args.best_dir), episodes=args.eval_episodes, max_steps=MAX_STEPS_PER_EPISODE, seed=args.seed,
             algo_force=MODEL_NAME.lower(), model_file=Path(model_path), stats_file=Path(stats_path))

if __name__ == "__main__":
    main()