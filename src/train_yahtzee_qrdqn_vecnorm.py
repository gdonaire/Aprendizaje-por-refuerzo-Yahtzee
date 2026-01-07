import os
import logging
import argparse

import yahtzee_env
from pathlib import Path
from final_evaluate_rl import evaluate, ActionMaskRepairVecWrapper, MAX_STEPS_PER_EPISODE

from sb3_contrib import QRDQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from callbacks import EpisodeRewardCallback , RewardLoggerCallback, RawRewardTrackerVecWrapper
from helper import plot_reward_evolution

MODEL_NAME = "qrdqn"

# -----------------------------
# Construccion del modelo QRDQN
# -----------------------------
def make_qrdqn(vec_env, args, *, n_steps=None, exploration_fraction=None, exploration_initial_eps=None):
    policy_kwargs = dict(net_arch=[256,256], n_quantiles=args.n_quantiles)

    model = QRDQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        # Off-policy scheduling
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        # n-step returns (SB3>=2.7.0)
        n_steps=args.n_steps if n_steps is None else n_steps,
        # Exploracion epsilon-greedy
        exploration_fraction=args.exploration_fraction if exploration_fraction is None else exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps if exploration_initial_eps is not None else args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        optimize_memory_usage=False, # <-- desactivar para compatibilidad con timeouts
        device="auto",
    )
    return model

# -----------------------------
# Argumentos de linea de comandos
# -----------------------------
def build_argparser():
    parser = argparse.ArgumentParser(description=f"Entrenar {MODEL_NAME.upper()} con VecNormalize (Yahtzee-v2)")
    # Fases
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Timesteps Fase 1")
    parser.add_argument("--timesteps2", type=int, default=500_000, help="Timesteps Fase 2")
    parser.add_argument("--timesteps3", type=int, default=500_000, help="Timesteps Fase 3")
    # Entornos / reproducibilidad
    parser.add_argument("--n_envs", type=int, default=8, help="Numero de entornos paralelos")
    parser.add_argument("--seed", type=int, default=42, help="Semilla")
    # Hiperparametros DQN
    parser.add_argument("--lr", type=float, default=1e-4, help="learning_rate")
    parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Tamaño del buffer de replay")
    parser.add_argument("--learning_starts", type=int, default=10000, help="Timesteps antes de empezar a aprender")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamaño de batch")
    parser.add_argument("--gamma", type=float, default=0.99, help="Factor de descuento")
    # n-step returns (SB3 >= 2.7.0)
    parser.add_argument("--n_steps", type=int, default=3, help=f"n-step returns para {MODEL_NAME.upper()} (SB3>=2.7.0)")
    parser.add_argument("--n_quantiles", type=int, default=51, help=f"Numero de cuantiles para {MODEL_NAME.upper()}")
    # Frecuencias / target
    parser.add_argument("--train_freq", type=int, default=1, help="Frecuencia de entrenamiento (steps)")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Actualizaciones por entrenamiento")
    parser.add_argument("--target_update_interval", type=int, default=10_000, help="Intervalo actualizacion target")
    # Exploracion
    parser.add_argument("--exploration_fraction", type=float, default=0.30, help="Fraccion de decay de epsilon")
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0, help="Epsilon inicial")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05, help="Epsilon minimo")
    # Reparacion de acciones durante training
    parser.add_argument("--repair_actions_training", type=lambda s: s.lower() in ["true", "1", "yes"], default=True,
                        help="Aplica reparacion de acciones invalidas durante el entrenamiento (por defecto: True)")
    # Evaluacion
    parser.add_argument("--eval_episodes", type=int, default=50, help="Episodios por evaluacion")
    parser.add_argument("--best_dir", type=str, default="./best", help="Carpeta para guardar el mejor modelo")
    # opcional: un suavizado largo para la grafica
    parser.add_argument("--plot_ma_long", type=int, default=200, help="Ventana larga para media movil en la grafica")

    return parser


def main():
    args = build_argparser().parse_args()

    # Mostrar argumentos
    print(f"Argumentos de entrenamiento {MODEL_NAME.upper()}:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    logger = logging.getLogger(f"train_{MODEL_NAME}_vecnorm")

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
    model = make_qrdqn(vec_env, args)

    # --- Callbacks ---
    
    # --- EpisodeRewardCallback (no depende de info["episode"]) ---
    train_cb = EpisodeRewardCallback(n_envs=args.n_envs)

    try:
        logger.info(f"Entrenando {MODEL_NAME.upper()} Fase 1 por {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps, callback=train_cb)
    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido manualmente en Fase 1.")

    # -----------------------------
    # Fase 2 (ajuste fino)
    # -----------------------------

    eps_prev = float(model.exploration_rate)
    model2 = make_qrdqn(
        vec_env, args,
        exploration_fraction=args.exploration_fraction / 2,
        exploration_initial_eps=eps_prev
    )
    model2.set_parameters(model.get_parameters())
    model2.replay_buffer = model.replay_buffer
    model2.learning_starts = 0                                      # warm-up minimo
    model2.target_update_interval = args.target_update_interval * 2  # target mas lento
    model2.gradient_steps = args.gradient_steps * 2                  # mas updates por step de entorno
    try:
        logger.info(f"Entrenando {MODEL_NAME.upper()} Fase 2 por {args.timesteps2} timesteps...")
        model2.learn(total_timesteps=args.timesteps2, callback=train_cb)
    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido manualmente en Fase 2.")
    model = model2

    # -----------------------------
    # Fase 3: mas piso de exploracion y target updates mas lentos
    # -----------------------------
    eps_prev = float(model.exploration_rate)
    model3 = make_qrdqn(
        vec_env, args,
        exploration_fraction=args.exploration_fraction / 4,
        exploration_initial_eps=eps_prev
    )   
    model3.set_parameters(model.get_parameters())
    model3.replay_buffer = model.replay_buffer
    model3.learning_starts = 0                                       # warm-up minimo
    # model3.lr_schedule = lambda _: args.lr * 0.5                     # 5e-5 efectivo
    model3.target_update_interval = args.target_update_interval * 3  # target mas lento
    model3.gradient_steps = args.gradient_steps * 2                  # mas updates por step de entorno
    model3.exploration_final_eps = max(model.exploration_final_eps, 0.10)   # mas exploracion residual
    try:
        logger.info(f"Entrenando {MODEL_NAME.upper()} Fase 3 por {args.timesteps3} timesteps...")
        model3.learn(total_timesteps=args.timesteps3, callback=train_cb)
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