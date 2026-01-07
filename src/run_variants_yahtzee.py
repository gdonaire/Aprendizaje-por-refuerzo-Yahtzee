# -*- coding: utf-8 -*-
"""
run_variants_yahtzee.py

Runner de variantes para el entrenamiento con VecNormalize en Yahtzee.
Prueba varias configuraciones (exploración, LR, target_update, batch, n_steps)
con múltiples seeds por configuración y genera informes agregados.

Uso:
    python run_variants_yahtzee.py \
        --model a2c \
        --variants a2c_baseline,a2c_highhorizon_lowvar,a2c_longrollout,a2c_fastexploit \
        --num_seeds 3 --base_seed 42 \
        --run_name a2c_variantes --n_envs 8 --eval_episodes 50
    
Requisitos: tener los distintos ficheros (python) de entrenamiento en el directorio actual y
las dependencias del entorno RL instaladas.
"""

import argparse
import sys
import time
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import math
import numpy as np
import pandas as pd

# -------------------------------
# Utilidades compartidas con el runner de seeds
# -------------------------------

def ensure_dirs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

def parse_metrics_file(path: Path) -> Dict[str, Any]:
    """
    path puede ser:
      - un directorio que contiene los *evaluation_metrics.txt
      - un fichero concreto de métricas
    """
    result = {"mean_reward": np.nan, "mean_score": np.nan}

    # Si nos pasan directamente el archivo, lo intentamos leer ya
    if path.is_file():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = ""
        # Regex correctas en Python:
        m_reward = re.search(r"Recompensa\s+media.*?:\s*([\-]?\d+(?:\.\d+)?)", content, flags=re.IGNORECASE)
        m_score  = re.search(r"Puntuaci[oó]n\s+final\s+promedio:\s*([\-]?\d+(?:\.\d+)?)", content, flags=re.IGNORECASE)
        if m_reward:
            result["mean_reward"] = float(m_reward.group(1))
        if m_score:
            result["mean_score"] = float(m_score.group(1))
        return result

    # Si es un directorio: buscamos ficheros candidatos
    candidates = [
        path / "ppo_vecnorm_final_evaluation_metrics.txt",
        path / "a2c_vecnorm_final_evaluation_metrics.txt",
        path / "mppo_vecnorm_final_evaluation_metrics.txt",
        path / "dqn_vecnorm_final_evaluation_metrics.txt",
        path / "qrdqn_vecnorm_final_evaluation_metrics.txt",
    ]
    # fallback: busca cualquier *_evaluation_metrics.txt
    if not any(p.exists() for p in candidates):
        candidates.extend(path.glob("*evaluation_metrics.txt"))

    for f in candidates:
        if not f.exists():
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m_reward = re.search(r"Recompensa\s+media.*?:\s*([\-]?\d+(?:\.\d+)?)", content, flags=re.IGNORECASE)
        m_score  = re.search(r"Puntuaci[oó]n\s+final\s+promedio:\s*([\-]?\d+(?:\.\d+)?)", content, flags=re.IGNORECASE)

        if m_reward:
            result["mean_reward"] = float(m_reward.group(1))
        if m_score:
            result["mean_score"] = float(m_score.group(1))
        if not (math.isnan(result["mean_reward"]) and math.isnan(result["mean_score"])):
            break

    return result


def add_flag(cmd, name, value):
    # No añadir si value es None, 'None', '' o -1 (sentinela)
    if value is None: 
        return cmd
    if isinstance(value, str) and value.strip().lower() == 'none':
        return cmd
    if value == '' or value == -1:
        return cmd
    cmd += [name, str(value)]
    return cmd

def run_one_seed(model:str,train_script: Path, seed: int, cfg: Dict[str, Any], base_out_dir: Path) -> Dict[str, Any]:
    variant = cfg["variant_name"]
    best_dir = base_out_dir / variant / f"seed_{seed}" / "best"    
    ensure_dirs(best_dir)
    log_path = best_dir.parent / "train.log"

    cmd = [
        sys.executable, str(train_script),
        "--seed", str(seed),
        "--n_envs", str(cfg.get("n_envs", 8)),
        "--timesteps", str(cfg.get("timesteps", 400_000)),
        "--timesteps2", str(cfg.get("timesteps2", 400_000)),
        "--timesteps3", str(cfg.get("timesteps3", 300_000)),
        "--eval_episodes", str(cfg.get("eval_episodes", 50)),
        "--best_dir", str(best_dir),
    ]

    if model in ["dqn", "qrdqn", "ppo", "a2c"]:
        cmd += [
            "--repair_actions_training", "true" if cfg.get("repair_actions_training", True) else "false",
        ]

    if model == "ppo":
        cmd = add_flag(cmd, "--lr", cfg.get("lr", None))
        cmd = add_flag(cmd, "--n_steps", cfg.get("n_steps", None))
        cmd = add_flag(cmd, "--batch_size", cfg.get("batch_size", None))
        cmd = add_flag(cmd, "--n_epochs", cfg.get("n_epochs", None))
        cmd = add_flag(cmd, "--gamma", cfg.get("gamma", None))
        cmd = add_flag(cmd, "--gae_lambda", cfg.get("gae_lambda", None))
        cmd = add_flag(cmd, "--ent_coef", cfg.get("ent_coef", None))
        cmd = add_flag(cmd, "--clip_range", cfg.get("clip_range", None)) 
        cmd = add_flag(cmd, "--vf_coef", cfg.get("vf_coef", None))
        cmd = add_flag(cmd, "--max_grad_norm", cfg.get("max_grad_norm", None))
        cmd = add_flag(cmd, "--target_kl", cfg.get("target_kl", None))
        cmd = add_flag(cmd, "--use_sde", cfg.get("use_sde", None))
        cmd = add_flag(cmd, "--sde_sample_freq", cfg.get("sde_sample_freq", None))
        cmd = add_flag(cmd, "--policy_arch", cfg.get("policy_arch", None))
           
    if model == "mppo":
        cmd = add_flag(cmd, "--lr", cfg.get("lr", None))
        cmd = add_flag(cmd, "--n_steps", cfg.get("n_steps", None))
        cmd = add_flag(cmd, "--batch_size", cfg.get("batch_size", None))
        cmd = add_flag(cmd, "--n_epochs", cfg.get("n_epochs", None))
        cmd = add_flag(cmd, "--gamma", cfg.get("gamma", None))
        cmd = add_flag(cmd, "--gae_lambda", cfg.get("gae_lambda", None))
        cmd = add_flag(cmd, "--ent_coef", cfg.get("ent_coef", None))
        cmd = add_flag(cmd, "--clip_range", cfg.get("clip_range", None)) 
        cmd = add_flag(cmd, "--vf_coef", cfg.get("vf_coef", None))
        cmd = add_flag(cmd, "--max_grad_norm", cfg.get("max_grad_norm", None))
        cmd = add_flag(cmd, "--policy_arch", cfg.get("policy_arch", None))

    if model == "a2c":
        cmd += [
            # Hiperparámetros
            "--lr", str(cfg.get("lr", 1e-4)),
            "--n_steps", str(cfg.get("n_steps", 5)),
            "--gamma", str(cfg.get("gamma", 0.99)),
            "--gae_lambda", str(cfg.get("gae_lambda", 1.00)),
            "--ent_coef", str(cfg.get("ent_coef", 0.00)),
        ]

    if model == "dqn" or model == "qrdqn":
        cmd += [
            # Hiperparámetros
            "--lr", str(cfg.get("lr", 1e-4)),
            "--buffer_size", str(cfg.get("buffer_size", 1_000_000)),
            "--learning_starts", str(cfg.get("learning_starts", 10_000)),
            "--batch_size", str(cfg.get("batch_size", 64)),
            "--gamma", str(cfg.get("gamma", 0.99)),
            "--n_steps", str(cfg.get("n_steps", 3)),
            "--train_freq", str(cfg.get("train_freq", 1)),
            "--gradient_steps", str(cfg.get("gradient_steps", 1)),
            "--target_update_interval", str(cfg.get("target_update_interval", 10_000)),
            "--exploration_fraction", str(cfg.get("exploration_fraction", 0.30)),
            "--exploration_initial_eps", str(cfg.get("exploration_initial_eps", 1.0)),
            "--exploration_final_eps", str(cfg.get("exploration_final_eps", 0.05)),
        ]
    
    if model == "qrdqn":
        cmd = add_flag(cmd, "--n_quantiles", cfg.get("n_quantiles", None))

    print(f"\n[RUN] {variant} | Seed={seed}\nCMD: {' '.join(cmd)}")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as lf:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            lf.write(line)
        process.wait()
    elapsed = time.time() - start

    if model == "dqn":
        metrics_file = best_dir / "dqn_vecnorm_final_evaluation_metrics.txt"
    elif model == "qrdqn":
        metrics_file = best_dir / "qrdqn_vecnorm_final_evaluation_metrics.txt"
    elif model == "ppo":
        metrics_file = best_dir / "ppo_vecnorm_final_evaluation_metrics.txt"
    elif model == "a2c":
        metrics_file = best_dir / "a2c_vecnorm_final_evaluation_metrics.txt"
    elif model == "mppo":
        metrics_file = best_dir / "mppo_vecnorm_final_evaluation_metrics.txt"
    else:
        raise ValueError(f"Modelo desconocido para métricas: {model}")

    metrics = parse_metrics_file(metrics_file)
    result = {
        "variant": variant,
        "seed": seed,
        "best_dir": str(best_dir),
        "elapsed_sec": elapsed,
        **metrics,
    }
    (best_dir.parent / "summary_seed.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate_variant(results_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(results_seed)
    agg = {
        "n_runs": int(len(df)),
        "mean_reward_mean": float(np.nanmean(df["mean_reward"])) if "mean_reward" in df else np.nan,
        "mean_reward_std": float(np.nanstd(df["mean_reward"], ddof=1)) if "mean_reward" in df else np.nan,
        "mean_score_mean": float(np.nanmean(df["mean_score"])) if "mean_score" in df else np.nan,
        "mean_score_std": float(np.nanstd(df["mean_score"], ddof=1)) if "mean_score" in df else np.nan,
        "elapsed_mean_sec": float(np.nanmean(df["elapsed_sec"])) if "elapsed_sec" in df else np.nan,
    }
    return agg

def save_variant_reports(run_root: Path, run_name: str, variant: str, results_seed: List[Dict[str, Any]]) -> None:
    out_dir = run_root / run_name / variant
    ensure_dirs(out_dir)
    df = pd.DataFrame(results_seed)
    df.to_csv(out_dir / "results_by_seed.csv", index=False)
    agg = aggregate_variant(results_seed)
    (out_dir / "aggregate_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    # Informe Markdown por variante
    md = [
        f"# Variante: {variant}",
        f"**Ejecuciones (seeds):** {agg['n_runs']}",
        "",
        f"- Recompensa media (promedio seeds): {agg['mean_reward_mean']:.4f}",
        f"- Recompensa std: {agg['mean_reward_std']:.4f}",
        f"- Puntuación final media: {agg['mean_score_mean']:.4f}",
        f"- Puntuación std: {agg['mean_score_std']:.4f}",
        f"- Tiempo medio por seed (s): {agg['elapsed_mean_sec']:.2f}",
        "",
        "## Resultados por seed",
        df.to_markdown(index=False)
    ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

def save_global_report(run_root: Path, run_name: str, all_results: List[Dict[str, Any]]) -> None:
    out_dir = run_root / run_name
    ensure_dirs(out_dir)

    # Agregación por variante
    df = pd.DataFrame(all_results)
    group = df.groupby("variant").agg(
        n_runs=("seed", "count"),
        mean_reward_mean=("mean_reward", "mean"),
        mean_reward_std=("mean_reward", "std"),
        mean_score_mean=("mean_score", "mean"),
        mean_score_std=("mean_score", "std"),
        elapsed_mean_sec=("elapsed_sec", "mean")
    ).reset_index()
    group.to_csv(out_dir / f"{run_name}_summary.csv", index=False)

    # Ranking por puntuación media
    group_sorted = group.sort_values(by="mean_score_mean", ascending=False)
    md = [
        f"# Informe global de variantes: {run_name}",
        "",
        "## Resumen por variante (ordenado por puntuación final media)",
        group_sorted.to_markdown(index=False),
        "",
        "## Notas",
        "- Los valores son promedios sobre los seeds de cada variante.",
        "- Recomiendo contrastar con la varianza (std) para elegir configuración estable.",
    ]
    (out_dir / f"{run_name}_report.md").write_text("\n".join(md), encoding="utf-8")

# -------------------------------
# Definición de variantes
# -------------------------------

def predefined_variants(base_args: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Devuelve un diccionario de variantes con hiperparámetros modificados."""
    v = {}
    # Perfiles DQN
    v["dqn_baseline"] = {**base_args, "variant_name": "dqn_baseline",}
    v["dqn_batch_big"] = {**base_args, "variant_name": "dqn_batch_big",
        "batch_size": 128,
    }
    v["dqn_target_slow_grad_more"] = {**base_args, "variant_name": "dqn_target_slow_grad_more",
        "gradient_steps": 2,
        "target_update_interval": 30_000,
    }
    v["dqn_nstep5_epsfloor010"] = {**base_args, "variant_name": "dqn_nstep5_epsfloor010",
        "n_steps": 5,
        "exploration_final_eps": 0.10,
    }
    # Perfiles QR-DQN
    v["qrdqn_baseline"] = {**base_args, "variant_name": "qrdqn_baseline" }
    v["qrdqn_target_slow_grad_more"] = {**base_args, "variant_name": "qrdqn_target_slow_grad_more",
        "exploration_fraction": 0.45,
        "target_update_interval": 30_000,
        "gradient_steps": 2
    }
    v["qrdqn_nq25_batch128"] = {**base_args, "variant_name": "qrdqn_nq25_batch128",
        "n_quantiles": 25,
        "batch_size": 128
    }
    v["qrdqn_nq101_batch64"] = {**base_args, "variant_name": "qrdqn_nq101_batch64", 
        "n_quantiles": 101,
        "batch_size": 64
    }
    # Perfiles A2C
    v["a2c_baseline"] = {**base_args, "variant_name": "a2c_baseline" }
    v["a2c_highhorizon_lowvar"] = {**base_args, "variant_name": "a2c_highhorizon_lowvar", 
        "n_steps": 128, "gae_lambda": 0.97 }
    v["a2c_longrollout"] = {**base_args, "variant_name": "a2c_longrollout", 
        "n_steps": 256 }
    v["a2c_fastexploit"] = {**base_args, "variant_name": "a2c_fastexploit", 
        "n_steps": 128, "ent_coef": 0.01 }
    # Perfiles PPO
    v["ppo_baseline"] = {**base_args, "variant_name": "ppo_baseline" }
    v["ppo_highhorizon_lowvar"] = {**base_args, "variant_name": "ppo_highhorizon_lowvar",
        "n_steps": 4096, "batch_size": 128, "gamma": 0.995, "gae_lambda": 0.97}
    v["ppo_conservative_clip_kl"] = {**base_args, "variant_name": "ppo_conservative_clip_kl", 
        "n_epochs": 8, "clip_range": 0.1, "target_kl": 0.02 }
    v["ppo_big_net"] = {**base_args, "variant_name": "ppo_big_net", 
        "max_grad_norm": 1.0, "policy_arch": '{"pi":[256, 256,256], "vf":[256, 256,256]}' }
    # Perfiles MPPO
    v["mppo_baseline"] = {**base_args, "variant_name": "mppo_baseline" }
    v["mppo_highhorizon_lowvar"] = {**base_args, "variant_name": "mppo_highhorizon_lowvar", 
        "n_steps": 4096, "batch_size": 128, "gamma": 0.995, "gae_lambda": 0.97 }
    v["mppo_conservative_clip_kl"] = {**base_args, "variant_name": "mppo_conservative_clip_kl", 
        "n_epochs": 8, "clip_range": 0.1, "target_kl": 0.02 }
    v["mppo_big_net"] = {**base_args, "variant_name": "mppo_big_net", 
        "max_grad_norm": 1.0, "policy_arch": '{"pi":[256, 256,256], "vf":[256, 256,256]}' }
    
    return v

# -------------------------------
# Main CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Runner de variantes (múltiples variantes + seeds) para Yahtzee con VecNormalize")
    parser.add_argument("--variants", type=str, default="", help="Lista de variantes a ejecutar (nombres predefinidos separados por comas)")
    parser.add_argument("--num_seeds", type=int, default=3, help="Número de seeds por variante")
    parser.add_argument("--base_seed", type=int, default=42, help="Seed base para generar la lista de seeds por variante")
    parser.add_argument("--run_name", type=str, default="yahtzee_variants", help="Nombre del experimento para carpetas de salida")
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--eval_episodes", type=int, default=50)

    # Timesteps por fase (puedes reducir para pruebas)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--timesteps2", type=int, default=400_000)
    parser.add_argument("--timesteps3", type=int, default=300_000)

    # Frecuencias y rutas
    parser.add_argument("--eval_freq_env_steps", type=int, default=20_000)
    parser.add_argument("--out_root", type=str, default="./runs", help="Raíz para informes agregados")

    # Defaults de hiperparámetros (se pueden sobreescribir en variantes)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--target_update_interval", type=int, default=10_000)
    parser.add_argument("--exploration_fraction", type=float, default=0.30)
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0)
    parser.add_argument("--exploration_final_eps", type=float, default=0.05)
    parser.add_argument("--repair_actions_training", type=lambda s: s.lower() in ["true", "1", "yes"], default=True)
    # modelo a usar (DQN por defecto)
    parser.add_argument("--model", type=str, default="dqn", choices=["dqn", "qrdqn", "a2c", "ppo", "mppo"], help="Modelo RL a usar (DQN o QRDQN)")
    # Soporte QR-DQN (se pasa tal cual al script)
    parser.add_argument("--n_quantiles", type=int, default=None, help="Sobrescribe n_quantiles para QR-DQN (opcional)")

    args = parser.parse_args()

    # modelo de entrenamiento
    switcher = {
        "dqn": "train_yahtzee_dqn_vecnorm.py",
        "qrdqn": "train_yahtzee_qrdqn_vecnorm.py",
        "a2c": "train_yahtzee_a2c_vecnorm.py",
        "ppo": "train_yahtzee_ppo_vecnorm.py",
        "mppo": "train_yahtzee_mppo_vecnorm.py",
    }

    if args.model not in switcher:
        print(f"ERROR: Modelo desconocido '{args.model}'. Opciones válidas: {list(switcher.keys())}")
        sys.exit(1)

    train_script = Path(switcher[args.model])
    # verifica que el script de entrenamiento existe
    if not train_script.exists():
        print(f"ERROR: No se encuentra '{train_script}' en el directorio actual.")
        sys.exit(1)    

    print(f"\n=== Runner de variantes Yahtzee {args.model.upper()} con VecNormalize ===")
    print(f"Usando script de entrenamiento: {train_script}")
    # Mostrar argumentos
    print("Argumentos de entrenamiento (Runner):")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=============================================\n")
    # Config base para variantes
    base_cfg = {
        "n_envs": args.n_envs,
        "timesteps": args.timesteps,
        "timesteps2": args.timesteps2,
        "timesteps3": args.timesteps3,
        "eval_episodes": args.eval_episodes,
        "lr": args.lr,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "n_steps": args.n_steps,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": args.exploration_fraction,
        "exploration_initial_eps": args.exploration_initial_eps,
        "exploration_final_eps": args.exploration_final_eps,
        "repair_actions_training": args.repair_actions_training,
        "n_quantiles": args.n_quantiles if args.model == "qrdqn" else None,
    }

    variants_all = predefined_variants(base_cfg)
    names_requested = [n.strip() for n in args.variants.split(',') if n.strip()]
    variants_to_run = {name: variants_all[name] for name in names_requested if name in variants_all}
    if not variants_to_run:
        print("No hay variantes válidas para ejecutar. Revisa --variants.")
        sys.exit(1)

    # Seeds por variante
    seeds = [args.base_seed * (i + 1) for i in range(args.num_seeds)]
    print(f"Variantes a ejecutar: {list(variants_to_run.keys())}")
    print(f"Seeds por variante: {seeds}")

    base_out_dir = Path("./best") / args.run_name
    all_results: List[Dict[str, Any]] = []

    # Ejecuta cada variante y cada seed
    for vname, cfg in variants_to_run.items():
        cfg = {**cfg, "variant_name": vname}
        variant_results: List[Dict[str, Any]] = []
        for sd in seeds:
            try:
                res = run_one_seed(args.model, train_script, sd, cfg, base_out_dir)
                variant_results.append(res)
                all_results.append(res)
            except Exception as e:
                print(f"[WARN] Variante {vname}, seed {sd} falló: {e}")
                variant_results.append({"variant": vname, "seed": sd, "mean_reward": np.nan, "mean_score": np.nan, "elapsed_sec": np.nan, "best_dir": ""})
                all_results.append(variant_results[-1])
                continue
        # Informes por variante
        save_variant_reports(Path(args.out_root), args.run_name, vname, variant_results)
    
    # Informe global
    save_global_report(Path(args.out_root), args.run_name, all_results)
    print("\nInformes generados en:")
    print(f"  - Por variante: runs/{args.run_name}/<variante>/(CSV/JSON/MD)")
    print(f"  - Global: runs/{args.run_name}/{args.run_name}_report.md y {args.run_name}_summary.csv")


if __name__ == "__main__":
    main()
