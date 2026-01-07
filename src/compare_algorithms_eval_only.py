# -*- coding: utf-8 -*-
"""
compare_algorithms_eval_only.py

Recoree los informes de variantes de cada algoritmo, escoge la mejor variante por
`mean_score_mean`, localiza el mejor seed (por `mean_score`) y **evalúa** los
modelos ya entrenados **sin re-entrenar**, cargando modelo + VecNormalize stats
con la función `evaluate(...)` de `final_evaluate_rl.py`.

Salida: genera un resumen comparativo CSV/Markdown en `runs/<run_name_global>/`.

Requisitos:
- Archivos de informe Markdown (por ejemplo):
    a2c_variants_report.md, ppo_variants_report.md, dqn_variants_report.md,
    mppo_variants_report.md (opcional), qrdqn_variants_report.md (opcional)
- Estructura de salidas del runner:
    runs/<run_name>/<variant>/results_by_seed.csv
- Estructura de mejores modelos por seed (generada por los scripts de train):
    best/<run_name>/<variant>/seed_<seed>/best/

Uso típico:
    python compare_algorithms_eval_only.py \
      --a2c_report a2c_variants_report.md \
      --ppo_report ppo_variants_report.md \
      --dqn_report dqn_variants_report.md \
      --mppo_report mppo_variants_report.md \
      --qrdqn_report qrdqn_variants_report.md \
      --global_run_name final_compare_evalonly \
      --episodes 200 --max_steps 1000 --seed 42

"""
import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _load_rewards_std_from_bestdir(best_dir: Path, algo: str) -> float:
    """Lee el CSV de recompensas por episodio guardado por evaluate() y devuelve std.
    Si no existe el fichero, devuelve NaN.
    """
    import numpy as _np
    import pandas as _pd
    fname = f"{algo}_final_episode_rewards.csv"
    f = best_dir / fname
    if not f.exists():
        return float('nan')
    try:
        data = _np.loadtxt(f, delimiter=',')
        if data.ndim == 0:
            return 0.0
        return float(_np.nanstd(data, ddof=1))
    except Exception:
        try:
            s = _pd.read_csv(f, header=None).iloc[:,0].astype(float)
            return float(s.std())
        except Exception:
            return float('nan')


def _load_scores_std_from_bestdir(best_dir: Path, algo: str) -> float:
    """Lee el CSV de puntuaciones por episodio guardado por evaluate() y devuelve std.
    Si no existe el fichero, devuelve NaN.
    """
    import numpy as _np
    import pandas as _pd
    fname = f"{algo}_final_episode_scores.csv"
    f = best_dir / fname
    if not f.exists():
        return float('nan')
    try:
        # Es un CSV de una sola columna sin cabecera: usar numpy o pandas read_csv header=None
        data = _np.loadtxt(f, delimiter=',')
        if data.ndim == 0:  # un único valor
            return 0.0
        return float(_np.nanstd(data, ddof=1))
    except Exception:
        try:
            s = _pd.read_csv(f, header=None).iloc[:,0].astype(float)
            return float(s.std())
        except Exception:
            return float('nan')

def _save_barplot(df, out_png: Path, metric_col: str = 'mean_score', err_col: str = 'score_std') -> None:
    """Crea y guarda un barplot del metric_col por algoritmo con barras de error opcionales."""
    if df.empty or metric_col not in df.columns:
        return
    # Ordenar por valor descendente
    d = df.sort_values(by=metric_col, ascending=False).copy()
    algos = d.index.tolist()
    vals = d[metric_col].values
    errs = d[err_col].values if err_col in d.columns else None

    plt.figure(figsize=(10,6))
    bars = plt.bar(range(len(algos)), vals, yerr=errs if err_col in d.columns else None,
                   capsize=6, color='#3b82f6', alpha=0.8, edgecolor='black')
    plt.xticks(range(len(algos)), [a.upper() for a in algos])
    plt.ylabel('Puntuación media (sin normalizar)')
    plt.title('Comparativa final (eval-only): puntuación media por algoritmo')
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Anotar valores sobre las barras
    for rect, v in zip(bars, vals):
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, h + max(vals)*0.01,
                 f"{v:.2f}", ha='center', va='bottom', fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# Importa evaluate y el wrapper desde final_evaluate_rl.py (mismo directorio)
try:
    from final_evaluate_rl import evaluate  # type: ignore
except Exception as e:
    print("ERROR: no se puede importar 'evaluate' desde final_evaluate_rl.py.\n", e)
    sys.exit(1)

#VARIANT_REGEX = re.compile(r"^(a2c|ppo|mppo|dqn|qrdqn)_[A-Za-z0-9_]+\s*$")

# Detecta nombres de variante del estilo: a2c_baseline, ppo_longrollout, etc.
# Sin saltos de línea en la alternancia:
VARIANT_REGEX = re.compile(r"^(?:a2c|ppo|mppo|dqn|qrdqn)_[A-Za-z0-9_]+\s*$")



def parse_run_name_from_md(md_text: str) -> Optional[str]:
    """Intenta extraer el run_name del título del informe MD.
    Ejemplos de primera línea:
      '# Informe global de variantes: dqn_variants'
      '# Informe global de abla…: ppo_ablation'
    """
    for line in md_text.splitlines():
        line = line.strip()
        if line.startswith('#') and ':' in line:
            try:
                rn = line.split(':', 1)[1].strip()
                rn = rn.strip('# ').strip()
                if rn:
                    return rn
            except Exception:
                continue
    return None


def parse_best_variant_from_md(md_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Parsea el MD y devuelve (best_variant_name, run_name).
    Criterio: mayor 'mean_score_mean'.
    """
    if not md_path.exists():
        return None, None
    txt = md_path.read_text(encoding='utf-8', errors='ignore')
    run_name = parse_run_name_from_md(txt)
    #print(f"[DEBUG] parse_best_variant_from_md: run_name='{run_name}' from {md_path}")

    # Intenta leer la tabla con pandas si es viable
    # Si falla, usa un parser manual de respaldo.
    best_variant = None
    best_score = -np.inf

    # --- 1) Parser para tablas con '|' (formato más común en tus informes) ---
    # Esperamos cabecera y filas tipo:
    # | variant | n_runs | mean_reward_mean | mean_reward_std | mean_score_mean | mean_score_std | elapsed_mean_sec |
    for line in txt.splitlines():
        s = line.strip()
        #print(f"[DEBUG] parse_best_variant_from_md: line='{s}'")
        if not s or not s.startswith("|"):
            continue
        # Ignorar cabeceras y separadores de Markdown
        if s.startswith("|:") or set(s.replace("|", "").strip()) <= set("-:"):
            continue
        parts = [p.strip() for p in s.split("|")]
        # parts típicamente: ["", "variant", "n_runs", "mean_reward_mean", ... , ""]
        if len(parts) < 8:
            continue
        variant = parts[1]
        # La columna 5 suele ser mean_score_mean según tu MD
        try:
            mean_score_mean = float(parts[5])
        except Exception:
            # Si no se puede convertir, intenta detectar por nombre
            try:
                # Buscar la columna que contenga 'mean_score_mean' en la cabecera
                pass
            except Exception:
                continue
        if VARIANT_REGEX.match(variant):
            if mean_score_mean > best_score:
                best_score = mean_score_mean
                best_variant = variant

    # --- 2) Respaldo: parser manual anterior (por si el informe viene en otro formato) ---
    if best_variant is None:
        current = None
        nums = []
        def flush():
            nonlocal current, nums, best_variant, best_score
            if current is not None and len(nums) >= 4:
                try:
                    mean_score_mean = float(nums[3])
                    if mean_score_mean > best_score:
                        best_score = mean_score_mean
                        best_variant = current
                except Exception:
                    pass
            current, nums = None, []
        for line in txt.splitlines():
            s = line.strip()
            if not s:
                flush()
                continue
            if VARIANT_REGEX.match(s):
                flush()
                current = s
                nums = []
                continue
            try:
                val = s.rstrip(',')
                if val.lower() in ('nan', 'none', 'null'):
                    nums.append(np.nan)
                else:
                    nums.append(float(val))
            except Exception:
                continue
        flush()

    #print(f"[DEBUG] parse_best_variant_from_md: best_variant='{best_variant}' with mean_score={best_score}")
    return best_variant, run_name


def find_best_seed_bestdir(run_root: Path, run_name: str, variant: str) -> Optional[Path]:
    """Busca runs/<run_name>/<variant>/results_by_seed.csv, elige la fila con mayor
    'mean_score' y devuelve Path(best_dir) de esa fila.
    """
    csv_path = run_root / run_name / variant / 'results_by_seed.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # Se espera columna 'mean_score' y 'best_dir'
    cols = [c.lower() for c in df.columns]
    # Normaliza nombres básicos
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == 'mean_score':
            rename[c] = 'mean_score'
        elif cl == 'best_dir':
            rename[c] = 'best_dir'
    if rename:
        df = df.rename(columns=rename)
    if 'mean_score' not in df.columns or 'best_dir' not in df.columns:
        return None
    # Elegir la fila de mayor mean_score (ignorando NaN)
    df_valid = df.dropna(subset=['mean_score'])
    if df_valid.empty:
        return None
    row = df_valid.sort_values(by='mean_score', ascending=False).iloc[0]
    best_dir = Path(str(row['best_dir']))
    return best_dir


ALG_ALIAS = {
    'a2c': 'a2c',
    'ppo': 'ppo',
    'mppo': 'mppo',
    'dqn': 'dqn',
    'qrdqn': 'qrdqn',
}


def algo_from_variant(variant: str) -> Optional[str]:
    for k in ALG_ALIAS.keys():
        if variant.lower().startswith(f"{k}_"):
            return k
    return None


def main():
    ap = argparse.ArgumentParser(description='Evalúa mejores variantes sin re-entrenar (carga modelo + VecNormalize y compara algoritmos).')
    ap.add_argument('--a2c_report', type=str, default='a2c_variants_report.md')
    ap.add_argument('--ppo_report', type=str, default='ppo_variants_report.md')
    ap.add_argument('--dqn_report', type=str, default='dqn_variants_report.md')
    ap.add_argument('--mppo_report', type=str, default='', help='Informe MaskablePPO (opcional)')
    ap.add_argument('--qrdqn_report', type=str, default='', help='Informe QRDQN (opcional)')

    ap.add_argument('--run_root', type=str, default='runs', help='Raíz de informes del runner (CSV/JSON/MD)')
    ap.add_argument('--global_run_name', type=str, default='final_compare_evalonly', help='Nombre para la carpeta de salida comparativa')

    ap.add_argument('--episodes', type=int, default=200, help='Episodios de evaluación por algoritmo')
    ap.add_argument('--max_steps', type=int, default=1000, help='Límite de steps por episodio')
    ap.add_argument('--seed', type=int, default=42, help='Semilla para los entornos de evaluación')

    args = ap.parse_args()

    reports = {
        'a2c': Path(args.a2c_report) if args.a2c_report else None,
        'ppo': Path(args.ppo_report) if args.ppo_report else None,
        'dqn': Path(args.dqn_report) if args.dqn_report else None,
        'mppo': Path(args.mppo_report) if args.mppo_report else None,
        'qrdqn': Path(args.qrdqn_report) if args.qrdqn_report else None,
    }

    run_root = Path(args.run_root)
    out_dir = run_root / args.global_run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, object]] = {}

    # 1) Para cada reporte, elegir mejor variante y localizar best_dir del mejor seed
    for alg, md_path in reports.items():
        if md_path is None or not md_path.exists():
            print(f"[WARN] No hay informe para {alg}. Se ignora.")
            continue
        best_variant, run_name = parse_best_variant_from_md(md_path)
        if not best_variant:
            print(f"[WARN] No se pudo detectar mejor variante en {md_path}.")
            continue
        # Si el prefijo del nombre de variante no coincide con alg (por seguridad), ajusta alg_real
        alg_real = algo_from_variant(best_variant) or alg
        if not run_name:
            print(f"[INFO] No pude extraer run_name de {md_path}; intenta deducirlo manualmente o pasa informes correctos.")
            # Se podría poner un default, pero mejor avisar
        else:
            print(f"[INFO] {alg.upper()}: mejor variante detectada: {best_variant} (run_name='{run_name}')")
        # Localiza best_dir desde results_by_seed.csv
        best_dir = find_best_seed_bestdir(run_root, run_name or '', best_variant)
        if best_dir is None or not best_dir.exists():
            print(f"[WARN] No se encontró best_dir para {best_variant}. Buscando en ./best ...")
            # Fallback: buscar en ./best/<run_name>/<variant>/seed_*/best
            pattern = Path('best') / (run_name or '*') / best_variant / 'seed_*' / 'best'
            matches = list(pattern.parent.parent.parent.glob(str(pattern.name))) if pattern.exists() else list(Path('best').glob(f"{run_name or '*'}/*/seed_*/best"))
            best_dir = None
            for p in matches:
                if best_variant in str(p):
                    best_dir = p
                    break
            if best_dir is None:
                print(f"[ERROR] No se pudo localizar best_dir para {best_variant}. Se omite {alg}.")
                continue
        print(f"[OK] {alg_real.upper()} -> variante='{best_variant}', best_dir='{best_dir}'")

        # 2) Evaluar cargando modelo+stats (sin re-entrenar)
        try:
            mean_reward, mean_score = evaluate(
                best_dir=best_dir,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                algo_force=alg_real,
                model_file=None,
                stats_file=None,
            )
        except Exception as e:
            print(f"[ERROR] Falló evaluate() para {alg_real} en {best_dir}: {e}")
            continue

        results[alg_real] = {
            'variant': best_variant,
            'best_dir': str(best_dir),
            'episodes': args.episodes,
            'mean_reward': float(mean_reward),
            'mean_score': float(mean_score),
        }

    # 3) Guardar resumen comparativo
    if not results:
        print("[ERROR] No hay resultados que guardar.")
        return
    df = pd.DataFrame.from_dict(results, orient='index')
    csv_path = out_dir / 'algorithms_evalonly_summary.csv'
    md_path = out_dir / 'algorithms_evalonly_summary.md'
    df.to_csv(csv_path)
    md_lines = ["# Comparativa final (eval-only)", "", df.reset_index().rename(columns={'index': 'algorithm'}).to_markdown(index=False)]
    md_path.write_text("\n".join(md_lines), encoding='utf-8')

    print("\nResumen comparativo guardado en:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")

if __name__ == '__main__':
    main()
