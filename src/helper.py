
import os
import numpy as np
import matplotlib
# En entornos headless (Docker/CI/WSL) es recomendable usar Agg
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Gráfica de evolución de recompensas por episodio
# -----------------------------
def plot_reward_evolution(
    rewards,
    best_dir,
    algo_name="qrdqn",
    n_envs=8,
    window_short=10,
    window_long=200,
    filename=None,
    title_prefix=None,
    y_label="Recompensa",
    alpha_series=0.6,
) -> str:
    """
    Genera y guarda la gráfica de evolución de recompensas por episodio.

    Parámetros
    ----------
    rewards : array-like (list o np.ndarray)
        Recompensas por episodio (idealmente sin normalizar si es lo que quieres reflejar).
    best_dir : str o Path
        Directorio donde se guardará la imagen.
    algo_name : str, opcional
        Nombre del algoritmo a mostrar y para el nombre del fichero (por defecto: 'qrdqn').
    n_envs : int, opcional
        Número de entornos paralelos, solo para mostrar en el título.
    window_short : int, opcional
        Tamaño de la media móvil corta (por defecto: 10).
    window_long : int, opcional
        Tamaño de la media móvil larga (por defecto: 200).
    filename : str, opcional
        Nombre del archivo a guardar. Si no se indica, se usa '{algo_name}_vecnorm_reward_evolution.png'.
    title_prefix : str, opcional
        Prefijo del título. Si no se indica, se usa 'Evolucion del aprendizaje {ALGO} con VecNormalize'.
    y_label : str, opcional
        Etiqueta del eje Y (por defecto: 'Recompensa').
    alpha_series : float, opcional
        Transparencia de la serie principal de recompensas (por defecto: 0.6).

    Retorna
    -------
    str
        Ruta del archivo PNG guardado.
    """
    # Normaliza entrada a np.array float32
    rewards = np.array(rewards, dtype=np.float32)

    # Configurar nombre de archivo
    if filename is None:
        filename = f"{algo_name}_vecnorm_reward_evolution.png"
    plot_path = os.path.join(str(best_dir), filename)

    # Preparar figura
    plt.figure(figsize=(10, 6))

    # Serie principal
    if rewards.size > 0:
        plt.plot(rewards, label="Recompensa por episodio", alpha=alpha_series)
    else:
        # Si no hay datos, dejamos un mensaje en la figura para facilitar el debug
        ax = plt.gca()
        ax.text(
            0.5, 0.5,
            "Sin datos de episodios (lista vacía)",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=11
        )

    # Media móvil corta
    if rewards.size >= window_short and window_short > 1:
        moving_avg = np.convolve(rewards, np.ones(window_short)/window_short, mode="valid")
        plt.plot(
            range(window_short - 1, len(rewards)),
            moving_avg,
            label=f"Media movil ({window_short})",
            color="red",
            linewidth=2
        )

    # Media móvil larga
    if rewards.size >= window_long and window_long > 1:
        moving_avg_long = np.convolve(rewards, np.ones(window_long)/window_long, mode="valid")
        plt.plot(
            range(window_long - 1, len(rewards)),
            moving_avg_long,
            label=f"Media movil ({window_long})",
            color="green",
            linewidth=2
        )

    # Título y ejes
    if title_prefix is None:
        title_prefix = f"Evolucion del aprendizaje {algo_name.upper()} con VecNormalize"
    plt.title(f"{title_prefix} ({n_envs} entornos)")
    plt.xlabel("Episodio")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    # Guardar y cerrar
    os.makedirs(best_dir, exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"Grafica guardada en {plot_path}")
    return plot_path