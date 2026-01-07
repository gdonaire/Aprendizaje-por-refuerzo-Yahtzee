# Informe global de variantes: mppo_variants

## Resumen por variante (ordenado por puntuación final media)
| variant                   |   n_runs |   mean_reward_mean |   mean_reward_std |   mean_score_mean |   mean_score_std |   elapsed_mean_sec |
|:--------------------------|---------:|-------------------:|------------------:|------------------:|-----------------:|-------------------:|
| mppo_baseline             |        3 |            138.618 |           5.72163 |           140.72  |          5.80496 |            2168.38 |
| mppo_conservative_clip_kl |        3 |            132.577 |           2.58841 |           134.72  |          2.65774 |            1650.62 |
| mppo_highhorizon_lowvar   |        3 |            131.605 |           6.04964 |           133.747 |          6.04921 |            1521.69 |
| mppo_big_net              |        3 |            124.807 |              5.52 |               127 |          5.57606 |             4579.5 |

## Notas
- Los valores son promedios sobre los seeds de cada variante.
- Observa std para seleccionar configuraciones estables.