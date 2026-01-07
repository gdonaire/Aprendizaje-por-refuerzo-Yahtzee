# Informe global de variantes: ppo_variants

## Resumen por variante (ordenado por puntuación final media)
| variant                  |   n_runs |   mean_reward_mean |   mean_reward_std |   mean_score_mean |   mean_score_std |   elapsed_mean_sec |
|:-------------------------|---------:|-------------------:|------------------:|------------------:|-----------------:|-------------------:|
| ppo_baseline             |        3 |            123.588 |           1.20356 |           125.767 |          1.27441 |         1846.83    |
| ppo_conservative_clip_kl |        3 |            121.957 |           4.57959 |           124.153 |          4.47957 |         1956.55    |
| ppo_highhorizon_lowvar   |        3 |            120.94  |           2.2278  |           123.133 |          2.27388 |         1401.09    |
| ppo_big_net 			   |        3 |            120.683 |           3.38231 |           122.807 |          3.29268 |         3574.28    |

## Notas
- Los valores son promedios sobre los seeds de cada variante.
- Observa std para seleccionar configuraciones estables.