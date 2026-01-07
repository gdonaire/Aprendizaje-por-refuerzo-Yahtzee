# Informe global de variantes: dqn_variants

## Resumen por variante (ordenado por puntuación final media)
| variant                   |   n_runs |   mean_reward_mean |   mean_reward_std |   mean_score_mean |   mean_score_std |   elapsed_mean_sec |
|:--------------------------|---------:|-------------------:|------------------:|------------------:|-----------------:|-------------------:|
| dqn_batch_big             |        3 |            150.335 |          2.74974  |           152.527 |         2.66416  |            1933.75 |
| dqn_baseline              |        3 |            149.91  |          0.321081 |           152.06  |         0.302655 |            1849.46 |
| dqn_target_slow_grad_more |        3 |            148.889 |          4.73106  |           151.12  |         4.66952  |            2863.6  |
| dqn_nstep5_epsfloor010    |        3 |            144.108 |          1.51638  |           146.34  |         1.59737  |            1750.39 |

## Notas
- Los valores son promedios sobre los seeds de cada variante.
- Recomiendo contrastar con la varianza (std) para elegir configuración estable.