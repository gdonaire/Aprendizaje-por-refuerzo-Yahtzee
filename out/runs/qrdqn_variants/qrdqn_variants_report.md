# Informe global de variantes: qrdqn_variants

## Resumen por variante (ordenado por puntuación final media)
| variant                     |   n_runs |   mean_reward_mean |   mean_reward_std |   mean_score_mean |   mean_score_std |   elapsed_mean_sec |
|:----------------------------|---------:|-------------------:|------------------:|------------------:|-----------------:|-------------------:|
| qrdqn_nq25_batch128         |        3 |            153.453 |           3.81374 |           155.607 |          3.71022 |            3592.42 |
| qrdqn_target_slow_grad_more |        3 |            151.495 |           4.50794 |           153.613 |          4.47689 |            7497.06 |
| qrdqn_baseline              |        3 |            147.696 |          11.1008  |           149.8   |         11.1316  |            3737.48 |
| qrdqn_nq101_batch64         |        3 |            145.118 |          12.8254  |           147.36  |         12.7797  |            9725.49 |

## Notas
- Los valores son promedios sobre los seeds de cada variante.
- Recomiendo contrastar con la varianza (std) para elegir configuración estable.