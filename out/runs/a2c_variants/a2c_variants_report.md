# Informe global de variantes: a2c_variants

## Resumen por variante (ordenado por puntuación final media)
| variant                |   n_runs |   mean_reward_mean |   mean_reward_std |   mean_score_mean |   mean_score_std |   elapsed_mean_sec |
|:-----------------------|---------:|-------------------:|------------------:|------------------:|-----------------:|-------------------:|
| a2c_baseline           |        3 |            118.314 |           2.61705 |           120.533 |          2.62392 |            622.85  |
| a2c_longrollout        |        3 |            115.662 |           3.70835 |           117.807 |          3.66177 |            577.85  |
| a2c_fastexploit        |        3 |            114.946 |           0.82952 |           117.053 |          0.85049 |            583.226 |
| a2c_highhorizon_lowvar |        3 |            113.925 |           2.59777 |           116.107 |          2.59248 |            573.963 |

## Notas
- Los valores son promedios sobre los seeds de cada variante.
- Observa std para seleccionar configuraciones estables.