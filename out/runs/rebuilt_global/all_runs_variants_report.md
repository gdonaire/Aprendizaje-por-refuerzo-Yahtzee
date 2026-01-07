# Informe unificado de variantes: rebuilt_global

## Resumen por variante (todas las campañas), ordenado por puntuación final media
| run_name               | variant                     |   n_runs |   mean_reward_mean |   mean_reward_std |   mean_score_mean |   mean_score_std |   elapsed_mean_sec |
|:-----------------------|:----------------------------|---------:|-------------------:|------------------:|------------------:|-----------------:|-------------------:|
| qrdqn_variants         | qrdqn_nq25_batch128         |        3 |           153.453  |          3.81374  |           155.607 |         3.71022  |           3592.42  |
| qrdqn_variants         | qrdqn_target_slow_grad_more |        3 |           151.495  |          4.50794  |           153.613 |         4.47689  |           7497.06  |
| dqn_variants           | dqn_batch_big               |        3 |           150.335  |          2.74974  |           152.527 |         2.66416  |           1933.75  |
| dqn_variants           | dqn_baseline                |        3 |           149.91   |          0.321081 |           152.06  |         0.302655 |           1849.46  |
| dqn_variants           | dqn_target_slow_grad_more   |        3 |           148.889  |          4.73106  |           151.12  |         4.66952  |           2863.6   |
| qrdqn_variants         | qrdqn_baseline              |        3 |           147.696  |         11.1008   |           149.8   |        11.1316   |           3737.48  |
| qrdqn_variants         | qrdqn_nq101_batch64         |        3 |           145.118  |         12.8254   |           147.36  |        12.7797   |           9725.49  |
| dqn_variants           | dqn_nstep5_epsfloor010      |        3 |           144.108  |          1.51638  |           146.34  |         1.59737  |           1750.39  |
| mppo_variants          | mppo_baseline               |        3 |           138.618  |          5.72163  |           140.72  |         5.80496  |           2168.38  |
| mppo_variants          | mppo_conservative_clip_kl   |        3 |           132.577  |          2.58841  |           134.72  |         2.65774  |           1650.62  |
| mppo_variants          | mppo_highhorizon_lowvar     |        3 |           131.605  |          6.04964  |           133.747 |         6.04921  |           1521.69  |
| mppo_variants          | mppo_big_net                |        3 |           124.807  |          5.52     |           127     |         5.57606  |           4579.5   |
| ppo_variants           | ppo_baseline                |        3 |           123.588  |          1.20356  |           125.767 |         1.27441  |           1846.83  |
| ppo_variants           | ppo_conservative_clip_kl    |        3 |           121.957  |          4.57959  |           124.153 |         4.47957  |           1956.55  |
| ppo_variants           | ppo_highhorizon_lowvar      |        3 |           120.94   |          2.2278   |           123.133 |         2.27388  |           1401.09  |
| ppo_variants           | ppo_big_net                 |        3 |           120.683  |          3.38231  |           122.807 |         3.29268  |           3574.28  |
| a2c_variants           | a2c_baseline                |        3 |           118.314  |          2.61705  |           120.533 |         2.62392  |            622.85  |
| a2c_variants           | a2c_longrollout             |        3 |           115.662  |          3.70835  |           117.807 |         3.66177  |            577.85  |
| a2c_variants           | a2c_fastexploit             |        3 |           114.946  |          0.82952  |           117.053 |         0.85049  |            583.226 |
| a2c_variants           | a2c_highhorizon_lowvar      |        3 |           113.925  |          2.59777  |           116.107 |         2.59248  |            573.963 |

## TOP por variante (global)
| run_name       | algorithm   | variant                     |   mean_score_mean |   mean_score_std |   mean_reward_mean |   mean_reward_std |   elapsed_mean_sec |
|:---------------|:------------|:----------------------------|------------------:|-----------------:|-------------------:|------------------:|-------------------:|
| qrdqn_variants | qrdqn       | qrdqn_nq25_batch128         |           155.607 |         3.71022  |            153.453 |          3.81374  |            3592.42 |
| qrdqn_variants | qrdqn       | qrdqn_target_slow_grad_more |           153.613 |         4.47689  |            151.495 |          4.50794  |            7497.06 |
| dqn_variants   | dqn         | dqn_batch_big               |           152.527 |         2.66416  |            150.335 |          2.74974  |            1933.75 |
| dqn_variants   | dqn         | dqn_baseline                |           152.06  |         0.302655 |            149.91  |          0.321081 |            1849.46 |
| dqn_variants   | dqn         | dqn_target_slow_grad_more   |           151.12  |         4.66952  |            148.889 |          4.73106  |            2863.6  |
| qrdqn_variants | qrdqn       | qrdqn_baseline              |           149.8   |        11.1316   |            147.696 |         11.1008   |            3737.48 |
| qrdqn_variants | qrdqn       | qrdqn_nq101_batch64         |           147.36  |        12.7797   |            145.118 |         12.8254   |            9725.49 |
| dqn_variants   | dqn         | dqn_nstep5_epsfloor010      |           146.34  |         1.59737  |            144.108 |          1.51638  |            1750.39 |
| mppo_variants  | mppo        | mppo_baseline               |           140.72  |         5.80496  |            138.618 |          5.72163  |            2168.38 |
| mppo_variants  | mppo        | mppo_conservative_clip_kl   |           134.72  |         2.65774  |            132.577 |          2.58841  |            1650.62 |

## TOP por algoritmo (mejor mean_score_mean)
| algorithm   | run_name       | variant             |   mean_score_mean |   mean_score_std |   mean_reward_mean |   mean_reward_std |   elapsed_mean_sec |
|:------------|:---------------|:--------------------|------------------:|-----------------:|-------------------:|------------------:|-------------------:|
| a2c         | a2c_variants   | a2c_baseline        |           120.533 |          2.62392 |            118.314 |           2.61705 |             622.85 |
| dqn         | dqn_variants   | dqn_batch_big       |           152.527 |          2.66416 |            150.335 |           2.74974 |            1933.75 |
| mppo        | mppo_variants  | mppo_baseline       |           140.72  |          5.80496 |            138.618 |           5.72163 |            2168.38 |
| ppo         | ppo_variants   | ppo_baseline        |           125.767 |          1.27441 |            123.588 |           1.20356 |            1846.83 |
| qrdqn       | qrdqn_variants | qrdqn_nq25_batch128 |           155.607 |          3.71022 |            153.453 |           3.81374 |            3592.42 |

## Notas
- Agregación basada en results_by_seed.csv y/o aggregate_summary.json ya existentes.
- No se re-entrena ni se re-evalúa nada, solo se re-lee y consolida.
- Desviación estándar calculada con ddof=1 (muestral).