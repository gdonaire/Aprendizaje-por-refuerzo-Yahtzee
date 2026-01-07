# Comparativa final (eval-only)

| algorithm   | variant             | best_dir                                              |   episodes |   mean_reward |   mean_score |
|:------------|:--------------------|:------------------------------------------------------|-----------:|--------------:|-------------:|
| a2c         | a2c_baseline        | best/a2c_variants/a2c_baseline/seed_42/best           |        200 |       118     |      120.18  |
| ppo         | ppo_baseline        | best/ppo_variants/ppo_baseline/seed_42/best           |        200 |       119.92  |      122.075 |
| dqn         | dqn_batch_big       | best/dqn_variants/dqn_batch_big/seed_42/best          |        200 |       147.825 |      149.95  |
| mppo        | mppo_baseline       | best/mppo_variants/mppo_baseline/seed_42/best         |        200 |       134.792 |      136.94  |
| qrdqn       | qrdqn_nq25_batch128 | best/qrdqn_variants/qrdqn_nq25_batch128/seed_126/best |        200 |       150.474 |      152.65  |