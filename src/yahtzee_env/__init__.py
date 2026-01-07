from gymnasium.envs.registration import register

register(
    id="yahtzee_env/Yahtzee-v0",
    entry_point="yahtzee_env.envs:YahtzeeEnv",
)

register(
    id="yahtzee_env/Yahtzee-v1",
    entry_point="yahtzee_env.envs:YahtzeeEnvV1",
)

register(
    id="yahtzee_env/Yahtzee-v2",
    entry_point="yahtzee_env.envs:YahtzeeEnvV2",
)