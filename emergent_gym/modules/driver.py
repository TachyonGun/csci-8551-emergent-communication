import json
from argparse import ArgumentParser
from emergent_gym.modules.gym_env.Config.GameGymConfig import GymConfig
from emergent_gym.modules.gym_env.GameGym import GameGym


def load_config(filename: str):
    fP = open(filename, "r")
    config = GymConfig(json.load(fP))
    fP.close()
    return config


if __name__ == "__main__":
    parser = ArgumentParser("Test Driver")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(config)
    environment = GameGym(config)
