import yaml
from argparse import ArgumentParser
from emergent_gym.modules.gym_env.Config.GameGymConfig import GymConfig
from emergent_gym.modules.gym_env.GameGym import GameGym


class Game:
    def __init__(self, filename:str, verbose: bool = False):
        self._config = self.load_config(filename)
        self._verbose = verbose
        if self._verbose:
            print(self.__str__())

    def load_config(self, filename: str):
        fP = open(filename, "r")
        config_file = yaml.load(fP,yaml.FullLoader)
        fP.close()
        print(config_file)
        config = GymConfig(**config_file)
        return config

    @property
    def config(self):
        return self._config


    def run(self):
        #Todo: This is where the training script will be integrated
        pass

    def __str__(self):
        return self._config.__str__()


if __name__ == "__main__":
    parser = ArgumentParser("Test Game")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("-v","--verbose", type=bool, default=False)
    args = parser.parse_args()
    game = Game(args.config,args.verbose)
    game.run()


