import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

class FrozenLake:
    def __init__(self, size):
        self.size = size


    def get_env(self):

        random_map = generate_random_map(size = self.size, p = 0.8)
        env = gym.make("FrozenLake-v0", desc = random_map)
        env.reset()
        env.render()
        return env



