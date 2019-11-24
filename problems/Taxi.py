import gym
import numpy as np

class Taxi:
    def __init__(self):
        env = gym.make("Taxi-v3").env
        env.reset()
        env.render()

    def get_env(self):

        #random_map = generate_random_map(size = self.size, p = 0.8)
        env = gym.make("Taxi-v3")
        env.reset()
        env.render()
        return env        


