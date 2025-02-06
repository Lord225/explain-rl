import gym3 
import numpy as np
from procgen import ProcgenGym3Env


# create environment overide the default seed

class ProcGenWrapper(gym3.Wrapper):
    def __init__(self, env, seed):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass