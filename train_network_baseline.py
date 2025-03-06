import gymnasium as gym
from stable_baselines3 import PPO
from procgen import ProcgenEnv
from stable_baselines3.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize,
    DummyVecEnv
)
import numpy as np

class ProcGenWrapper(gym.Env):
    def __init__(self, env_name="starpilot", num_levels=0, start_level=0, distribution_mode="easy"):
        super().__init__()

        # Create single-instance ProcgenEnv (num_envs=1)
        self.venv = ProcgenEnv(num_envs=1, env_name=env_name, num_levels=num_levels, 
                               start_level=start_level, distribution_mode=distribution_mode)

        # Extract observation & action space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(n=15)

    def reset(self, seed=None, options=None):
        obs = self.venv.reset()
        return obs['rgb'], {}  

    def step(self, action):
        obs, rewards, dones, infos = self.venv.step(np.array([action])) 
        obs = obs['rgb']
        return obs, rewards[0], dones[0], False, infos[0] 

    def render(self, mode="human"):
        self.venv.render()

    def close(self):
        self.venv.close()

# Create environment
venv = ProcGenWrapper(env_name="starpilot", num_levels=1, start_level=0, distribution_mode="easy")

# Define and train model
model = PPO("CnnPolicy", venv, verbose=1)
model.learn(total_timesteps=1_000_000)

