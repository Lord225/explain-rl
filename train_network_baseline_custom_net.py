import argparse
from typing import Any, Optional, Union
import gymnasium as gym
from procgen import ProcgenEnv
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch
import torch.nn as nn
import tensorboardX
import cv2

from rl.common import splash_screen


params = argparse.Namespace()

params.env_name = "starpilot"
params.version = "v3.0"
params.DRY_RUN = False

class ProcGenWrapper(gym.Env):
    def __init__(self, env_name="starpilot", num_levels=0, start_level=0, distribution_mode="easy", human=False):
        super().__init__()

        # Create single-instance ProcgenEnv (num_envs=1)
        self.venv = ProcgenEnv(num_envs=1, env_name=env_name, num_levels=num_levels, 
                               start_level=start_level, distribution_mode=distribution_mode)

        # Extract observation & action space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(n=15)
        self.human = human

    def reset(self, seed=None, options=None):
        obs = self.venv.reset()
        return obs['rgb'], {}  

    def step(self, action):
        obs, rewards, dones, infos = self.venv.step(np.array([action])) 
        obs = obs['rgb']
        if self.human:
            frame = obs.reshape((64, 64, 3))
            frame = cv2.resize(frame, (frame.shape[1]*8, frame.shape[0]*8), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("Human", frame)

        return obs, rewards[0], dones[0], False, infos[0] 


    def close(self):
        self.venv.close()

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate feature size after convolutions
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, 64, 64)
            output_size = self.cnn(sample_input).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(output_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.linear(x)

# Custom policy using the custom feature extractor
class CustomCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space, action_space, lr_schedule,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
            **kwargs
        )

from rl import config
from tensorboardX import SummaryWriter
from ppo import PPO


if __name__ == "__main__":
    run_name = config.RUN_NAME

    train_summary_writer = SummaryWriter(config.LOG_DIR_ROOT + run_name + params.version) # type: ignore        
        
    venv = ProcGenWrapper(env_name="starpilot", num_levels=1, start_level=0, distribution_mode="easy")

    # Define and train model
    model = PPO(CustomCnnPolicy, venv, verbose=1, tensorboard_log=config.LOG_DIR_ROOT)

    for i in range(10):
        model.learn(total_timesteps=100000, log_interval=1, tb_log_name=f"{run_name}{params.version}")

        model.save(config.MODELS_DIR + f"{run_name}_{i}_{params.version}")