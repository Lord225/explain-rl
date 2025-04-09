import argparse
from typing import Any, Optional, Tuple, Union
import gymnasium as gym
from procgen import ProcgenEnv
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch
import torch.nn as nn
import tensorboardX
import cv2

from procgenwrapper import ProcGenWrapper
from rl.common import splash_screen

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="resume from a model") 

args = parser.parse_args()

params = argparse.Namespace()

params.env_name = "starpilot"
params.version = "vCNN_v2.0"
params.DRY_RUN = False
params.resume = args.resume
if args.resume is not None:
    params.resumed = 'resumed from: ' + args.resume
    # path has form of /path/to/model/episode so take last part of path
    # 20250311-160440-FigurePictureMemory_10_v3.0
    # parse _10_ 
    BASE_EPISODE = int(args.resume.split("_")[-2]) + 1
else:
    params.resumed = 'not resumed'
    BASE_EPISODE = None

from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Unflatten(1, (9, 64, 64)),
            # feature_dim -> 64x64x3, use convolutional layers
            nn.Conv2d(9, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, last_layer_dim_pi),
            nn.ReLU(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Unflatten(1, (9, 64, 64)),
            nn.Conv2d(9, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, last_layer_dim_vf),
        )
           

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicyCNN(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

import torchsummary

torchsummary.summary(CustomNetwork(64).policy_net.cuda(),(9*64*64,))

from rl import config
from tensorboardX import SummaryWriter
from ppo import PPO, CustomPPO
from rl import config
from tensorboardX import SummaryWriter
from ppo import PPO


if __name__ == "__main__":
    run_name = config.RUN_NAME

    train_summary_writer = SummaryWriter(config.LOG_DIR_ROOT + run_name + params.version, comet_config={"disabled": True}) # type: ignore        
        
    venv = ProcGenWrapper(env_name="starpilot", 
                          num_levels=0, 
                          start_level=0, 
                          distribution_mode="easy")


    # Define and train model
    model = PPO(CustomActorCriticPolicyCNN, 
                venv, 
                verbose=1, 
                tensorboard_log=config.LOG_DIR_ROOT,
                normalize_advantage=True,
                batch_size=256,
                n_steps=256*18*8,
                n_epochs=3,
                gamma=0.99,
                gae_lambda=0.98,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                learning_rate=1e-5,
    )
    if BASE_EPISODE is not None:
        print(f"Resumed from episode {BASE_EPISODE*100000}")
        # model.load(params.resume, venv, custom_objects={"CustomActorCriticPolicy": CustomActorCriticPolicy, "ViT": ViT})
        print("Loading model from", params.resume, "episode", BASE_EPISODE*100000)
        model = PPO.load(
                params.resume, 
                print_system_info=True, 
                env=venv, 
                tensorboard_log=config.LOG_DIR_ROOT,
                n_steps=256*8*8*4,
                custom_objects={"CustomActorCriticPolicy": CustomActorCriticPolicyCNN, "RolloutBuffer": model.rollout_buffer },
                n_epochs=3,
                learning_rate=3e-7,
                ent_coef=0.02,
             )

    for i in range(100):
        model.learn(total_timesteps=100000, log_interval=1, tb_log_name=f"{run_name}{params.version}", reset_num_timesteps=False, progress_bar=True)

        model.save(config.MODELS_DIR + f"{run_name}_{i}_{params.version}")