
from collections import deque
import random
from typing import Union
import gym
import numpy as np
import argparse
import tensorflow as tf
import sys
import sys
import os
import h5py
import numpy as np
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import torch
import tqdm
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from ppo import PPO, CustomPPO

import matplotlib.pyplot as plt
import cv2
from gym3 import Interactive, VideoRecorderWrapper, unwrap

import procgenwrapper


def collect_data_for(model: Union[PPO, CustomPPO], num_samples: int):
    # Create environment
    venv = procgenwrapper.ProcGenWrapper(env_name="starpilot", 
                          num_levels=0, 
                          start_level=0, 
                          distribution_mode="easy",
                          collect_seg=True)
    
    observations = []
    actions = []
    rewards = []
    dones = []
    seg_observations = []
    next_observations = []
    with tqdm.tqdm(total=num_samples) as pbar:
        while len(observations) < num_samples:
            obs, _ = venv.reset()
            done = False
            while not done:
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(obs, "cuda")  # type: ignore
                    obs_tensor = obs_tensor.reshape((1, -1))
                    action, _, _ = model.policy(obs_tensor)
                    action = action.cpu().numpy()[0]

                next_obs, reward, done, _, info = venv.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                seg_observations.append(info['seg'])
                next_observations.append(next_obs)
                obs = next_obs
            pbar.update(1)
    
    return observations, actions, rewards, dones, seg_observations, next_observations

argsparser = argparse.ArgumentParser()

argsparser.add_argument("--models", type=str, default="/home/lord225/pyrepos/explain-rl/preserve", help="path to models")
argsparser.add_argument("--num_samples", type=int, default=50_000, help="number of samples to collect")
argsparser.add_argument("--output", type=str, default="/home/lord225/pyrepos/explain-rl/explain/records", help="output path")
argsparser.add_argument("--override", action="store_true", help="override existing records")

def main():
    args = argsparser.parse_args()

    # check if models is path or file
    if os.path.isdir(args.models):
        models = [os.path.join(args.models, model) for model in os.listdir(args.models)]
    else:
        models = [args.models]

    PATH = "/home/lord225/pyrepos/explain-rl/explain/records"
    
    for model_path in models:
        model = PPO.load(model_path, device="cuda")
        # model name is file name
        model_name = model_path.split("/")[-1]
        observations, actions, rewards, dones, seg_observations, next_observations = collect_data_for(model, args.num_samples)
        with h5py.File(f"{PATH}/{model_name}_replay.h5", "w") as f:
            f.create_dataset("observations", data=np.array(observations))
            f.create_dataset("actions", data=np.array(actions))
            f.create_dataset("rewards", data=np.array(rewards))
            f.create_dataset("dones", data=np.array(dones))
            f.create_dataset("seg_observations", data=np.array(seg_observations))
            f.create_dataset("next_observations", data=np.array(next_observations))
        
        # print statistics
        print(f"Model: {model_name}")
        print(f"Avg reward: {np.mean(rewards)}")
        print(f"Max reward: {np.max(rewards)}")
        print(f"Std reward: {np.min(rewards)}")
        
if __name__ == "__main__":
    main()





