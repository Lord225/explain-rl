from collections import deque
import os
from typing import Any, Optional, Tuple, Union
import gymnasium as gym
from procgen import ProcgenEnv
import numpy as np
import cv2

import torch
import torch.nn as nn


class ProcGenWrapper(gym.Env):
    UNIQUE_COLORS = np.array([[  0,   0,   0],
                            [127,  63, 127],
                            [127,  63, 191],
                            [127, 127, 255],
                            [127, 191,  63],
                            [127, 255, 127],
                            [191, 255, 255],
                            [255, 127, 127],
                            [255, 191, 191],
                            [255, 191, 255]], dtype=np.uint8)
    POOLER = nn.MaxPool2d(kernel_size=4, stride=4)

    def __init__(self, env_name="starpilot", 
                 num_envs=1, 
                 num_levels=0, 
                 start_level=0, 
                 distribution_mode="easy", 
                 frame_stack_count=3, 
                 human=False, 
                 collect_seg=True,
                 raw_seg=False,
                 one_hot=False,
                 use_background=False):
        super().__init__()

        # Create single-instance ProcgenEnv (num_envs=1)
        self.venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, 
                               start_level=start_level, distribution_mode=distribution_mode, use_backgrounds=use_background)
        self.venv_segmented = ProcgenEnv(num_envs=num_envs,
                            env_name=env_name, 
                            num_levels=num_levels, 
                            start_level=start_level,
                            distribution_mode=distribution_mode, 
                            use_monochrome_assets=True,
                            use_backgrounds=False,
                            restrict_themes=True,
                            ) 
        # Extract observation & action space
        self.num_envs = num_envs
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3*frame_stack_count), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(n=15)
        self.human = human
        self.collect_seg = collect_seg
        self.raw_seg = raw_seg
        self.frame_stack_count = frame_stack_count
        self.frame_stack = deque(maxlen=self.frame_stack_count)

    def get_observation(self):
        return np.concatenate(self.frame_stack, axis=2, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        obs = self.venv.reset()
        states = self.venv.env.callmethod("get_state")
        self.venv_segmented.env.callmethod("set_state", states)
        
        for _ in range(self.frame_stack_count-1):
            self.frame_stack.append(np.zeros((64, 64, 3), dtype=np.uint8))
            
        self.frame_stack.append(obs['rgb'][0])

        return self.get_observation(), {}  

    def step(self, action):
        obs, rewards, dones, infos = self.venv.step(np.array([action]))
        obs_seg, _, _, _ = self.venv_segmented.step(np.array([action]))
        obs = obs['rgb']
        obs_seg = obs_seg['rgb']

        self.frame_stack.append(obs[0])
        if self.human:
            frame = obs.reshape((64, 64, 3))
            frame = cv2.resize(frame, (frame.shape[1]*8, frame.shape[0]*8), interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow("Human", frame)
        
        mono = obs_seg[0]
        # mono = np.argmax((mono.reshape(-1, 1, 3) == ProcGenWrapper.UNIQUE_COLORS).all(axis=2), axis=1).reshape(16, 16, 1).astype(np.int32)
        
        if self.raw_seg:
            infos[0]['seg_onehot'] = mono
        elif self.collect_seg:
            mono = torch.from_numpy(mono).permute(2, 0, 1)
            mono = ProcGenWrapper.POOLER(mono).permute(1, 2, 0).numpy()
            mono = np.argmax((mono.reshape(-1, 1, 3) == ProcGenWrapper.UNIQUE_COLORS).all(axis=2), axis=1).reshape(16, 16, 1).astype(np.int32)
        
            infos[0]['seg_onehot'] = mono
        else:
            infos[0]['seg_onehot'] = np.array(cv2.resize(mono, (16, 16), interpolation=cv2.INTER_NEAREST), dtype=np.float32)/255.0

        return self.get_observation(), rewards[0], dones[0], False, infos[0] 


    def close(self):
        self.venv.close()