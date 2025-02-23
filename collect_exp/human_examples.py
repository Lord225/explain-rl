


from collections import deque
import random
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

from rl import enviroment
import matplotlib.pyplot as plt
import cv2
from procgen import ProcgenGym3Env
from gym3 import Interactive, VideoRecorderWrapper, unwrap


PATH = "F:/pyrepos/explain-rl/collect_exp/database/"

class ProcgenInteractiveCollector(Interactive):
    # collect: states, actions, rewards and save as hdf5 file with datetime, env_name, and final score into thge path
    data_collection = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_state = None
        self.frame_stack_count = kwargs.get("frame_stack_count", 3)
        self.frame_stack = deque(maxlen=self.frame_stack_count)
        for _ in range(self.frame_stack_count):
            self.frame_stack.append(np.zeros((1, 64, 64, 3), dtype=np.uint8))


    def collect_data(self):
        # get current rgb observation
        reward = self._last_rew
        action = self._last_ac
        state = self._last_ob
        
        self.frame_stack.append(state['rgb'] if state is not None else np.zeros((1, 64, 64, 3), dtype=np.uint8))
        state = np.concatenate(self.frame_stack, axis=3)

        self.data_collection.append((state, action, reward))

    def _update(self, dt, keys_clicked, keys_pressed):
        if "LEFT_SHIFT" in keys_pressed and "F1" in keys_clicked:
            print("save state")
            self._saved_state = unwrap(self._env).get_state()
        elif "F1" in keys_clicked:
            print("load state")
            if self._saved_state is not None:
                unwrap(self._env).set_state(self._saved_state)
        
        if self._episode_steps == 0:
            self.save_data(PATH)
            for _ in range(self.frame_stack_count):
                self.frame_stack.append(np.zeros((1, 64, 64, 3), dtype=np.uint8))
            self.data_collection = []
            
        self.collect_data()

        super()._update(dt, keys_clicked, keys_pressed)
    
    def save_data(self, path):
        # calculate final score and save data
        final_score = sum([r for _, _, r in self.data_collection if r is not None])

        if final_score == 0 or len(self.data_collection) == 0:
            print("No data collected")
            return

        # convert to numpy arrays
        states = np.array([s for s, _, _ in self.data_collection if s is not None])
        actions = np.array([a for _, a, _ in self.data_collection if a is not None])
        rewards = np.array([r for _, _, r in self.data_collection if r is not None])

        # squeeze states
        states = np.squeeze(states)

        run_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_name = "starpilot"
        filename = f"{run_datetime}_{env_name}_{final_score}.h5"

        with h5py.File(os.path.join(path, filename), 'w') as f:
            f.create_dataset('states', data=states)
            f.create_dataset('actions', data=actions)
            f.create_dataset('rewards', data=rewards)
            f.attrs['final_score'] = final_score
            f.attrs['env_name'] = env_name
            f.attrs['run_datetime'] = run_datetime
        
        print(f"Data saved to {os.path.join(path, filename)}")



def make_interactive(vision, record_dir, **kwargs):
    info_key = None
    ob_key = None
    if vision == "human":
        info_key = "rgb"
        kwargs["render_mode"] = "rgb_array"
    else:
        ob_key = "rgb"

    env = enviroment.ProcGenWrapper("starpilot", num=1, return_segments=False, frame_stack_count=3)
    if record_dir is not None:
        env = VideoRecorderWrapper(
            env=env, directory=record_dir, ob_key=ob_key, info_key=info_key
        )
    h, w, _ = env.ob_space["rgb"].shape
    return ProcgenInteractiveCollector(
        env,
        ob_key=ob_key,
        info_key=info_key,
        width=w * 12,
        height=h * 12,
    )
        

def main():
    ia = make_interactive(
        vision="rgb",
        record_dir=None,
        num_levels=0,
        start_level=0,
        distribution_mode="easy",
        render_mode="rgb_array",
        use_sequential_levels=True,
    )
    ia.run()


if __name__ == "__main__":

    main()
