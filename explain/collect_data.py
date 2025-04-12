# collect data from model and env into h5 file
from collections import deque
from typing import Union
import numpy as np
import argparse
import sys
import os
import h5py
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import tqdm
from ppo import PPO, CustomPPO


import procgenwrapper
venv = procgenwrapper.ProcGenWrapper("starpilot", human=False, collect_seg=True, raw_seg=True)

def collect_data_for(model: Union[PPO, CustomPPO], num_samples: int):
    observations = np.zeros((num_samples, 64, 64, 9), dtype=np.uint8)
    actions = np.zeros((num_samples), dtype=np.int32)
    rewards = np.zeros((num_samples), dtype=np.float32)
    dones = np.zeros((num_samples), dtype=np.bool_)
    seg_observations = np.zeros((num_samples, 64, 64, 3), dtype=np.uint8)
    next_observations = np.zeros((num_samples, 64, 64, 9), dtype=np.uint8)
    i = 0
    END = False
    with tqdm.tqdm(total=num_samples) as pbar:
        while not END:
            obs, _ = venv.reset()
            done = False
            while not done:
                if i > num_samples-1:
                    END = True
                    break
                action, _ = model.predict(obs)

                next_obs, reward, done, _, info = venv.step(action)

                observations[i] = np.uint8(obs)
                actions[i] = action
                rewards[i] = reward
                dones[i] = done
                seg_observations[i] = np.uint8(info["seg_onehot"])
                next_observations[i] = np.uint8(next_obs)

                i += 1
                
                obs = next_obs

                pbar.update(1)
                
                if done:
                    pbar.set_description(f"r: {np.sum(rewards)/np.sum(dones):.2f}")
                    break

    return observations, actions, rewards, dones, seg_observations, next_observations

argsparser = argparse.ArgumentParser()

argsparser.add_argument("--models", type=str, default="/home/lord225/pyrepos/explain-rl/preserve", help="path to models")
argsparser.add_argument("--num_samples", type=int, default=10_000, help="number of samples to collect")
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
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, device="cuda")
        # model name is file name
        model_name = model_path.split("/")[-1]
        observations, actions, rewards, dones, seg_observations, next_observations = collect_data_for(model, args.num_samples)
        print(f"Saving collected samples incrementally as {model_name}_replay.h5")
        with h5py.File(f"{PATH}/{model_name}_replay.h5", "w") as f:
            f.create_dataset("observations", shape=observations.shape, dtype=np.uint8, chunks=True, compression="gzip")
            f.create_dataset("actions", shape=actions.shape, dtype=np.int32, chunks=True, compression="gzip")
            f.create_dataset("rewards", shape=rewards.shape, dtype=np.float32, chunks=True, compression="gzip")
            f.create_dataset("dones", shape=dones.shape, dtype=np.bool_, chunks=True, compression="gzip")
            f.create_dataset("seg_observations", shape=seg_observations.shape, dtype=np.uint8, chunks=True, compression="gzip")
            f.create_dataset("next_observations", shape=next_observations.shape, dtype=np.uint8, chunks=True, compression="gzip")
            
            f["observations"][:] = observations
            f["actions"][:] = actions
            f["rewards"][:] = rewards
            f["dones"][:] = dones
            f["seg_observations"][:] = seg_observations
            f["next_observations"][:] = next_observations
        
        # print statistics
        print(f"Model: {model_name}")
        print(f"Avg reward: {np.sum(rewards)/np.sum(dones)}")
        print(f"Max reward: {np.max(rewards)}")
        print(f"Std reward: {np.min(rewards)}")
        del observations
        del actions
        del rewards
        del dones
        del seg_observations
        del next_observations
        del model
        
if __name__ == "__main__":
    main()





