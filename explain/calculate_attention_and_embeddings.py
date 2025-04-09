import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import h5py
import numpy as np
from vit_pytorch import ViT
import cv2
import argparse
import sys
import tqdm
sys.path.append("/home/lord225/pyrepos/explain-rl")
import ppo
from procgenwrapper import ProcGenWrapper
import torch as th

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default=None, help="ppo model")
parser.add_argument("--dataset", type=str, default=None, help="cached dataset")

args = parser.parse_args()

model = ppo.PPO.load(args.model, print_system_info=True)

if args.dataset is None:
    RECORDS_PATH = '/home/lord225/pyrepos/explain-rl/explain/records'
    args.dataset = os.path.join(RECORDS_PATH, f"{args.model.split('/')[-1]}_replay.h5")

from vit_pytorch.extractor import Extractor
from vit_pytorch.recorder import Recorder

net = model.policy
vit = net.mlp_extractor.policy_net[1]
action_net = net.action_net

extractor = Extractor(vit)
recorder = Recorder(vit)

dataset = h5py.File(args.dataset, "r")

print(args.dataset)
print(args.model)

observations = np.array(dataset["observations"])
actions = np.array(dataset["actions"])
rewards = np.array(dataset["rewards"])
dones = np.array(dataset["dones"])
seg_observations = np.array(dataset["seg_observations"])
next_observations = np.array(dataset["next_observations"])

batch_size = 4
probas_list = []
attention_list = []
features_list = []

recorder = recorder.cpu()
extractor = extractor.cpu()

for i in tqdm.tqdm(range(0, len(observations), batch_size)):
    batch = th.tensor(observations[i:i+batch_size], device="cpu").permute(0, 3, 1, 2).float().cpu()
    probas, attention = recorder(batch)
    probas, features = extractor(batch)
    probas_list.append(probas.cpu().detach().numpy())
    attention_list.append(attention.cpu().detach().numpy())
    features_list.append(features.cpu().detach().numpy())

    del probas
    del attention
    del features

np.float = float # type: ignore
import dask.array as da

# concat all into dask
probas = da.concatenate(probas_list, axis=0)
attention = da.concatenate(attention_list, axis=0)
features = da.concatenate(features_list, axis=0)

# save to h5
with h5py.File(os.path.join(RECORDS_PATH, f"{args.model.split('/')[-1]}_vitextracted.h5"), "a") as f:
    # meta data
    f.attrs["model"] = args.model
    f.attrs["dataset"] = args.dataset
    f.create_dataset("probas", data=probas)
    f.create_dataset("attention", data=attention)
    f.create_dataset("features", data=features)