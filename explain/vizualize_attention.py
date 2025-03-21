import cv2
import argparse
import sys
sys.path.append("/home/lord225/pyrepos/explain-rl")
import ppo
from procgenwrapper import ProcGenWrapper
import torch as th

parser = argparse.ArgumentParser()

parser.add_argument("--resume", type=str, default=None, help="resume from a model")

args = parser.parse_args()


PATH = args.resume
model = ppo.PPO.load(PATH, print_system_info=True)


env = ProcGenWrapper("starpilot", human=True)

obs, _ = env.reset()

import cv2
import numpy as np
import cv2
def attention_map(attention):
    # avg attention over heads
    attention = np.array(attention).mean(axis=1)

    grid_size = int(np.sqrt(attention.shape[-1] - 1))
    num_layers = attention.shape[0]
    num_heads = attention.shape[1]
    reshaped = attention.reshape(
        (num_layers, num_heads, grid_size**2 + 1, grid_size**2 + 1)
    )

    # From Appendix D.6 in the paper ...
    # Average the attention weights across all heads.
    reshaped = reshaped.mean(axis=1)

    # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    reshaped = reshaped + np.eye(reshaped.shape[1])
    reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = mask / mask.max()
    return mask.reshape(grid_size, grid_size)


def visualize_attention(recorder, obs):
    obs_th = th.tensor(obs).reshape(-1, 64, 64, 9).permute(0, 3, 1, 2).float().cuda()
    _, attention = recorder(obs_th)

    map = attention_map(attention.cpu().detach().numpy())

    return map

reward_sum = 0


net = model.policy

vit = net.mlp_extractor.policy_net[1]
action_net = net.action_net

from vit_pytorch.extractor import Extractor
from vit_pytorch.recorder import Recorder

extractor = Extractor(vit)
recorder = Recorder(vit)

while True:
    action, _ = model.predict(obs)

    obs, rew, done, _, info = env.step(action)

    reward_sum += rew

    segments = cv2.resize(info["seg"], (64*8, 64*8))

    att_map = visualize_attention(recorder, obs)
    
    att_map = cv2.resize(att_map, (64*8, 64*8))
    att_map = cv2.applyColorMap((att_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

    obs_map = cv2.resize(obs.astype(np.uint8), (64*8, 64*8))[:,:,6:]
    
    att_map = cv2.addWeighted(att_map, 0.5, obs_map, 0.5, 0)
    
    cv2.imshow("attention", att_map)
    cv2.imshow("segments", segments)

    cv2.waitKey(32)

    if done:
        print(reward_sum)
        reward_sum = 0
        obs, _ = env.reset()

    