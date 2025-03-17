



import cv2
import ppo
from procgenwrapper import ProcGenWrapper
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--resume", type=str, default=None, help="resume from a model")

args = parser.parse_args()


PATH = args.resume
model = ppo.PPO.load(PATH, print_system_info=True)


env = ProcGenWrapper("starpilot", human=True)

obs, _ = env.reset()

reward_sum = 0
while True:
    action, _ = model.predict(obs)

    obs, rew, done, _, _ = env.step(action)

    reward_sum += rew

    cv2.waitKey(32)

    if done:
        print(reward_sum)
        reward_sum = 0
        obs, _ = env.reset()

    