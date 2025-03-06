



import cv2
import ppo
from train_network_baseline_custom_net import ProcGenWrapper


PATH = "/home/lord225/pyrepos/explain-rl/models/20250305-135940-GreatFewCoach_9_v3.0"
model = ppo.PPO.load(PATH, print_system_info=True)


env = ProcGenWrapper("starpilot", human=True)

obs, _ = env.reset()

reward_sum = 0
while True:
    action, _ = model.predict(obs)

    obs, rew, done, _, _ = env.step(action[0])

    reward_sum += rew

    cv2.waitKey(32)

    if done:
        print(reward_sum)9
        reward_sum = 0
        obs, _ = env.reset()

    