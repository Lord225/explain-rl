import random
import gym
import numpy as np
import argparse
import tensorflow as tf
import sys
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from rl import enviroment
import matplotlib.pyplot as plt
import cv2
def get_actor(model_path):
    if model_path is not None:
        model = tf.keras.models.load_model(model_path + '/actor')
        print("Loaded model from", model_path)
        return model
    return None

def main(model_path):
    seed = random.randint(0, 2 ** 31 - 1)
    env = enviroment.ProcGenWrapper('starpilot', 1, False, 3, seed) 

    actor = get_actor(model_path)

    step = 0
    total_reward = 0
    obs = env.reset()
    while True:
        if actor:
            action = actor.predict(obs)
        else:
            action = env.action_space.sample()
        
        action = np.argmax(action, axis=1)
        obs, rew, done = env.step(action)

        env.render()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            env.reset()
        
        total_reward += rew
        
        print(f"Step {step} reward {rew} done {done}")
        step += 1

    avg_reward = total_reward / step
    print(f"Average reward: {avg_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to the model to load')
    args = parser.parse_args()
    main(args.model)
