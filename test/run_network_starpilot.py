import random
import gym
import numpy as np
import argparse
import tensorflow as tf
import sys
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from advanced_networks import VisualTransformer
from rl import enviroment
import matplotlib.pyplot as plt
import cv2



def get_actor(model_path):
    if model_path is not None:
        model = tf.keras.models.load_model(model_path + '/actor', custom_objects={
                                                      'ClassToken': VisualTransformer.ClassToken,
                                                      'AddPositionEmbs': VisualTransformer.AddPositionEmbs,
                                                      'MultiHeadSelfAttention': VisualTransformer.MultiHeadSelfAttention,
                                                      'TransformerBlock': VisualTransformer.TransformerBlock,
                                                      'ViT': VisualTransformer.ViT,
                                                    })
        print("Loaded model from", model_path)
        return model
    return None

def is_vit(model):
    return any([layer.name.startswith('Transformer') for layer in model.layers])

def main(model_path, vit_show_attention):
    seed = random.randint(0, 2 ** 31 - 1)
    env = enviroment.ProcGenWrapper('starpilot', 1, False, 3, seed) 

    actor = get_actor(model_path)

    step = 0
    total_reward = 0
    obs = env.reset()
    while True:
        if actor:
            action = actor.predict(obs, verbose=0)
            # check if the model is a Vision Transformer model
            if is_vit(actor) and vit_show_attention:
                attention_map = VisualTransformer.attention_map(actor, obs.reshape(64, 64, 9))
            else: 
                attention_map = None
        else:
            action = np.random.randn(15)
            attention_map = None
            
        action = np.argmax(action, axis=1)
        obs, rew, done = env.step(action)

        frame = env.render()

        cv2.imshow("Normal", frame)
        if attention_map is not None:
            attention_map = cv2.resize(attention_map, (64*4, 64*4))
            cv2.imshow("Attention", attention_map)

        if cv2.waitKey(16) & 0xFF == ord('q'):
            env.reset()
        
        total_reward += rew
        
        print(f"Step {step} reward {rew} done {done}")
        step += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to the model to load', default="f:\pyrepos\explain-rl\models\starpilot_20250222-163156-EducationPassSoldier\\17000")
    parser.add_argument('--vit-show-attention', action='store_true', help='Show attention maps for ViT model', default=True)
    args = parser.parse_args()
    main(args.model, args.vit_show_attention)
