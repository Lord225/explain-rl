# this script is used to analize why starpilotv1_20250206-213758 model is not working. 
# Issue in run `20250208-230433v1`

from collections import deque
import tensorflow as tf
import tensorboard
import os
import argparse
import tqdm
import os
import numpy as np

# add root directory to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from rl import enviroment
from rl import config


parser = argparse.ArgumentParser()

parser.add_argument('--resume', type=str, default=None)

args = parser.parse_args()


def get_actor():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'/actor')
        print("loaded model from", args.resume)
        return model



def get_critic():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'/critic')
        print("loaded model from", args.resume)
        return model

    return model

def get_curiosity_autoencoder():
    if args.resume is not None:
        autoencoder = tf.keras.models.load_model(args.resume+'/autoencoder')
        print(autoencoder.layers)
        encoder = tf.keras.Model(inputs=autoencoder.encoder.input, outputs=autoencoder.encoder.output) # type: ignore
        print("loaded model from", args.resume)
        return encoder, autoencoder


def get_curiosity():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'/curiosity')
        print("loaded model from", args.resume)
        return model

    return model
    

actor = get_actor()
critic = get_critic()
curiosity = get_curiosity()
encoder, autoencoder = get_curiosity_autoencoder()


env = enviroment.ProcGenWrapper("caveflyer", 1, False, 3)


# try getting the next state
state = env.reset()

state = tf.convert_to_tensor(state, dtype=tf.float32)

action = actor(state)

print("state", state)
print("action", action)

# try getting the critic model predictions
value = critic(state)
print("value", value)

# try getting the curiosity model predictions (encoder, autoencoder)
encoded_state = encoder(state)
decoded_state = autoencoder(state)

# plot decoded state
import matplotlib.pyplot as plt
plt.imshow(decoded_state[0][:,:,6:])
plt.show()

# plot encoded state
plt.imshow(encoded_state[0][:,:,6:])
plt.show()

# check curiosity model prediction based on choosed action and state






