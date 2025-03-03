from collections import deque
import tensorflow as tf
import tensorboard
import os
import argparse
import tqdm
import os
import numpy as np
from advanced_networks import VisualTransformer
import tf_agents
import rl.enviroment as enviroments

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="resume from a model") 

args = parser.parse_args()

env = enviroments.ProcGenWrapper("starpilot", 1, False, 3)

params = argparse.Namespace()

params.env_name = env.env
params.version = "v3"
params.DRY_RUN = False

params.actor_lr  = 1e-6
params.critic_lr = 3e-6

params.action_space = 15
params.observation_space_raw =  (64, 64, 9)
params.observation_space = (64, 64, 9)
params.encoding_size = 256

params.episodes = 200_000
params.max_steps_per_episode = 250

params.discount_rate = 0.99

params.eps_decay_len = 10
params.eps_min = 0.15

params.clip_ratio = 0.20
params.lam = 0.98

# params.curius_coef = 0.013
params.curius_coef = 0.00000005

params.batch_size = 128
params.batch_size_curius = 128

params.train_interval = 20
params.iters = 100
params.iters_courious = 100            

params.save_freq = 500



if args.resume is not None:
    params.resumed = 'resumed from: ' + args.resume
    # path has form of /path/to/model/episode so take last part of path
    BASE_EPISODE = int(os.path.basename(args.resume))
else:
    params.resumed = 'not resumed'
    BASE_EPISODE = 0


def get_actor():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'/actor')
        print("loaded model from", args.resume)
        return model

    model = VisualTransformer.ViT(
        image_size=(66, 66, 9),
        patch_size=6,
        num_layers=8,
        hidden_size=32,
        num_heads=4,
        name='actor',
        mlp_dim=16,
        classes=15,
        dropout=0.1,
        activation='linear',
        representation_size=16,
        preprocess=tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(64, 64, 9), dtype=tf.float32),
                tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [2, 0], [2, 0], [0, 0]])),
                tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0),
            ]
        )
    )

    model.build((None, 64, 64, 9))

    model.summary()

    return model


def get_critic():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'/critic')
        print("loaded model from", args.resume)
        return model

    observation_input = tf.keras.Input(shape=params.observation_space, dtype=tf.uint8)
    x = tf.cast(observation_input, tf.float32)
    x = x / 255.0 # type: ignore
    x = tf.keras.layers.Conv2D(16, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Dense(32, activation=tf.nn.elu)(x)


    value = tf.squeeze(tf.keras.layers.Dense(1)(x))
    model = tf.keras.Model(inputs=observation_input, outputs=value, name='critic')
    model.summary()
    return model


from tf_agents.agents.ppo import ppo_agent


actor = get_actor()
critic = get_critic()

optimizer = tf.keras.optimizers.Adam(learning_rate=params.actor_lr)

agent = ppo_agent.PPOAgent(
    time_step_spec=env.time_step_spec,
    action_spec=env.action_spec,
    actor_network=actor,
    critic_network=critic,
    optimizer=optimizer,
    normalize_observations=True,
    normalize_rewards=True,
    discount=params.discount_rate,
    lambda_value=params.lam,
    entropy_regularization=0.0,
    importance_ratio_clipping=0.2,
    value_pred_loss_coef=0.5,
    num_epochs=10,
    use_gae=True,
    use_td_lambda_return=True,
    normalize_advantages=True,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    train_step_counter=tf.Variable(0),
    name='PPOAgent')

from tf_agents.replay_buffers import tf_uniform_replay_buffer



replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=env.collect_data_spec,
    batch_size=params.batch_size,
    max_length=100_000)
