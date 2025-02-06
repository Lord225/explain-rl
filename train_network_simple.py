# Vanilla PPO with curiosity

from collections import deque
import gymnasium
import tensorflow as tf
import tensorboard
import os
import argparse
import tqdm
from rl.common import splash_screen
import rl.config as config
from rl.episode_runner import get_curius_ppo_runner_2
import rl.ppo as ppo
import rl.enviroment as enviroments
import os
import numpy as np





parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="resume from a model") 

args = parser.parse_args()

env = enviroments.ProcGenWrapper("starpilot", 1, False, 3)

params = argparse.Namespace()

params.env_name = env.env
params.version = "v6.1"
params.DRY_RUN = False

params.actor_lr  = 1e-4
params.critic_lr = 1e-3

params.action_space = 15
params.observation_space_raw =  (64, 64, 9)
params.observation_space = (64, 64, 9)
params.encoding_size = 128

params.episodes = 100000
params.max_steps_per_episode = 1200

params.discount_rate = 0.99

params.eps_decay_len = 1000
params.eps_min = 0.1

params.clip_ratio = 0.20
params.lam = 0.98

# params.curius_coef = 0.013
params.curius_coef = 0.0001

params.batch_size = 4096
params.batch_size_curius = 300

params.train_interval = 1
params.iters = 20
params.iters_courious = 30

params.save_freq = 1000
if args.resume is not None:
    params.resumed = 'resumed from: ' + os.path.basename(args.resume)

splash_screen(params)

def get_actor():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'.actor.h5')
        print("loaded model from", args.resume)
        return model

    observation_input = tf.keras.Input(shape=params.observation_space, dtype=tf.int8)
    x = tf.cast(observation_input, tf.float32)
    x = x / 255.0 # type: ignore
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(64, 2, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(64, 2, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(64, 2, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Dense(32, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Dense(32, activation=tf.nn.elu)(x)

    logits = tf.keras.layers.Dense(params.action_space)(x)

    model = tf.keras.Model(inputs=observation_input, outputs=logits, name='actor')

    model.summary()

    return model

def get_critic():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'.critic.h5')
        print("loaded model from", args.resume)
        return model

    observation_input = tf.keras.Input(shape=params.observation_space, dtype=tf.int8)
    x = tf.cast(observation_input, tf.float32)
    x = x / 255.0 # type: ignore
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Dense(16, activation=tf.nn.elu)(x)


    value = tf.squeeze(tf.keras.layers.Dense(1)(x))
    model = tf.keras.Model(inputs=observation_input, outputs=value, name='critic')
    model.summary()
    return model

def get_curiosity_autoencoder():
    if args.resume is not None:
        autoencoder = tf.keras.models.load_model(args.resume+'.autoencoder.h5')
        encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output) # type: ignore
        print("loaded model from", args.resume)
        return encoder, autoencoder
    

    observation_input = tf.keras.Input(shape=params.observation_space, dtype=tf.int8)
    x = tf.cast(observation_input, tf.float32)
    x = x / 255.0 # type: ignore
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', activation=tf.nn.elu)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.elu)(x)
    
    z_mean = tf.keras.layers.Dense(params.encoding_size, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(params.encoding_size, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = tf.keras.layers.Lambda(sampling, output_shape=(params.encoding_size,), name='z')([z_mean, z_log_var])

    encoder_model = tf.keras.Model(inputs=observation_input, outputs=z, name='encoder')

    latent_inputs = tf.keras.Input(shape=(params.encoding_size,), dtype=tf.float32)
    x = tf.keras.layers.Dense(64, activation=tf.nn.elu)(latent_inputs)
    x = tf.keras.layers.Dense(32, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Reshape((1, 1, 32))(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(8, 3, strides=1, padding='same', activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(6, 3, strides=1, padding='same', activation=tf.nn.elu)(x)

    autoencoder_model = tf.keras.Model(inputs=[observation_input, latent_inputs], outputs=x, name='autoencoder')

    def vae_loss(observation_input, x):
        reconstruction_loss = tf.keras.losses.mse(tf.keras.backend.flatten(observation_input), tf.keras.backend.flatten(x))
        reconstruction_loss *= params.observation_space[0] * params.observation_space[1]
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return tf.keras.backend.mean(reconstruction_loss + kl_loss)

    autoencoder_model.add_loss(vae_loss(observation_input, x))
    autoencoder_model.compile(optimizer='adam')

    autoencoder_model.summary()

    return encoder_model, autoencoder_model

def get_curiosity():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'.curiosity.h5')
        print("loaded model from", args.resume)
        return model
    
    observation_input = tf.keras.Input(shape=params.encoding_size, dtype=tf.float32)
    action_input = tf.keras.Input(shape=params.action_space, dtype=tf.float32)

    x = tf.keras.layers.Concatenate()([observation_input, action_input]) # 128 + 10
    x = tf.keras.layers.Dense(128, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Dense(params.encoding_size, activation=tf.nn.tanh)(x)

    model =  tf.keras.Model(inputs=[observation_input, action_input], outputs=x, name='curiosity')

    model.summary()

    return model
    

actor = get_actor()
critic = get_critic()
curiosity = get_curiosity()
encoder, autoencoder = get_curiosity_autoencoder()


policy_optimizer = tf.keras.optimizers.Adam(learning_rate=params.actor_lr)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=params.critic_lr)
curiosity_optimizer = tf.keras.optimizers.Adam(learning_rate=params.critic_lr)
autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=params.critic_lr)

def log_stats(stats, step):
    # list of kl, policy_loss, mean_ratio, mean_clipped_ratio, mean_advantage, mean_logprob
    tf.summary.scalar('kl', np.mean([x[0] for x in stats]), step=step)
    tf.summary.scalar('loss', np.mean([x[1] for x in stats]), step=step)
    tf.summary.scalar('mean_ratio', np.mean([x[2] for x in stats]), step=step)
    tf.summary.scalar('mean_clipped_ratio', np.mean([x[3] for x in stats]), step=step)
    tf.summary.scalar('mean_logprob', np.mean([x[5] for x in stats]), step=step)
    
def run():
    running_avg = deque(maxlen=200)

    memory = ppo.PPOReplayMemory(16_000, params.observation_space, gamma=params.discount_rate, lam=params.lam, gather_next_states=True)

    env_step = enviroments.make_tensorflow_env_step(env, lambda x: x) # type: ignore
    env_reset = enviroments.make_tensorflow_env_reset(env, lambda x: x) # type: ignore

    runner = get_curius_ppo_runner_2(env_step)
    runner = tf.function(runner)

    action_space = tf.constant(params.action_space, dtype=tf.int32)
    max_steps_per_episode = tf.constant(params.max_steps_per_episode, dtype=tf.int64)
    batch_size = tf.constant(params.batch_size, dtype=tf.int64)
    batch_size_curius = tf.constant(params.batch_size_curius, dtype=tf.int64)
    clip_ratio = tf.constant(params.clip_ratio, dtype=tf.float32)

    t = tqdm.tqdm(range(params.episodes))
    for episode in t:
        initial_state = env_reset()

        initial_state = tf.constant(initial_state, dtype=tf.int8)
        
        curius_coef = tf.constant(params.curius_coef, dtype=tf.float32)

        (states, actions, rewards, values, log_probs, next_states), total_rewards, curiosity_mean, curiosity_std = runner(initial_state, actor, critic, curiosity, encoder, max_steps_per_episode, action_space, curius_coef) # type: ignore
        
        memory.add(states, actions, rewards, values, log_probs, next_states)
        
        curiosity_sum = curiosity_mean*params.curius_coef*states.shape[0]
        
        running_avg.append(total_rewards-curiosity_sum)
        avg = sum(running_avg)/len(running_avg)

        tf.summary.scalar('reward', total_rewards-curiosity_sum, step=episode)
        tf.summary.scalar('reward_avg', avg, step=episode)
        tf.summary.scalar('lenght', states.shape[0], step=episode)
        tf.summary.scalar('curiosity_mean', curiosity_mean, step=episode)
        tf.summary.scalar('curiosity_std', curiosity_std, step=episode)
        tf.summary.scalar('curiosity_sum', curiosity_sum, step=episode)

        t.set_description(f"Reward: {total_rewards:.2f} - Reward(Raw): {total_rewards-curiosity_sum:.2f}  - Avg: {avg:.2f} - Iterations: {states.shape[0]} curiosity: {curiosity_mean:.2f} curiosity epoisode: {curiosity_sum:.2f}")

        episode_tf = tf.constant(episode, dtype=tf.int64)

        if len(memory) >= batch_size and int(episode) % params.train_interval == 0:
            stats = [] # kl, policy_loss, mean_ratio, mean_clipped_ratio, mean_advantage, mean_logprob
            for _ in range(params.iters):
                batch = memory.sample(batch_size)
                history = ppo.training_step_ppo(batch, actor, action_space, clip_ratio, policy_optimizer, episode_tf)
                stats.append(history)

            log_stats(stats, episode_tf)

            for _ in range(params.iters):
                batch = memory.sample_critic(batch_size)
                ppo.training_step_critic(batch, critic, value_optimizer, episode_tf)

            for _ in range(params.iters_courious):
                batch = memory.sample_encoded_curiosity(batch_size_curius, encoder)
                ppo.training_step_curiosty(batch, curiosity, curiosity_optimizer, action_space,  episode_tf)

            for _ in range(params.iters_courious):
                batch = memory.sample_autoencoder(batch_size_curius)
                ppo.training_step_autoencoder(batch, autoencoder, autoencoder_optimizer, episode_tf)

            # memory.reset()



        if episode % params.save_freq == 0 and episode > 0: 
            NAME = f"{config.MODELS_DIR}{params.env_name}{params.version}_{config.RUN_NAME}_{episode}"
            
            actor.save(f"{NAME}.actor.h5") # type: ignore
            critic.save(f"{NAME}.critic.h5") # type: ignore
            curiosity.save(f"{NAME}.curiosity.h5") # type: ignore
            autoencoder.save(f"{NAME}.autoencoder.h5") # type: ignore

run()




