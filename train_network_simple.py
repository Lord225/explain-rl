# Vanilla PPO with curiosity

from collections import deque
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

env = enviroments.ProcGenWrapper("caveflyer", 1, False, 3)

params = argparse.Namespace()

params.env_name = env.env
params.version = "v1"
params.DRY_RUN = False

params.actor_lr  = 1e-5
params.critic_lr = 3e-5

params.action_space = 15
params.observation_space_raw =  (64, 64, 9)
params.observation_space = (64, 64, 9)
params.encoding_size = 128

params.episodes = 20000
params.max_steps_per_episode = 200

params.discount_rate = 0.99

params.eps_decay_len = 1000
params.eps_min = 0.1

params.clip_ratio = 0.20
params.lam = 0.98

# params.curius_coef = 0.013
params.curius_coef = 0.0000003

params.batch_size = 4096
params.batch_size_curius = 512

params.train_interval = 10
params.iters = 200
params.iters_courious = 300

params.save_freq = 500
if args.resume is not None:
    params.resumed = 'resumed from: ' + args.resume
    # path has form of /path/to/model/episode so take last part of path
    BASE_EPISODE = int(os.path.basename(args.resume))
else:
    params.resumed = 'not resumed'
    BASE_EPISODE = 0

splash_screen(params)

def get_actor():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'/actor')
        print("loaded model from", args.resume)
        return model

    observation_input = tf.keras.Input(shape=params.observation_space, dtype=tf.uint8)
    x = tf.cast(observation_input, tf.float32)
    x = x / 255.0 # type: ignore
    x = tf.keras.layers.Conv2D(8, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, activation=tf.nn.elu)(x)
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

def get_curiosity_autoencoder():
    latent_dim = params.encoding_size

    if args.resume is not None:
        autoencoder = tf.keras.models.load_model(args.resume+'/autoencoder')
        inputs = autoencoder.encoder.input
        latent_output = autoencoder.encoder.output

        encoder = tf.keras.Model(inputs=inputs, outputs=latent_output[:, :latent_dim], name='encoder')
        print("loaded model from", args.resume)
        return encoder, autoencoder
    
    class CVAE(tf.keras.Model):
        def __init__(self, latent_dim):
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(64, 64, 9), dtype=tf.float32),
                    tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0),
                    tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(latent_dim * 2, name='latent_space'),
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=2*2*32, activation=tf.nn.relu),
                    tf.keras.layers.Reshape(target_shape=(2, 2, 32)),
                    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose(filters=9, kernel_size=2, strides=1, padding='same', activation='sigmoid'),
                ]
            )

        @tf.function
        def sample(self, eps=None):
            if eps is None:
                eps = tf.random.normal(shape=(100, self.latent_dim))
            return self.decode(eps, apply_sigmoid=True)

        def encode(self, x):
            mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
            return mean, logvar

        def reparameterize(self, mean, logvar):
            batch = tf.shape(mean)[0]
            dim = tf.shape(mean)[1]
            eps = tf.random.normal(shape=(batch, dim))
            return eps * tf.exp(logvar * .5) + mean

        def decode(self, z, apply_sigmoid=False):
            logits = self.decoder(z)
            if apply_sigmoid:
                probs = tf.sigmoid(logits)
                return probs
            return logits

        def call(self, inputs):
            mean, logvar = self.encode(inputs)
            z = self.reparameterize(mean, logvar)
            return self.decode(z)


    autoencoder = CVAE(latent_dim)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam())

    latent_output = autoencoder.encoder.get_layer('latent_space').output # 256 - 128 for mean and 128 for logvar
    encoder = tf.keras.Model(inputs=autoencoder.encoder.input, outputs=latent_output[:, :latent_dim], name='encoder')

    # Build the autoencoder model by calling it on a batch of data
    autoencoder(tf.random.normal([1, 64, 64, 9]))
    autoencoder.summary()
    encoder.summary()

    return encoder, autoencoder

def get_curiosity():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'/curiosity')
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

import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')

def log_curiosity_predicton(curiosity, encoder, autoencoder, memory: ppo.PPOReplayMemory, episode):
    try:
        # log the curiosity prediction as images to tensorboard
        states, actions_buffer, next_states  = memory.sample_curiosity(100)

        states = tf.cast(states, tf.float32)
        next_states = tf.cast(next_states, tf.float32)
        # autoencoder
        decoded_states = autoencoder(states) # 5x64x64x9 (3 images)
        # encoded states
        encoded_states = encoder(states) # 5x128 
        # predicted next states
        # onehot
        actions_buffer = tf.one_hot(actions_buffer, params.action_space) # 5x15
        predicted_next_states = curiosity([encoded_states, actions_buffer]) # 5x128
        # encoded true next_state
        encoded_next_states = encoder(next_states) # 5x128

        decoded_next_states = autoencoder(next_states) # 5x64x64x9

        # build one collage of images for each state
        # put the 3 frames together, then the 3 predicted frames together, then the 3 next frames together using cv2
        image = np.zeros((256, 192, 3), dtype=np.float32)
        

        states = tf.cast(states, tf.float32) / 255.0
        next_states = tf.cast(next_states, tf.float32) / 255.0

        buf = io.BytesIO()
        
        # plot states, next_states, decoded_states and decoded_next_states on 4x3 grid (plot just first image)
        for i in range(3):
            plt.subplot(4, 3, i+1)
            plt.imshow(states[0, :, :, i*3:i*3+3])
            plt.title(f'Normal {i}')

            plt.subplot(4, 3, i+4)
            plt.imshow(decoded_states[0, :, :, i*3:i*3+3])
            plt.title(f'Decoded {i}')

            plt.subplot(4, 3, i+7)
            plt.imshow(next_states[0, :, :, i*3:i*3+3])
            plt.title(f'Normal Next {i}')

            plt.subplot(4, 3, i+10)
            plt.imshow(decoded_next_states[0, :, :, i*3:i*3+3])
            plt.title(f'Decoded Next {i}')
        
        plt.savefig(buf, format='png')
        
        plt.close()
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        tf.summary.image('curiosity_prediction', image, step=episode)

        # calculate cosine similarity between predicted and true next states
        similarity = tf.keras.losses.cosine_similarity(encoded_next_states, predicted_next_states)
        similarity = tf.reduce_mean(similarity)

        tf.summary.scalar('curiosity_similarity', similarity, step=episode)

        # summary image
    except Exception as e:
        print(e)

class TimeLogger:
    def __init__(self, name):
        self.name = name
        self.times = deque(maxlen=10)

    def __enter__(self):
        self.start = tf.timestamp()

    def __exit__(self, *args):
        self.times.append(tf.timestamp() - self.start)

    def log(self, step):
        if len(self.times) > 0:
            tf.summary.scalar(self.name, np.mean(self.times), step=step)
    
def run():
    running_avg = deque(maxlen=200)

    memory = ppo.PPOReplayMemory(20_000, params.observation_space, gamma=params.discount_rate, lam=params.lam, gather_next_states=True)

    env_step = enviroments.make_tensorflow_env_step(env, lambda x: x) # type: ignore
    env_reset = enviroments.make_tensorflow_env_reset(env, lambda x: x) # type: ignore

    runner = get_curius_ppo_runner_2(env_step)
    runner = tf.function(runner)

    action_space = tf.constant(params.action_space, dtype=tf.int32)
    max_steps_per_episode = tf.constant(params.max_steps_per_episode, dtype=tf.int64)
    batch_size = tf.constant(params.batch_size, dtype=tf.int64)
    batch_size_curius = tf.constant(params.batch_size_curius, dtype=tf.int64)
    clip_ratio = tf.constant(params.clip_ratio, dtype=tf.float32)

    runner_logger = TimeLogger('runner_time')
    store_logger = TimeLogger('store_time')
    train_ppo_logger = TimeLogger('train_ppo_time')
    train_critic_logger = TimeLogger('train_critic_time')
    train_curiosity_logger = TimeLogger('train_curiosity_time')
    train_autoencoder_logger = TimeLogger('train_autoencoder_time')

    def log_timings(step):
        runner_logger.log(step)
        store_logger.log(step)
        train_ppo_logger.log(step)
        train_critic_logger.log(step)
        train_curiosity_logger.log(step)
        train_autoencoder_logger.log(step)

    t = tqdm.tqdm(range(BASE_EPISODE, params.episodes))
    for episode in t:
        initial_state = env_reset()

        initial_state = tf.constant(initial_state, dtype=tf.uint8)
        
        curius_coef = tf.constant(params.curius_coef, dtype=tf.float32)

        with runner_logger:
            (states, actions, rewards, values, log_probs, next_states), total_rewards, curiosity_mean, curiosity_std = runner(initial_state, actor, critic, curiosity, encoder, max_steps_per_episode, action_space, curius_coef) # type: ignore
        
        with store_logger:
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

        if episode % 10 == 0:
            log_timings(episode_tf)

        if len(memory) >= batch_size and int(episode) % params.train_interval == 0 and int(episode) > 0:
            stats = [] # kl, policy_loss, mean_ratio, mean_clipped_ratio, mean_advantage, mean_logprob
            with train_ppo_logger:
                for _ in range(params.iters):
                    batch = memory.sample(batch_size)
                    history = ppo.training_step_ppo(batch, actor, action_space, clip_ratio, policy_optimizer, episode_tf)
                    stats.append(history)

            log_stats(stats, episode_tf)

            with train_critic_logger:
                stats = []
                for _ in range(params.iters):
                    batch = memory.sample_critic(batch_size)
                    history = ppo.training_step_critic(batch, critic, value_optimizer, episode_tf)
                    stats.append(float(history))
                tf.summary.scalar('critic_loss', np.mean(stats), step=episode_tf)

            with train_curiosity_logger:
                stats = []
                for _ in range(params.iters_courious):
                    batch = memory.sample_encoded_curiosity(batch_size_curius, encoder)
                    history = ppo.training_step_curiosty(batch, curiosity, curiosity_optimizer, action_space,  episode_tf)
                    stats.append(float(history))
                tf.summary.scalar('curiosity_loss', np.mean(stats), step=episode_tf)
            
            with train_autoencoder_logger:
                stats = []
                for _ in range(params.iters_courious):
                    batch = memory.sample_autoencoder(batch_size_curius)
                    history = ppo.training_step_autoencoder(batch, autoencoder, autoencoder_optimizer, episode_tf)
                    stats.append(float(history))
                tf.summary.scalar('autoencoder_loss', np.mean(stats), step=episode_tf)

            # memory.reset()
        if int(episode+90) % 100 == 0 and int(episode) > 0:
            log_curiosity_predicton(curiosity, encoder, autoencoder, memory, episode_tf) 
        

        if episode % params.save_freq == 0 and episode > 0:
            NAME = f"{config.MODELS_DIR}{params.env_name}_{config.RUN_NAME}/{episode}/"
            
            actor.save(f"{NAME}actor", save_format="tf") # type: ignore
            critic.save(f"{NAME}critic", save_format="tf") # type: ignore
            curiosity.save(f"{NAME}curiosity", save_format="tf") # type: ignore
            autoencoder.save(f"{NAME}autoencoder", save_format="tf") # type: ignore

run()




