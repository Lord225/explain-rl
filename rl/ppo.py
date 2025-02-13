import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from .common import HistorySampleType, HistorySampleCriticType, HistorySampleCuriosityType

@tf.function
def discounted_cumulative_sums_tf(x, discount_rate)-> tf.Tensor:
    size = tf.shape(x)[0]
    x = tf.reverse(x, axis=[0])
    buffer = tf.TensorArray(dtype=tf.float32, size=size)

    discounted_sum = tf.constant(0.0, dtype=tf.float32)

    for i in tf.range(size):
        discounted_sum = x[i] + discount_rate * discounted_sum # type: ignore
        buffer = buffer.write(i, discounted_sum)
    
    return tf.reverse(buffer.stack(), axis=[0]) # type: ignore

class PPOReplayMemory:
    def __init__(self, max_size, state_shape, next_state_shape=None, gamma=0.99, lam=0.95, gather_next_states=False):
        states_shape = (max_size,) + state_shape
        if next_state_shape is None:
            next_state_shape = states_shape
        
        with tf.device('/CPU:0'):
            self.states_buffer = tf.Variable(tf.zeros(states_shape,  dtype=tf.uint8), trainable=False, dtype=tf.uint8)
            self.advantages_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
            self.actions_buffer = tf.Variable(tf.zeros((max_size), dtype=tf.int32), trainable=False, dtype=tf.int32)
            self.rewards_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
            self.return_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
            self.logprobability_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)

        if gather_next_states:
            self.next_states_buffer = tf.Variable(tf.zeros(next_state_shape, dtype=tf.uint8), trainable=False, dtype=tf.uint8)
        else:
            self.next_states_buffer = None

        self.gamma = gamma
        self.lam = lam
        self.max_size = max_size
        self.count = 0
        self.real_size = 0

    def reset(self):
        self.real_size = 0
        self.count = 0

    
    @tf.function(reduce_retracing=True)
    def add_tf(self, states, actions, rewards, values, logprobabilities, next_states,
               states_buffer, advantages_buffer, actions_buffer, rewards_buffer, return_buffer, logprobability_buffer, next_states_buffer,
               gamma, lam, max_size, count):
        
        size = len(states)
        indices = tf.range(count, count + size) % max_size
        
        states_buffer = tf.tensor_scatter_nd_update(states_buffer, indices[:, None], states)
        actions_buffer = tf.tensor_scatter_nd_update(actions_buffer, indices[:, None], actions)
        rewards_buffer = tf.tensor_scatter_nd_update(rewards_buffer, indices[:, None], rewards)
        logprobability_buffer = tf.tensor_scatter_nd_update(logprobability_buffer, indices[:, None], logprobabilities)

        if next_states_buffer is not None:
            next_states_buffer = tf.tensor_scatter_nd_update(next_states_buffer, indices[:, None], next_states)
        
        count = (count + size) % max_size

        # finish trajectory
        rewards = tf.concat([rewards, [0.0]], axis=0)
        values = tf.concat([values, [0.0]], axis=0)

        deltas = rewards[:-1] + gamma * values[1:] - values[:-1] # type: ignore

        advantages = discounted_cumulative_sums_tf(
            deltas, gamma * lam
        )
        returns = discounted_cumulative_sums_tf(
            rewards, gamma
        )[:-1] # type: ignore

        advantages_buffer = tf.tensor_scatter_nd_update(advantages_buffer, indices[:, None], advantages)
        return_buffer = tf.tensor_scatter_nd_update(return_buffer, indices[:, None], returns)

        return states_buffer, advantages_buffer, actions_buffer, rewards_buffer, return_buffer, logprobability_buffer, next_states_buffer, count
    
    #@tf.function
    def add_multiple_tf(self, states, actions, rewards, values, logprobabilities, next_states,
                        states_buffer, advantages_buffer, actions_buffer, rewards_buffer, return_buffer, logprobability_buffer, next_states_buffer, dones,
                        gamma, lam, max_size, count):
        # in dones buffer we have 1 for every new trajectory start. We need to add each of them as a separate trajectory using add_tf function so we need to split them here and then add each of them separately.
        ones_idx = tf.where(dones == 1)
        ones_idx = tf.concat([ones_idx, [[len(dones)]]], axis=0)

        for i in range(len(ones_idx) - 1):
            start = ones_idx[i][0]
            end = ones_idx[i+1][0]
            states_buffer, advantages_buffer, actions_buffer, rewards_buffer, return_buffer, logprobability_buffer, next_states_buffer, count = self.add_tf(
                states[start:end], 
                actions[start:end], 
                rewards[start:end], 
                values[start:end], 
                logprobabilities[start:end],
                next_states[start:end],
                states_buffer, 
                advantages_buffer, 
                actions_buffer, 
                rewards_buffer, 
                return_buffer, 
                logprobability_buffer,
                next_states_buffer,
                gamma, 
                lam, 
                max_size, 
                count)
        

    def add(self, observations, actions, rewards, values, logprobabilities, next_states):
        count = tf.constant(self.count, dtype=tf.int32)
        max_size = tf.constant(self.max_size, dtype=tf.int32)
        gamma = tf.constant(self.gamma, dtype=tf.float32)
        lam = tf.constant(self.lam, dtype=tf.float32)

        states_buffer, advantages_buffer, actions_buffer, rewards_buffer, return_buffer, logprobability_buffer, next_states_buffer, new_count = self.add_tf(
                    observations, 
                    actions, 
                    rewards, 
                    values, 
                    logprobabilities,
                    next_states,
                    self.states_buffer, 
                    self.advantages_buffer, 
                    self.actions_buffer, 
                    self.rewards_buffer, 
                    self.return_buffer, 
                    self.logprobability_buffer,
                    self.next_states_buffer,
                    gamma, 
                    lam, 
                    max_size, 
                    count) # type: ignore
        
        self.states_buffer = states_buffer
        self.advantages_buffer = advantages_buffer
        self.actions_buffer = actions_buffer
        self.rewards_buffer = rewards_buffer
        self.return_buffer = return_buffer
        self.logprobability_buffer = logprobability_buffer
        self.next_states_buffer = next_states_buffer
        
        self.count = int(new_count) # type: ignore
        self.real_size = min(self.real_size + len(observations), self.max_size)
    

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        advantages = tf.gather(self.advantages_buffer, indices)
        advantages = (advantages - tf.math.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-7)

        return HistorySampleType(
            tf.gather(self.states_buffer, indices),
            tf.gather(self.actions_buffer, indices),
            tf.gather(self.logprobability_buffer, indices),
            advantages,
        )


    def sample_critic(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        return HistorySampleCriticType(
            tf.gather(self.states_buffer, indices),
            tf.gather(self.return_buffer, indices),
        )
    
    def sample_curiosity(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"
        assert self.next_states_buffer is not None, "next_states_buffer is not gathered"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        return HistorySampleCuriosityType(
            tf.gather(self.states_buffer, indices),
            tf.gather(self.actions_buffer, indices),
            tf.gather(self.next_states_buffer, indices),
        )
    
    def sample_encoded_curiosity(self, batch_size, encoder):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"
        assert self.next_states_buffer is not None, "next_states_buffer is not gathered"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        return HistorySampleCuriosityType(
            encoder(tf.gather(self.states_buffer, indices)),
            tf.gather(self.actions_buffer, indices),
            encoder(tf.gather(self.next_states_buffer, indices)),
        )

    def sample_autoencoder(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        return tf.gather(self.states_buffer, indices)
         
    def __len__(self) -> int:
        return self.real_size
    

@tf.function
def logprobabilities(logits, actions, num_actions):
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(tf.one_hot(actions, num_actions) * logprobabilities_all, axis=1)
    return logprobability

@tf.function
def clip(values, clip_ratio):
    return tf.minimum(tf.maximum(values, 1-clip_ratio), 1+clip_ratio)

@tf.function
def training_step_ppo(batch,
                      actor,
                      num_of_actions,
                      clip_ratio,
                      optimizer: tf.keras.optimizers.Optimizer,
                      step: int
                      ):
    
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer = batch
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        logits = actor(observation_buffer)
        
        ratio = tf.exp(
            logprobabilities(logits, action_buffer, num_of_actions)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(logprobability_buffer - logprobabilities(logits, action_buffer, num_of_actions))
    kl = tf.reduce_mean(kl)

    # tf.summary.scalar('kl', kl, step=step) # type: ignore
    # tf.summary.scalar('loss', policy_loss, step=step)
    # tf.summary.scalar('mean_ratio', tf.reduce_mean(ratio), step=step)
    # tf.summary.scalar('mean_clipped_ratio', tf.reduce_mean(min_advantage), step=step)
    # tf.summary.scalar('mean_advantage', tf.reduce_mean(advantage_buffer), step=step)
    # tf.summary.scalar('mean_logprob', tf.reduce_mean(logprobability_buffer), step=step)

    mean_ratio = tf.reduce_mean(ratio)
    mean_clipped_ratio = tf.reduce_mean(min_advantage)
    mean_advantage = tf.reduce_mean(advantage_buffer)
    mean_logprob = tf.reduce_mean(logprobability_buffer)

    return kl, policy_loss, mean_ratio, mean_clipped_ratio, mean_advantage, mean_logprob



@tf.function
def training_step_critic(
        batch,
        critic,
        optimizer: tf.keras.optimizers.Optimizer,
        step: int
):
    observation_buffer, target_buffer = batch
    with tf.GradientTape() as tape:
        values = critic(observation_buffer)
        loss = tf.reduce_mean(tf.square(target_buffer - values))

    gradients = tape.gradient(loss, critic.trainable_variables)
    optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

    return tf.reduce_mean(loss)


@tf.function
def training_step_curiosty(
        batch,
        curiosity,
        optimizer: tf.keras.optimizers.Optimizer,
        num_of_actions,
        step: int
):
    observation_buffer, action_buffer, next_observation_buffer = batch

    action_buffer = tf.one_hot(action_buffer, num_of_actions, dtype=tf.float32)

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(next_observation_buffer - curiosity([observation_buffer, action_buffer])))

    gradients = tape.gradient(loss, curiosity.trainable_variables)
    optimizer.apply_gradients(zip(gradients, curiosity.trainable_variables))

    return tf.reduce_mean(loss)


@tf.function
def training_step_autoencoder(
    batch,
    autoencoder,
    optimizer: tf.keras.optimizers.Optimizer,
    step: int
):
    observation_buffer = tf.cast(batch, tf.float32)
    observation_buffer_cast = tf.cast(batch, tf.float32) / 255.0

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(observation_buffer_cast - autoencoder(observation_buffer)))

    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

    return tf.reduce_mean(loss)