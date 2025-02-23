from typing import Callable, Tuple
import tensorflow as tf
from .common import PPOReplayHistoryCuriosityType, ParPPOReplayHistoryCuriosityType
from .ppo import logprobabilities

def get_curius_ppo_runner(tf_env_step: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]):
    #@tf.function(reduce_retracing=True)
    def run_episode(
            initial_state: tf.Tensor,
            actor: tf.keras.Model,
            critic: tf.keras.Model,
            curiosity: tf.keras.Model,
            max_steps: int,
            env_actions: int,
            curius_coef: float,
            ):
        states = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        next_states = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)

        curiosities = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps, dtype=tf.int32):

            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t = actor(state) # type: ignore

            value_t = critic(state) # type: ignore

            action = tf.squeeze(tf.random.categorical(action_logits_t, 1), axis=1)
            action = tf.cast(action, tf.int32)
            action = tf.squeeze(action)

            next_state, reward, done = tf_env_step(action)

            next_state.set_shape(initial_state_shape)
            
            action_one_hot = tf.one_hot(action, env_actions)
            action_input_processed = tf.expand_dims(action_one_hot, axis=0)
            # calculate curiosity
            predicted_state = curiosity([state, action_input_processed]) # type: ignore
            predicted_state = tf.squeeze(predicted_state, axis=0)
            predicted_state.set_shape(initial_state_shape) # type: ignore
            curiosity_reward = tf.reduce_sum(tf.square(predicted_state - next_state)) # type: ignore

            reward = reward + curius_coef * curiosity_reward # type: ignore
            
            log_prob = logprobabilities(action_logits_t, action, env_actions)
            
            # store results
            curiosities = curiosities.write(t, curiosity_reward)
            states = states.write(t,  tf.squeeze(state))
            next_states = next_states.write(t, tf.squeeze(next_state))
            actions = actions.write(t, action)
            rewards = rewards.write(t, reward)
            values = values.write(t, tf.squeeze(value_t))
            log_probs = log_probs.write(t, tf.squeeze(log_prob))

            state = next_state

            if tf.cast(done, tf.bool): # type: ignore
                break

        states = states.stack()
        actions = actions.stack()
        rewards = rewards.stack()
        values = values.stack()
        log_probs = log_probs.stack()
        next_states = next_states.stack()
        curiosities = curiosities.stack()

        return PPOReplayHistoryCuriosityType(states, actions, rewards, values, log_probs, next_states), tf.reduce_sum(rewards), tf.reduce_mean(curiosities),  tf.math.reduce_std(curiosities)
                
    return run_episode

#! ASSUME num = 1
def get_curius_ppo_runner_2(tf_env_step: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]):
    #@tf.function(reduce_retracing=True)
    def run_episode(
            initial_state: tf.Tensor,
            actor: tf.keras.Model,
            critic: tf.keras.Model,
            curiosity: tf.keras.Model,
            encoder: tf.keras.Model,
            max_steps: int,
            env_actions: int,
            curius_coef: float,
            epsilon: float,
            ):
        states = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        next_states = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)

        curiosities = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps, dtype=tf.int32):
            # Run the model and to get action probabilities and critic value
            action_logits_t = actor(state) # type: ignore

            value_t = critic(state) # type: ignore
            
            action = tf.cond(
                tf.random.uniform(()) < epsilon,
                lambda: tf.squeeze(tf.random.categorical(action_logits_t, 1), axis=1),
                lambda: tf.argmax(action_logits_t, axis=1)
            )
            action = tf.cast(action, tf.int32)
            action = tf.squeeze(action)

            next_state, reward, done = tf_env_step(action)

            next_state.set_shape(initial_state_shape)
            
            action_one_hot = tf.one_hot(action, env_actions)
            action_input_processed = tf.expand_dims(action_one_hot, axis=0)
            
            # calculate curiosity
            encoded_state = encoder(state)
            encoded_next_state = encoder(next_state)
            encoded_predicted_state = curiosity([encoded_state, action_input_processed])
            encoded_predicted_state = tf.squeeze(encoded_predicted_state, axis=0)

            curiosity_reward = tf.reduce_sum(tf.square(encoded_next_state - encoded_predicted_state))
            
            reward = reward + curius_coef * curiosity_reward # type: ignore
            
            log_prob = logprobabilities(action_logits_t, action, env_actions)
            
            # store results
            curiosities = curiosities.write(t, curiosity_reward)
            states = states.write(t,  tf.squeeze(state))
            next_states = next_states.write(t, tf.squeeze(next_state))
            actions = actions.write(t, tf.squeeze(action))
            rewards = rewards.write(t, tf.squeeze(reward))
            values = values.write(t, tf.squeeze(value_t))
            log_probs = log_probs.write(t, tf.squeeze(log_prob))

            state = next_state

            if tf.cast(done, tf.bool): # type: ignore
                break

        states = states.stack()
        actions = actions.stack()
        rewards = rewards.stack()
        values = values.stack()
        log_probs = log_probs.stack()
        next_states = next_states.stack()
        curiosities = curiosities.stack()

        return PPOReplayHistoryCuriosityType(states, actions, rewards, values, log_probs, next_states), tf.reduce_sum(rewards), tf.reduce_mean(curiosities),  tf.math.reduce_std(curiosities)
                
    return run_episode

@tf.function(reduce_retracing=True)
def get_exp_from_buffer(
        observations: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        curiosity: tf.keras.Model,
        encoder: tf.keras.Model,
        env_actions: int,
        curius_coef: float,
        ):
    max_steps = tf.shape(actions)[0] - 1
    
    states = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    actions_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    curiosities = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    for t in tf.range(max_steps, dtype=tf.int32):
        state = observations[t]
        action = actions[t]
        reward = rewards[t]
        next_state = observations[t + 1]

        # Expand to batch dimension for model input
        state_expanded = tf.expand_dims(state, axis=0)
        
        # Compute value from critic
        value_t = critic(state_expanded)
        value_t = tf.squeeze(value_t)
        
        # Compute action logits from actor for log_probs
        action_logits_t = actor(state_expanded)
        action_logits_t = tf.squeeze(action_logits_t, axis=0)
        
        # Compute log probability of the action from buffer
        log_prob = logprobabilities(action_logits_t, action, env_actions)
        
        # Prepare action for curiosity model
        action_one_hot = tf.one_hot(action, env_actions)
        action_input_processed = tf.reshape(action_one_hot, (1, env_actions))
        
        # Encode current and next state
        encoded_state = encoder(state_expanded)
        next_state_expanded = tf.expand_dims(next_state, axis=0)
        encoded_next_state = encoder(next_state_expanded)
        
        # Predict next state with curiosity model
        encoded_predicted_state = curiosity([encoded_state, action_input_processed])
        encoded_predicted_state = tf.squeeze(encoded_predicted_state, axis=0)
        
        # Calculate curiosity reward
        curiosity_reward = tf.reduce_sum(tf.square(encoded_next_state - encoded_predicted_state))
        total_reward = reward + curius_coef * curiosity_reward
        
        # Store in TensorArrays
        states = states.write(t, state)
        actions_ta = actions_ta.write(t, action)
        rewards_ta = rewards_ta.write(t, total_reward)
        values = values.write(t, value_t)
        log_probs = log_probs.write(t, log_prob)
        next_states = next_states.write(t, next_state)
        curiosities = curiosities.write(t, curiosity_reward)

    # Convert TensorArrays to tensors
    states = states.stack()
    actions_ta = actions_ta.stack()
    rewards_ta = rewards_ta.stack()
    values = values.stack()
    log_probs = log_probs.stack()
    next_states = next_states.stack()
    curiosities = curiosities.stack()

    return (
        PPOReplayHistoryCuriosityType(states, actions_ta, rewards_ta, values, log_probs, next_states),
        tf.reduce_sum(rewards_ta),
        tf.reduce_mean(curiosities),
        tf.math.reduce_std(curiosities)
    )


def get_curius_ppo_runner_paraller(tf_env_step: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]):
    #@tf.function(reduce_retracing=True)
    def run_episode(
            initial_state: tf.Tensor,
            actor: tf.keras.Model,
            critic: tf.keras.Model,
            curiosity: tf.keras.Model,
            encoder: tf.keras.Model,
            max_steps: int,
            env_actions: int,
            curius_coef: float,
            ):
        # store on cpu
        states = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        next_states = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
        dones = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        curiosities = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps, dtype=tf.int32):
            # Run the model and to get action probabilities and critic value
            action_logits_t = actor(state) # type: ignore

            value_t = critic(state) # type: ignore
            
            action = tf.squeeze(tf.random.categorical(action_logits_t, 1), axis=1)
            action = tf.cast(action, tf.int32)
            action = tf.squeeze(action)

            next_state, reward, done = tf_env_step(action)

            next_state.set_shape(initial_state_shape)
            
            action_one_hot = tf.one_hot(action, env_actions)
            
            # calculate curiosity
            encoded_state = encoder(state)
            encoded_next_state = encoder(next_state)
            encoded_predicted_state = curiosity([encoded_state, action_one_hot])
            curiosity_reward = tf.reduce_sum(tf.square(encoded_next_state - encoded_predicted_state), axis=1)
            
            reward = reward + curius_coef * curiosity_reward # type: ignore
            
            log_prob = logprobabilities(action_logits_t, action, env_actions)
            
            # store results
            curiosities = curiosities.write(t, curiosity_reward)
            states = states.write(t,  tf.squeeze(state))
            next_states = next_states.write(t, tf.squeeze(next_state))
            actions = actions.write(t, tf.squeeze(action))
            rewards = rewards.write(t, tf.squeeze(reward))
            values = values.write(t, tf.squeeze(value_t))
            log_probs = log_probs.write(t, tf.squeeze(log_prob))
            dones = dones.write(t, tf.squeeze(done))

            state = next_state

        states = states.stack()
        actions = actions.stack()
        rewards = rewards.stack()
        values = values.stack()
        log_probs = log_probs.stack()
        next_states = next_states.stack()
        curiosities = curiosities.stack()
        dones = dones.stack()

        return ParPPOReplayHistoryCuriosityType(states, actions, rewards, values, log_probs, next_states, dones), tf.reduce_sum(rewards, axis=0), tf.reduce_mean(curiosities, axis=0),  tf.math.reduce_std(curiosities, axis=0)
                
    return run_episode