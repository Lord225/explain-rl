from typing import Any, Callable, Tuple
import gym3 
import numpy as np
import tensorflow as tf
from procgen import ProcgenGym3Env
from collections import deque
import cv2

# create environment overide the default seed

class ProcGenWrapper(gym3.Wrapper):
    UNIQUE_COLORS = np.array([(191, 255, 191), (191, 63, 191), (191, 255, 255), (127, 127, 255), (255, 127, 63), (0.0, 0.0, 0.0), (255, 191, 191), (255, 127, 127), (255, 191, 255), (127, 255, 127), (127, 63, 127), (63, 127, 127), (127, 63, 191), (127, 191, 63)], dtype=np.uint8)[np.newaxis]
    def __init__(self, env, num, return_segments, frame_stack_count, human=False):
        self.return_segments = return_segments
        self.num = num
        self.env = env 
        self.frame_stack_count = frame_stack_count
        self.human = human

    def reset(self):
        self.env_normal = ProcgenGym3Env(num=self.num, 
                            env_name=self.env, 
                            distribution_mode="easy",
                            use_backgrounds=False,
                            render_mode="rgb_array",
                            )
        self.env_mono = ProcgenGym3Env(num=self.num,
                            env_name=self.env, 
                            distribution_mode="easy", 
                            render_mode="rgb_array",
                            use_monochrome_assets=True,
                            use_backgrounds=False,
                            restrict_themes=True) 
        self.frame_stack = deque(maxlen=self.frame_stack_count)

        if self.return_segments:
            for _ in range(self.frame_stack_count):
                self.frame_stack.append(np.zeros((self.num, 64, 64, 4), dtype=np.uint8))
        else:
            for _ in range(self.frame_stack_count):
                self.frame_stack.append(np.zeros((self.num, 64, 64, 3), dtype=np.uint8))

        states = self.env_normal.callmethod("get_state")
        self.env_mono.callmethod("set_state", states)

        self.render()

        if self.human:
            # set size of the window to be 4x the size of the frame
            cv2.namedWindow("Normal", cv2.WINDOW_NORMAL)
        
        return np.concatenate(self.frame_stack, axis=3, dtype=np.uint8)
    
    def render(self):
        if self.human:
            obs = self.observe()
            # render frames using cv2
            frame = obs[0][0, :, :, :3]
            # resize
            frame = cv2.resize(frame, (frame.shape[1]*8, frame.shape[0]*8), interpolation=cv2.INTER_NEAREST)
            return frame
    
    def observe(self):
        rew_normal, obs_normal, first_normal = self.env_normal.observe()
        if self.return_segments:
            _, obs_mono, _ = self.env_mono.observe()

        if self.return_segments:
            mono = obs_mono['rgb']
            mono = np.argmax((mono.reshape(-1, 1, 3) == ProcGenWrapper.UNIQUE_COLORS).all(axis=2), axis=1).reshape(mono.shape[0], 64, 64, 1).astype(np.uint8)
            obs = np.concatenate([obs_normal['rgb'], mono], axis=3, dtype=np.uint8)
        else:
            obs = obs_normal['rgb']
        
        return obs, rew_normal, first_normal
    
              
    def step(self, action):
        if self.return_segments:
            self.env_mono.act(action)

        self.env_normal.act(action)

        raw_obs, rew, first = self.observe()

        self.frame_stack.append(raw_obs)

        return np.concatenate(self.frame_stack, axis=3, dtype=np.uint8), rew, first
    
    
ObservationTransformer = Callable[[Any], np.ndarray]

def make_tensorflow_env_step(env: ProcGenWrapper, observation_transformer: ObservationTransformer)-> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    def step(action):
        state, reward, done = env.step(np.array([action]))
        return (np.array(state, np.uint8), np.array(reward, np.float32), np.array(done, np.int32))

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32)] # type: ignore
    )
    def tf_env_step(action):
        return tf.numpy_function(step, [action], (tf.uint8, tf.float32, tf.int32))
    return tf_env_step # type: ignore
    

def make_tensorflow_env_reset(env: ProcGenWrapper, observation_transformer: ObservationTransformer) -> Callable[[], tf.Tensor]:
    def reset():
        state = env.reset()
        return state
    
    @tf.function
    def tf_env_reset():
        return tf.numpy_function(reset, [], tf.uint8)  # type: ignore
    
    return tf_env_reset # type: ignore

if __name__ == "__main__":
    import random
    env = ProcGenWrapper("caveflyer", num=2, return_segments=True, frame_stack_count=2)
    env.reset()
    # action = random.randint(0, 15)
    # rew, obs, first = env.step(np.array([action]*10))
    # print(obs.shape)
    # print(rew)
    # print(first)
    # # plot the first frame of the first batch on two subplots
    # import matplotlib.pyplot as plt
    # plt.subplot(2, 2, 1)
    # plt.imshow(obs[0, :, :, 0:3]/255.0)
    # plt.title('Normal 1')
    # plt.subplot(2, 2, 2)
    # plt.imshow(obs[0, :, :, 3:6]/255.0)
    # plt.title('Monochrome 1')
    # plt.subplot(2, 2, 3)
    # plt.imshow(obs[0, :, :, 6:9]/255.0)
    # plt.title('Normal 2')
    # plt.subplot(2, 2, 4)
    # plt.imshow(obs[0, :, :, 9:12]/255.0)
    # plt.title('Monochrome 2')
    # plt.show()
    unque = set()
    for i in range(1024):
        actions = np.random.randint(0, 15, size=(2,))
    
        rew, obs, first = env.step(actions)
        print(rew)
        print(obs.shape)
        print(first)
        # plot the first frame of the first batch on two subplots
        import matplotlib.pyplot as plt
        plt.subplot(2, 2, 1)
        plt.imshow(obs[0, :, :, 4:7]/255.0)
        plt.title('Normal 1')
        plt.subplot(2, 2, 2)
        plt.imshow(obs[0, :, :, 7])
        plt.title('Monochrome 1')
        plt.show()


    print(f"Unique colors in mono layer {unque}")
            

