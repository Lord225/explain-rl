import gym3 
import numpy as np
from procgen import ProcgenGym3Env
from collections import deque

# create environment overide the default seed

class ProcGenWrapper(gym3.Wrapper):
    def __init__(self, env, num, return_segments, frame_stack_count):
        self.return_segments = return_segments
        self.num = num
        self.env = env 
        self.frame_stack_count = frame_stack_count

    def reset(self):
        self.env_normal = ProcgenGym3Env(num=self.num, 
                            env_name=self.env, 
                            distribution_mode="easy", 
                            render_mode="rgb_array")
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
                self.frame_stack.append(np.zeros((self.num, 64, 64, 6)))
        else:
            for _ in range(self.frame_stack_count):
                self.frame_stack.append(np.zeros((self.num, 64, 64, 3)))

        states = self.env_normal.callmethod("get_state")
        self.env_mono.callmethod("set_state", states)
              
    def step(self, action):
        if self.return_segments:
            self.env_mono.act(action)
            _, obs_mono, _ = self.env_mono.observe()

        self.env_normal.act(action)
        rew_normal, obs_normal, first_normal = self.env_normal.observe()

        if self.return_segments:
            obs = np.concatenate([obs_normal['rgb'], obs_mono['rgb']], axis=3)
        else:
            obs = obs_normal['rgb']

        self.frame_stack.append(obs)
        
        return rew_normal, np.concatenate(self.frame_stack, axis=3), first_normal
    

if __name__ == "__main__":
    import random
    env = ProcGenWrapper("caveflyer", num=10, return_segments=True, frame_stack_count=2)
    env.reset()
    action = random.randint(0, 15)
    rew, obs, first = env.step(np.array([action]*10))
    print(obs.shape)
    print(rew)
    print(first)
    # plot the first frame of the first batch on two subplots
    import matplotlib.pyplot as plt
    plt.subplot(2, 2, 1)
    plt.imshow(obs[0, :, :, 0:3]/255.0)
    plt.title('Normal 1')
    plt.subplot(2, 2, 2)
    plt.imshow(obs[0, :, :, 3:6]/255.0)
    plt.title('Monochrome 1')
    plt.subplot(2, 2, 3)
    plt.imshow(obs[0, :, :, 6:9]/255.0)
    plt.title('Normal 2')
    plt.subplot(2, 2, 4)
    plt.imshow(obs[0, :, :, 9:12]/255.0)
    plt.title('Monochrome 2')
    plt.show()
