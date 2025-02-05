import random
from gym3 import types_np
from procgen import ProcgenGym3Env
import matplotlib.pyplot as plt
import numpy as np

seed = random.randint(0, 2 ** 31 - 1)

env_normal = ProcgenGym3Env(num=1, 
                            env_name="caveflyer", 
                            distribution_mode="easy", 
                            render_mode="rgb_array",)

env_mono = ProcgenGym3Env(num=1, 
                          env_name="caveflyer", 
                          distribution_mode="easy", 
                          render_mode="rgb_array",
                          num_levels=1,
                          start_level=seed,
                          use_monochrome_assets=True,
                          use_backgrounds=False,
                          restrict_themes=True)

states = env_normal.callmethod("get_state")
env_mono.callmethod("set_state", states)


step = 0
while True:
    action = types_np.sample(env_mono.ac_space, bshape=(env_mono.num,))    
    env_mono.act(action)
    env_normal.act(action)
    rew, obs, first = env_mono.observe()
    rew2, obs2, first2 = env_normal.observe()
    
    print(obs)
    plt.subplot(1, 2, 1)
    plt.imshow(obs['rgb'][0])
    plt.title('Monochrome')

    print(obs2)
    plt.subplot(1, 2, 2)
    plt.imshow(obs2['rgb'][0])
    plt.title('Normal')

    plt.show()
    
    
    print(f"step {step} reward {rew} first {first}")
    if step > 0 and first:
        break
    step += 1