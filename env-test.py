
from gym3 import types_np
from procgen import ProcgenGym3Env
import matplotlib.pyplot as plt

# human rendering
env = ProcgenGym3Env(num=1, 
                     env_name="coinrun", 
                     distribution_mode="easy", 
                     num_levels=0, 
                     start_level=0, 
                     render_mode="rgb_array",
                     use_generated_assets=True,
                     use_monochrome_assets=True,
                     use_backgrounds=False,
                     restrict_themes=True)
step = 0
while True:
    env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
    rew, obs, first = env.observe()
    print(obs['rgb'].shape)
    plt.imshow(obs['rgb'][0])
    plt.show()
    
    print(f"step {step} reward {rew} first {first}")
    if step > 0 and first:
        break
    step += 1