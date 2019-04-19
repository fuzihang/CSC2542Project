import numpy as np
from vizdoom_take_cover import VizdoomTakeCover
import os
env = VizdoomTakeCover()
NUM_EPISODES = 200
MAX_STEPS = 300
OUT_DIR = 'VAE_take_cover_valid_data'
t = 0
total_step = 0

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

for eps in range(NUM_EPISODES):

    obs = []
    acs = []
    o = env.reset()
    obs.append(o)


    old_total_step = total_step
    t = 0

    np.save(os.path.join(OUT_DIR, f'{eps}_{t}_obs'), o)


    while t < MAX_STEPS:
        action = np.random.randint(0, 2)
        o, r, d, _ = env.step(action)

        t += 1
        total_step += 1

        acs.append(action)
        obs.append(o)
        np.save(os.path.join(OUT_DIR, f'{eps}_{t}_obs'), o)

        if d:
            break


    # obs = np.array(obs, dtype=np.uint8)
    # acs = np.array(acs, dtype=np.uint8)
    #
    # np.savez_compressed(os.path.join(OUT_DIR, f'eps_{eps}_[{old_total_step},{total_step}]'), obs=obs, action=acs)

    total_step += 1


print(r, d)