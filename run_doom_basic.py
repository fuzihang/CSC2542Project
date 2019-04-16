import numpy as np
from vizdoombasic import VizdoomBasic
import os
env = VizdoomBasic()
NUM_EPISODES = 10
MAX_STEPS = 3000
OUT_DIR = 'VAE_train_data'
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

    while t < MAX_STEPS:
        action = np.random.randint(0, 3)
        o, r, d, _ = env.step(action)

        t += 1
        total_step += 1

        acs.append(action)
        obs.append(o)

        if d:
            break


    obs = np.array(obs, dtype=np.uint8)
    acs = np.array(acs, dtype=np.uint8)

    np.savez_compressed(os.path.join(OUT_DIR, f'eps_{eps}_[{old_total_step},{total_step}]'), obs=obs, action=acs)

    total_step += 1


print(r, d)