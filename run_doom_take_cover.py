import numpy as np
from vizdoom_take_cover import VizdoomTakeCover
import os
import sys
env = VizdoomTakeCover()
NUM_EPISODES = 100
MAX_STEPS = 300
OUT_DIR = 'VAE_take_cover_train_data_npz'
t = 0
total_step = 0
start_eps = int(sys.argv[1])
end_eps = int(sys.argv[2])

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

for eps in range(start_eps, end_eps):

    obs = []
    acs = []
    termination = []
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
        termination.append(1 if d else 0)
        # np.save(os.path.join(OUT_DIR, f'{eps}_{t}_obs'), o)

        if d:
            break


    obs = np.array(obs, dtype=np.uint8)
    acs = np.array(acs, dtype=np.uint8)
    termination = np.array(termination, dtype=np.uint8)

    # np.savez_compressed(os.path.join(OUT_DIR, f'eps_{eps}]'), obs=obs, action=acs, termination=termination)

    total_step += 1


print(r, d)