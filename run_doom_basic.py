import numpy as np
from vizdoombasic import VizdoomBasic


env = VizdoomBasic()
env.reset()
while True:
    action = np.random.randint(0, 3)
    env.step(action)
