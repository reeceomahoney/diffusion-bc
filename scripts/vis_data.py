import numpy as np

data = np.load('data/expert_data/fwd_rand_init/data.npy', allow_pickle=True).item()
obs = data['obs']
roll = obs[:1000, :, 15]
roll = np.transpose(roll, (1, 0))

import matplotlib.pyplot as plt
plt.plot(roll)
plt.savefig('roll.png')