import os
import numpy as np
import zarr


# load dataset
curr_dir = os.path.dirname(os.path.realpath(__file__))
dataset = 'fwd_50'
data = np.load(curr_dir + '/' + dataset + '/raw_data.npy', allow_pickle=True).item()

# process data
obs_dim = 33
data["obs"] = data.pop("observations")[..., :obs_dim]
data["action"] = data.pop("actions")
action_mean = np.array([-0.089, 0.712, -1.03, 0.089, 0.712, -1.03, -0.089, 
-0.712, 1.03, 0.089, -0.712, 1.03])
data["action"] -= action_mean
data["terminals"][:, -1] = 1

# dims = [1, obs_dim, 12]
# for k, v, dim in zip(data.keys(), data.values(), dims):
#     data[k] = v.reshape(-1, dim).astype(np.float32)

episode_ends = np.where(data["terminals"])[0] + 1
del data["terminals"]

for k, v in data.items():
    print(k, v.shape)
print('episode ends', episode_ends.shape)

np.save(curr_dir + '/' + dataset + '/data.npy', data, allow_pickle=True)
