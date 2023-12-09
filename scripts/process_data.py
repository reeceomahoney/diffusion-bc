import os
import click
import numpy as np


@click.command()
@click.option('-d', '--dataset', required=True)
@click.option('-o', '--output', required=True)
@click.option('-e', '--eps', default=1000)
def main(dataset, output, eps):
    # load dataset
    data_dir = os.path.dirname(os.path.realpath(__file__)) + '/../data/expert_data/'
    data = np.load(data_dir + 'raw_data/' + dataset + '.npy', allow_pickle=True).item()

    # process data
    obs_dim = 33
    data["obs"] = data.pop("observations")[..., :obs_dim]
    data["action"] = data.pop("actions")
    action_mean = np.array([-0.089, 0.712, -1.03, 0.089, 0.712, -1.03, -0.089, 
    -0.712, 1.03, 0.089, -0.712, 1.03])
    data["action"] -= action_mean
    data["terminals"][:, -1] = 1

    episode_ends = np.where(data["terminals"])[0] + 1
    del data["terminals"]

    # only use the first 250 time steps
    data["obs"] = data["obs"][:eps, :250]
    data["action"] = data["action"][:eps, :250]
    episode_ends = episode_ends[:eps]

    for k, v in data.items():
        print(k, v.shape)
    print('episode ends', episode_ends.shape)

    output_path = data_dir + 'processed_data/' + output + '.npy'
    print('saving to', output_path)
    np.save(output_path, data, allow_pickle=True)


if __name__ == "__main__":
    main()