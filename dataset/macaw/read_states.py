import h5py
import matplotlib.pyplot as plt


if __name__ == '__main__':
    hfile = h5py.File('sac_ant_dir_0/expert.hdf5')
    observations = hfile['observations']
    terminals = hfile['terminals']
    xs = []; ys = []
    for observation, terminal in zip(observations, terminals):
        if terminal:
            plt.plot(xs, ys)
            xs = []; ys = []
        else:
            xs.append(observation[0]); ys.append(observation[1])
    plt.savefig('fig/ant_dir_0_expert.png')
