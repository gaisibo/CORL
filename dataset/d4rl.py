import torch
import numpy as np
from d3rlpy.datasets import get_d4rl
from utils.k_means import kmeans


def cut_antmaze(name, start_point_index, end_point_index, directed=True, task_num=None):
    assert 'play' in name
    if 'large' in name and task_num is None:
        task_num = 7
    elif 'medium' in name and task_num is None:
        task_num = 3
    dataset, env = get_d4rl(name)
    end_point = []
    for episode in dataset.episodes:
        end_point.append(episode[-1].observation)
        print(episode[-1].observation[0])
    assert False
    end_point = np.array(end_point)
    end_point = torch.from_numpy(end_point).half().cuda()

    # kmeans
    torch.cuda.synchronize()
    codebook, distortion = kmeans(end_point, task_num, batch_size=6400000, iter=1)
    torch.cuda.synchronize()
