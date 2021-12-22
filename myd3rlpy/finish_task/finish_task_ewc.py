import torch
from torch.utils.dataset import DataLoader
import numpy as np
from myd3rlpy.finish_task.finish_task_co import finish_task_co
from myd3rlpy.finish_task.finish_task_mi import finish_task_mi

def finish_task_ewc_co(dataset_num, dataset, dataset_type, original, network, indexes_euclid, real_action_size, args, device, alpha=0.5):
    replay_dataset = finish_task_co(dataset_num, dataset, original, network, indexes_euclid, real_action_size, args, device, for_ewc=True)
    replay_dataloader = DataLoader(replay_dataset, batch_size=args.batch_size)
    network._impl.post_train_process(replay_dataloader, alpha=alpha)

def finish_task_ewc_mi(dataset_num, dataset, dataset_type, original, network, indexes_euclid, real_action_size, args, device, alpha=0.5):
    replay_dataset = finish_task_mi(dataset_num, dataset, original, network, indexes_euclid, real_action_size, args, device, for_ewc=True)
    replay_dataloader = DataLoader(replay_dataset, batch_size=args.batch_size)
    network._impl.post_train_process(replay_dataloader, alpha=alpha)
