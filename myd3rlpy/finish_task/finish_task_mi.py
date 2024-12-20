import torch
import numpy as np
from myd3rlpy.siamese_similar import similar_euclid_obs, similar_euclid_act


def finish_task_mi(dataset_num, task_nums, dataset, original, network, indexes_euclid, real_action_size, topk, device, for_ewc=False):
    # 关键算法
    task_id_numpy = np.eye(task_nums)[dataset_num].squeeze()
    task_id_numpy = np.broadcast_to(task_id_numpy, (original.shape[0], task_nums))
    original = torch.cat([original, torch.from_numpy(task_id_numpy).to(torch.float32).to(device)], dim=1)
    start_indexes = similar_euclid_obs(original.to(device), torch.from_numpy(dataset._observations).to(device), topk=topk)
    start_indexes = start_indexes.reshape((start_indexes.shape[0] * start_indexes.shape[1]))
    start_indexes = torch.unique(start_indexes)
    start_observations = torch.from_numpy(dataset._observations[start_indexes.cpu().numpy()]).to(device)
    start_actions = network._impl._policy(start_observations)
    if for_ewc:
        replay_indexes, replay_actions, replay_observations, replay_next_observations, replay_next_rewards, replay_terminals, replay_n_steps, replay_masks = [], [], [], [], [], [], [], []
    else:
        replay_indexes, replay_actions, replay_observations, replay_qs, replay_phis, replay_psis = [], [], [], [], [], []
    while len(start_indexes) != 0:
        near_observations = dataset._observations[indexes_euclid[start_indexes]]
        near_actions = dataset._actions[indexes_euclid[start_indexes]][:, :, :real_action_size]
        near_indexes_n = []
        # this_observations = start_observations.unsqueeze(dim=1).expand(-1, indexes_euclid.shape[1], -1)
        near_indexes, _, _ = similar_euclid_act(start_observations, start_actions, near_observations, near_actions, indexes_euclid[start_indexes], topk=topk)
        near_indexes = near_indexes.reshape((near_indexes.shape[0] * near_indexes.shape[1]))
        near_indexes_n.append(near_indexes)
        near_indexes = torch.cat(near_indexes_n, dim=0)
        near_indexes = torch.unique(near_indexes).cpu().numpy()
        start_indexes = near_indexes
        for replay_index in replay_indexes:
            start_indexes = np.setdiff1d(start_indexes, replay_index)
        start_indexes = start_indexes
        start_rewards = dataset._rewards[start_indexes]
        start_indexes = start_indexes[start_rewards != 1]
        if start_indexes.shape[0] == 0:
            break
        start_terminals = dataset.terminals[start_indexes]
        start_indexes = start_indexes[start_terminals == False]
        start_observations = torch.from_numpy(dataset._observations[start_indexes]).to(device)
        start_indexes = torch.from_numpy(start_indexes)
        start_rewards = torch.from_numpy(start_rewards)
        if for_ewc:
            start_terminals = torch.from_numpy(start_terminals)
            start_next_observations = torch.from_numpy(dataset._next_observations[start_indexes]).to(device)
            start_next_rewards = torch.from_numpy(dataset._next_rewards[start_indexes]).to(device)
            start_terminals = torch.from_numpy(dataset._terminals[start_indexes]).to(device)
            start_n_steps = torch.from_numpy(dataset._n_steps[start_indexes]).to(device)
            start_masks = torch.from_numpy(dataset._masks[start_indexes]).to(device)
            replay_indexes.append(start_indexes)
            replay_observations.append(start_observations)
            replay_actions.append(start_actions)
            # For recalc
            replay_next_observations.append(start_next_observations)
            replay_next_rewards.append(start_next_rewards)
            replay_terminals.append(start_terminals)
            replay_n_steps.append(start_n_steps)
            replay_masks.append(start_masks)
        else:
            start_actions = network._impl._policy(start_observations)
            start_psis = network._impl._psi.forward(start_observations)
            start_qs = network._impl._q_func.forward(start_observations, start_actions)
            start_phis = network._impl._phi.forward(start_observations, start_actions)
            replay_indexes.append(start_indexes)
            replay_observations.append(start_observations)
            replay_actions.append(start_actions)
            replay_qs.append(start_qs)
            replay_psis.append(start_psis)
            replay_phis.append(start_phis)
    if for_ewc:
        replay_indexes = torch.cat(replay_indexes, dim=0)
        replay_observations = torch.cat(replay_observations, dim=0).cpu()
        replay_actions = torch.cat(replay_actions, dim=0).cpu()
        replay_next_observations = torch.cat(replay_next_observations, dim=0).cpu()
        replay_next_rewards = torch.cat(replay_next_rewards, dim=0).cpu()
        replay_terminals = torch.cat(replay_terminals, dim=0).cpu()
        replay_n_steps = torch.cat(replay_n_steps, dim=0).cpu()
        replay_masks = torch.cat(replay_masks, dim=0).cpu()
        replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_next_observations, replay_next_rewards, replay_terminals, replay_n_steps, replay_masks)
    else:
        replay_indexes = torch.cat(replay_indexes, dim=0)
        replay_observations = torch.cat(replay_observations, dim=0).cpu()
        replay_actions = torch.cat(replay_actions, dim=0).cpu()
        replay_qs = torch.cat(replay_qs, dim=0).cpu()
        replay_phis = torch.cat(replay_phis, dim=0).cpu()
        replay_psis = torch.cat(replay_psis, dim=0).cpu()
        replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_qs, replay_phis, replay_psis)
    return replay_dataset
