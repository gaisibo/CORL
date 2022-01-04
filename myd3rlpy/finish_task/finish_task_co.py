import torch
import numpy as np
from myd3rlpy.siamese_similar import similar_psi, similar_phi


def finish_task_co(dataset_num, task_nums, dataset, original, network, indexes_euclid, real_action_size, topk, device, for_ewc=False):
    # 关键算法
    task_id_numpy = np.eye(task_nums)[dataset_num].squeeze()
    task_id_numpy = np.broadcast_to(task_id_numpy, (original.shape[0], task_nums))
    original = torch.cat([original, torch.from_numpy(task_id_numpy).to(torch.float32).to(device)], dim=1)
    start_indexes = similar_psi(original.to(device), torch.from_numpy(dataset._observations).to(device), network._impl._psi, topk=topk)
    start_indexes = start_indexes.reshape((start_indexes.shape[0] * start_indexes.shape[1]))
    start_indexes = torch.unique(start_indexes)
    start_observations = torch.from_numpy(dataset._observations[start_indexes.cpu().numpy()]).to(device)
    start_actions = network._impl._policy(start_observations)
    replay_indexes, replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals = [], [], [], [], [], [], [], []
    if not for_ewc:
        replay_policy_actions, replay_qs, replay_phis, replay_psis = [], [], [], []
    while len(start_indexes) != 0:
        near_observations = dataset._observations[indexes_euclid[start_indexes]]
        near_actions = dataset._actions[indexes_euclid[start_indexes]][:, :, :real_action_size]
        near_indexes_n = []
        # this_observations = start_observations.unsqueeze(dim=1).expand(-1, indexes_euclid.shape[1], -1)
        near_indexes, _, _ = similar_phi(start_observations, start_actions[:, :real_action_size], near_observations, near_actions, network._impl._phi, indexes_euclid[start_indexes], topk=topk)
        near_indexes = near_indexes.reshape((near_indexes.shape[0] * near_indexes.shape[1]))
        near_indexes_n.append(near_indexes)
        near_indexes = torch.cat(near_indexes_n, dim=0)
        near_indexes = torch.unique(near_indexes).cpu().numpy()
        start_indexes = near_indexes
        for replay_index in replay_indexes:
            start_indexes = np.setdiff1d(start_indexes, replay_index)
        start_rewards = dataset._rewards[start_indexes]
        start_indexes = start_indexes[start_rewards != 1]
        start_terminals = dataset.terminals[start_indexes]
        start_indexes = start_indexes[start_terminals == False]
        if start_indexes.shape[0] == 0:
            break
        start_observations = torch.from_numpy(dataset._observations[start_indexes]).to(device)
        start_actions = torch.from_numpy(dataset._actions[start_indexes]).to(device)
        start_rewards = torch.from_numpy(dataset._rewards[start_indexes]).to(device)
        start_terminals = torch.from_numpy(dataset.terminals[start_indexes]).to(device)
        # 超出范围的next_indexes会因为terminal的原因不起作用，所以随便赋一个值就行。
        start_next_indexes = np.where(start_indexes + 1 < dataset._observations.shape[0], start_indexes + 1, 0)
        start_next_observations = torch.from_numpy(dataset._observations[start_next_indexes]).to(device)
        start_next_actions = torch.from_numpy(dataset._actions[start_next_indexes]).to(device)
        start_next_rewards = torch.from_numpy(dataset._rewards[start_next_indexes]).to(device)
        replay_indexes.append(start_indexes)
        replay_observations.append(start_observations)
        replay_actions.append(start_actions)
        replay_rewards.append(start_rewards)
        # For recalc
        replay_next_observations.append(start_next_observations)
        replay_next_actions.append(start_next_actions)
        replay_next_rewards.append(start_next_rewards)
        replay_terminals.append(start_terminals)
        if not for_ewc:
            start_policy_actions = network._impl._policy(start_observations)
            start_psis = network._impl._psi.forward(start_observations)
            start_qs = network._impl._q_func.forward(start_observations, start_policy_actions)
            start_phis = network._impl._phi.forward(start_observations, start_policy_actions)
            replay_policy_actions.append(start_policy_actions)
            replay_qs.append(start_qs)
            replay_psis.append(start_psis)
            replay_phis.append(start_phis)
    replay_observations = torch.cat(replay_observations, dim=0).cpu()
    replay_actions = torch.cat(replay_actions, dim=0).cpu()
    replay_rewards = torch.cat(replay_rewards, dim=0).cpu()
    replay_next_observations = torch.cat(replay_next_observations, dim=0).cpu()
    replay_next_actions = torch.cat(replay_next_actions, dim=0).cpu()
    replay_next_rewards = torch.cat(replay_next_rewards, dim=0).cpu()
    replay_terminals = torch.cat(replay_terminals, dim=0).cpu()
    if for_ewc:
        replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals)
    else:
        replay_policy_actions = torch.cat(replay_policy_actions, dim=0).cpu()
        replay_qs = torch.cat(replay_qs, dim=0).cpu()
        replay_phis = torch.cat(replay_phis, dim=0).cpu()
        replay_psis = torch.cat(replay_psis, dim=0).cpu()
        replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals, replay_policy_actions, replay_qs, replay_phis, replay_psis)
    return replay_dataset
