import torch
from torch.distributions.normal import Normal
import numpy as np
from d3rlpy.dataset import Transition
from myd3rlpy.siamese_similar import similar_psi, similar_phi, similar_mb


def finish_task_co(id_size, task_nums, dataset, original, network, indexes_euclid, real_action_size, topk, device, orl_loss=True, bc_loss=False, in_task=False, max_export_time = 0, max_reward=None):
    # 关键算法
    task_id_tensor = np.eye(task_nums)[id_size].squeeze()
    task_id_tensor = torch.from_numpy(np.broadcast_to(task_id_tensor, (original.shape[0], task_nums))).to(torch.float32).to(device)
    original_observation = torch.cat([original, task_id_tensor], dim=1)
    original_action = network._impl._policy(original_observation)
    replay_indexes = []
    if bc_loss and not in_task:
        replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals = [], [], [], [], [], [], []
        replay_means, replay_std_logs, replay_qs, replay_phis, replay_psis = [], [], [], [], []
    if orl_loss or in_task:
        new_transitions = []

    export_time = 0
    start_indexes = torch.zeros(0)
    while start_indexes.shape[0] != 0 and original_observation is not None and export_time < max_export_time:
        if original_observation is not None:
            start_observations = original_observation
            start_actions = original_action
            original_observation = None
        else:
            start_observations = torch.from_numpy(dataset._observations[start_indexes.cpu().numpy()]).to(device)
            start_actions = network._impl._policy(start_observations)

        mus, logstds = []
        for model in network._models:
            mu, logstd = network.compute_stats(start_observations, start_actions[:, :real_action_size])
            mus.append(mu)
            logstds.append(logstd)
        mus = mus.stack(dim=1)
        logstds = logstds.stack(dim=1)
        mus = mus[torch.arange(start_observations.shape[0]), torch.randing(len(network._models), size=(start_observations.shape[0],))]
        logstds = logstds[torch.arange(start_observations.shape[0]), torch.randint(len(network._models), size=(start_observations.shape[0],))]
        dist = Normal(mus, torch.exp(logstds))
        pred = dist.rsample()
        pred_observations = torch.cat([pred[:, :-1], task_id_tensor])
        next_x = start_observations + pred_observations
        next_reward = pred[:, -1].view(-1, 1)

        near_indexes, _, _ = similar_mb(mus, logstds, dataset._observations, network._dynamics._impl._dynamics, topk=topk)
        if orl_loss or in_task:
            for i in range(start_observations.shape[0]):
                transition = Transition(
                    observation_shape = network._impl.observation_shape,
                    action_size = network._impl.action_size,
                    observation = start_observations[i],
                    action = start_actions[i],
                    reward = float(dataset._rewards[i]),
                    next_observations = next_x,
                    next_rewards = next_reward,
                )
                new_transitions.append(transition)
        if bc_loss and not in_task and start_indexes.shape[0] != 0:
            start_rewards = torch.from_numpy(dataset._rewards[start_indexes]).to(device)
            start_terminals = torch.from_numpy(dataset.terminals[start_indexes]).to(device)
            start_next_observations = next_x
            start_next_actions = network._impl._policy(start_next_observations)
            start_next_rewards = next_reward
            start_dists = network._impl._policy.dist(start_observations)
            start_means = start_dists.mean
            start_std_logs = start_dists.stddev
            start_psis = network._impl._psi.forward(start_observations)
            start_qs = network._impl._q_func.forward(start_observations, start_actions)
            start_phis = network._impl._phi.forward(start_observations, start_actions)
            replay_observations.append(start_observations)
            replay_actions.append(start_actions)
            replay_rewards.append(start_rewards)
            replay_next_observations.append(start_next_observations)
            replay_next_actions.append(start_next_actions)
            replay_next_rewards.append(start_next_rewards)
            replay_terminals.append(start_terminals)
            replay_means.append(start_means)
            replay_std_logs.append(start_std_logs)
            replay_qs.append(start_qs)
            replay_psis.append(start_psis)
            replay_phis.append(start_phis)

        near_indexes = near_indexes.reshape((near_indexes.shape[0] * near_indexes.shape[1]))
        near_indexes = torch.cat(near_indexes_n, dim=0)
        near_indexes = torch.unique(near_indexes).cpu().numpy()
        start_indexes = near_indexes
        for replay_index in replay_indexes:
            start_indexes = np.setdiff1d(start_indexes, replay_index)
        start_rewards = dataset._rewards[start_indexes]
        if max_reward is not None:
            start_indexes = start_indexes[start_rewards >= max_reward]
        if start_indexes.shape[0] == 0:
            break
        replay_indexes.append(start_indexes)
        replay_indexes = np.concatenate(replay_indexes, dim=0)

    if bc_loss and not in_task:
        replay_observations = torch.cat(replay_observations, dim=0).cpu()
        replay_actions = torch.cat(replay_actions, dim=0).cpu()
        replay_rewards = torch.cat(replay_rewards, dim=0).cpu()
        replay_next_observations = torch.cat(replay_next_observations, dim=0).cpu()
        replay_next_actions = torch.cat(replay_next_actions, dim=0).cpu()
        replay_next_rewards = torch.cat(replay_next_rewards, dim=0).cpu()
        replay_terminals = torch.cat(replay_terminals, dim=0).cpu()
        replay_means = torch.cat(replay_means, dim=0).cpu()
        replay_std_logs = torch.cat(replay_std_logs, dim=0).cpu()
        replay_qs = torch.cat(replay_qs, dim=0).cpu()
        replay_phis = torch.cat(replay_phis, dim=0).cpu()
        replay_psis = torch.cat(replay_psis, dim=0).cpu()
        replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_next_actions, replay_next_rewards, replay_terminals, replay_means, replay_std_logs, replay_qs, replay_phis, replay_psis)
    return replay_dataset
