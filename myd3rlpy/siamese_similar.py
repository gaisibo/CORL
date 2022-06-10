import sys
import os
import time
import itertools
import torch
from torch.multiprocessing import Pool
from torch.distributions.normal import Normal
import numpy as np
from myd3rlpy.models.torch.siamese import Phi, Psi


device = torch.device('cuda')
def compute_vector_norm(tensor1, tensor2):
    tensor1.half()
    tensor2.half()
    ret = torch.linalg.vector_norm(tensor1 - tensor2, dim = 2)
    return ret
def similar_euclid(obs_all, obs_transition, dataset_name, indexes_name, distances_name, input_indexes=None, eval_batch_size=1000, topk=64, compare_dim=2):
    near_indexes_ = []
    near_distances_ = []
    i = 0
    while i < obs_all.shape[0]:
        in_results = []
        j = 0
        if i + eval_batch_size < obs_all.shape[0]:
            while j < obs_transition.shape[0]:
                if j + eval_batch_size < obs_transition.shape[0]:
                    tensor1 = obs_all[i: i + eval_batch_size, :compare_dim].unsqueeze(dim=1).expand(-1, eval_batch_size, -1)
                    tensor2 = obs_transition[j: j + eval_batch_size, :compare_dim].unsqueeze(dim=0).expand(eval_batch_size, -1, -1)
                    siamese_distance = compute_vector_norm(tensor1, tensor2).cpu()
                    in_results.append(siamese_distance)
                else:
                    tensor1 = obs_all[i: i + eval_batch_size, :compare_dim].unsqueeze(dim=1).expand(-1, obs_transition.shape[0] - j, -1)
                    tensor2 = obs_transition[j:, :compare_dim].unsqueeze(dim=0).expand(eval_batch_size, -1, -1)
                    siamese_distance = compute_vector_norm(tensor1, tensor2).cpu()
                    in_results.append(siamese_distance)
                j += eval_batch_size
        else:
            while j < obs_transition.shape[0]:
                if j + eval_batch_size < obs_transition.shape[0]:
                    tensor1 = obs_all[i:, :compare_dim].unsqueeze(dim=1).expand(-1, eval_batch_size, -1)
                    tensor2 = obs_transition[j: j + eval_batch_size, :compare_dim].unsqueeze(dim=0).expand(obs_all.shape[0] - i, -1, -1)
                    siamese_distance = compute_vector_norm(tensor1, tensor2).cpu()
                    in_results.append(siamese_distance)
                else:
                    tensor1 = obs_all[i:, :compare_dim].unsqueeze(dim=1).expand(-1, obs_transition.shape[0] - j, -1)
                    tensor2 = obs_transition[j:, :compare_dim].unsqueeze(dim=0).expand(obs_all.shape[0] - i, -1, -1)
                    siamese_distance = compute_vector_norm(tensor1, tensor2).cpu()
                    in_results.append(siamese_distance)
                j += eval_batch_size
        i += eval_batch_size
        in_results = torch.cat(in_results, dim=1)
        near_distances, near_indexes = torch.topk(in_results, topk, largest=False)
        near_indexes_.append(near_indexes)
        near_distances_.append(near_distances)
    near_indexes = torch.cat(near_indexes_, dim=0)
    near_distances = torch.cat(near_distances_, dim=0)
    if input_indexes is not None:
        near_indexes = input_indexes[near_indexes]
    if not os.path.exists('near_indexes/' + dataset_name):
        os.makedirs('near_indexes/' + dataset_name)
    if not os.path.exists('near_distances/' + dataset_name):
        os.makedirs('near_distances/' + dataset_name)
    torch.save(near_indexes, indexes_name)
    torch.save(near_distances, distances_name)
    return near_indexes, distances_name

# 如果不存在一个obs有很多act的情况，可以不用这一个函数。
def similar_euclid_obs(obs_batch, obs_near, input_indexes=None, eval_batch_size=2500, topk=10):
    siamese_distance = torch.linalg.vector_norm(obs_batch.unsqueeze(dim=1).expand(-1, obs_near.shape[0], -1) - obs_near.unsqueeze(dim=0).expand(obs_batch.shape[0], -1, -1), dim=1)
    _, near_indexes = torch.topk(siamese_distance, topk, largest=False)
    if input_indexes is not None:
        for i in range(near_indexes.shape[0]):
            near_indexes[i, :] = input_indexes[i, near_indexes[i, :]]
    return near_indexes
# 如果不存在一个obs有很多act的情况，可以不用这一个函数。
def similar_psi(obs_batch, obs_near, psi, input_indexes=None, eval_batch_size=2500, topk=10):
    psi_batch = psi(obs_batch)
    psi_near = psi(obs_near)
    siamese_distance = torch.linalg.vector_norm(psi_batch.unsqueeze(dim=1).expand(-1, psi_near.shape[0], -1) - psi_near.unsqueeze(dim=0).expand(psi_batch.shape[0], -1, -1), dim=1)
    _, near_indexes = torch.topk(siamese_distance, topk, largest=False)
    if input_indexes is not None:
        for i in range(near_indexes.shape[0]):
            near_indexes[i, :] = input_indexes[i, near_indexes[i, :]]
    return near_indexes

def similar_euclid_act(obs_batch, act_batch, obs_near, act_near, input_indexes=None, topk=4):
    b, n, o = obs_near.shape
    cat_batch = torch.cat([obs_batch, act_batch], dim=1).unsqueeze(dim=1).expand(-1, n, -1)
    i = 0
    siamese_distance = []
    for i in range(b):
        cat_near = torch.cat([torch.from_numpy(obs_near[i, :, :]).to(cat_batch.device), torch.from_numpy(act_near[i, :, :]).to(cat_batch.device)], dim=1)
        siamese_distance.append(torch.linalg.vector_norm(cat_batch[i] - cat_near, dim=1))
    siamese_distance = torch.stack(siamese_distance, dim=0)
    near_distances, near_indexes = torch.topk(siamese_distance, topk, largest=False)
    if input_indexes is not None:
        for i in range(near_indexes.shape[0]):
            near_indexes[i, :] = input_indexes[i, near_indexes[i, :]]
    smallest_distance, smallest_index = near_distances[:, 0], near_indexes[:, 0]
    return near_indexes, smallest_index, smallest_distance

def similar_phi(obs_batch, act_batch, obs_near, act_near, phi, input_indexes=None, topk=4, eval_batch_size=32):
    b, n, o = obs_near.shape
    near_distances = []
    near_indexes = []
    i = 0
    while i < b:
        if i + eval_batch_size < b:
            phi_batch = phi(obs_batch[i: i + eval_batch_size], act_batch[i: i + eval_batch_size]).unsqueeze(dim=1).expand(-1, n, -1)
            obs_near_ = torch.from_numpy(obs_near[i: i + eval_batch_size]).to(phi_batch.device).reshape(eval_batch_size * n, -1)
            act_near_ = torch.from_numpy(act_near[i: i + eval_batch_size]).to(phi_batch.device).reshape(eval_batch_size * n, -1)
            phi_near = phi(obs_near_, act_near_)
            phi_near = phi_near.reshape(eval_batch_size, n, -1)
        else:
            phi_batch = phi(obs_batch[i:], act_batch[i:]).unsqueeze(dim=1).expand(-1, n, -1)
            obs_near_ = torch.from_numpy(obs_near[i:]).to(phi_batch.device).reshape((b - i) * n, -1)
            act_near_ = torch.from_numpy(act_near[i:]).to(phi_batch.device).reshape((b - i) * n, -1)
            phi_near = phi(obs_near_, act_near_)
            phi_near = phi_near.reshape((b - i), n, -1)
        siamese_distance = torch.linalg.vector_norm(phi_batch - phi_near, dim=2)
        near_distances_, near_indexes_ = torch.topk(siamese_distance, topk, largest=False)
        near_distances.append(near_distances_)
        near_indexes.append(near_indexes_)
        i += eval_batch_size
    near_distances = torch.cat(near_distances, dim=0).cpu().detach().numpy()
    near_indexes = torch.cat(near_indexes, dim=0).cpu().detach().numpy()
    if input_indexes is not None:
        for i in range(near_indexes.shape[0]):
            near_indexes[i, :] = input_indexes[i, near_indexes[i, :]]
    smallest_distance, smallest_index = near_distances[:, 0], near_indexes[:, 0]
    return near_indexes, smallest_index, smallest_distance

def similar_mb(mus, logstds, observations, rewards, topk=4, batch_size=64, input_indexes=None):
    normal = Normal(mus, torch.exp(logstds))
    if len(rewards.shape) == 1:
        rewards = rewards.unsqueeze(dim=1)
    log_prob = normal.log_prob(torch.cat([observations, rewards], dim=1).to(mus.device))
    log_prob = log_prob.sum(dim=1)
    print(f"max(log_prob): {max(log_prob)}")
    try:
        category = torch.distributions.categorical.Categorical(logits=log_prob)
    except:
        category = torch.distributions.categorical.Categorical(logits=log_prob)
    near_index = category.sample()
    if input_indexes is not None:
        near_index = input_indexes[near_index]
    return near_index

def similar_mb_euclid(obs_all, obs_transition, eval_batch_size=1000, topk=256, compare_dim=2):
    results = []
    near_distances = []
    i = 0
    if len(obs_all.shape) == 2:
        while i < obs_all.shape[0]:
            in_results = []
            j = 0
            if i + eval_batch_size < obs_all.shape[0]:
                while j < obs_transition.shape[0]:
                    if j + eval_batch_size < obs_transition.shape[0]:
                        tensor1 = obs_all[i: i + eval_batch_size, :compare_dim].unsqueeze(dim=1).expand(-1, eval_batch_size, -1)
                        tensor2 = obs_transition[j: j + eval_batch_size, :compare_dim].unsqueeze(dim=0).expand(eval_batch_size, -1, -1)
                        siamese_distance = torch.linalg.vector_norm(tensor1 - tensor2, dim = 2)
                        in_results.append(siamese_distance)
                    else:
                        tensor1 = obs_all[i: i + eval_batch_size, :compare_dim].unsqueeze(dim=1).expand(-1, obs_transition.shape[0] - j, -1)
                        tensor2 = obs_transition[j:, :compare_dim].unsqueeze(dim=0).expand(eval_batch_size, -1, -1)
                        siamese_distance = torch.linalg.vector_norm(tensor1 - tensor2, dim = 2)
                        in_results.append(siamese_distance)
                    j += eval_batch_size
            else:
                while j < obs_transition.shape[0]:
                    if j + eval_batch_size < obs_transition.shape[0]:
                        tensor1 = obs_all[i:, :compare_dim].unsqueeze(dim=1).expand(-1, eval_batch_size, -1)
                        tensor2 = obs_transition[j: j + eval_batch_size, :compare_dim].unsqueeze(dim=0).expand(obs_all.shape[0] - i, -1, -1)
                        siamese_distance = torch.linalg.vector_norm(tensor1 - tensor2, dim = 2)
                        in_results.append(siamese_distance)
                    else:
                        tensor1 = obs_all[i:, :compare_dim].unsqueeze(dim=1).expand(-1, obs_transition.shape[0] - j, -1)
                        tensor2 = obs_transition[j:, :compare_dim].unsqueeze(dim=0).expand(obs_all.shape[0] - i, -1, -1)
                        siamese_distance = torch.linalg.vector_norm(tensor1 - tensor2, dim = 2)
                        in_results.append(siamese_distance)
                    j += eval_batch_size
            i += eval_batch_size
            in_results = torch.cat(in_results, dim=1)
            near_distance, near_indexes = torch.topk(in_results, topk, largest=False)
            results.append(near_indexes)
            near_distances.append(near_distance)
    else:
        for i in range(obs_all.shape[0]):
            in_results = []
            j = 0
            while j < obs_transition.shape[0]:
                if j + eval_batch_size < obs_transition.shape[0]:
                    tensor1 = obs_all[i, :, :compare_dim].unsqueeze(dim=1).expand(-1, eval_batch_size, -1)
                    tensor2 = obs_transition[j: j + eval_batch_size, :compare_dim].unsqueeze(dim=0).expand(eval_batch_size, -1, -1)
                    siamese_distance = torch.linalg.vector_norm(tensor1 - tensor2, dim = 2)
                    in_results.append(siamese_distance)
                else:
                    tensor1 = obs_all[i, :, :compare_dim].unsqueeze(dim=1).expand(-1, obs_transition.shape[0] - j, -1)
                    tensor2 = obs_transition[j:, :compare_dim].unsqueeze(dim=0).expand(eval_batch_size, -1, -1)
                    siamese_distance = torch.linalg.vector_norm(tensor1 - tensor2, dim = 2)
                    in_results.append(siamese_distance)
                j += eval_batch_size
            in_results = torch.cat(in_results, dim=1)
            near_distance, near_indexes = torch.topk(in_results, topk, largest=False)
            results.append(near_indexes)
            near_distances.append(near_distance)
    near_indexes = torch.cat(results, dim=0).to(torch.int64)
    near_distances = torch.cat(near_distances, dim=0).to(torch.float32)
    return near_indexes, near_distances
