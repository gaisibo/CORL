import os
import time
import itertools
import torch
from torch.multiprocessing import Pool
import numpy as np
from myd3rlpy.models.torch.siamese import Phi, Psi


device = torch.device('cuda')
def compute_vector_norm(tensor1, tensor2):
    tensor1.half()
    tensor2.half()
    ret = torch.linalg.vector_norm(tensor1 - tensor2, dim = 2)
    return ret.cpu()
def similar_euclid(obs_all, dataset_name, dataset_num, input_indexes=None, eval_batch_size=10000, topk=256):
    filename = 'near_indexes_' + dataset_name + '_' + str(dataset_num) + '.pt'
    if os.path.exists(filename):
        return torch.load(filename)
    results = []
    i = 0
    while i < obs_all.shape[0]:
        in_results = []
        j = 0
        if i + eval_batch_size < obs_all.shape[0]:
            while j < obs_all.shape[0]:
                if j + eval_batch_size < obs_all.shape[0]:
                    tensor1 = obs_all[i: i + eval_batch_size, :2].unsqueeze(dim=1).expand(-1, eval_batch_size, -1)
                    tensor2 = obs_all[j: j + eval_batch_size, :2].unsqueeze(dim=0).expand(eval_batch_size, -1, -1)
                    siamese_distance = compute_vector_norm(tensor1, tensor2)
                    in_results.append(siamese_distance)
                else:
                    tensor1 = obs_all[i: i + eval_batch_size, :2].unsqueeze(dim=1).expand(-1, obs_all.shape[0] - j, -1)
                    tensor2 = obs_all[j:, :2].unsqueeze(dim=0).expand(eval_batch_size, -1, -1)
                    siamese_distance = compute_vector_norm(tensor1, tensor2)
                    in_results.append(siamese_distance)
                j += eval_batch_size
        else:
            while j < obs_all.shape[0]:
                if j + eval_batch_size < obs_all.shape[0]:
                    tensor1 = obs_all[i:, :2].unsqueeze(dim=1).expand(-1, eval_batch_size, -1)
                    tensor2 = obs_all[j: j + eval_batch_size, :2].unsqueeze(dim=0).expand(obs_all.shape[0] - i, -1, -1)
                    siamese_distance = compute_vector_norm(tensor1, tensor2)
                    in_results.append(siamese_distance)
                else:
                    tensor1 = obs_all[i:, :2].unsqueeze(dim=1).expand(-1, obs_all.shape[0] - j, -1)
                    tensor2 = obs_all[j:, :2].unsqueeze(dim=0).expand(obs_all.shape[0] - i, -1, -1)
                    siamese_distance = compute_vector_norm(tensor1, tensor2)
                    in_results.append(siamese_distance)
                j += eval_batch_size
        i += eval_batch_size
        in_results = torch.cat(in_results, dim=1)
        _, near_indexes = torch.sort(in_results)
        near_indexes = near_indexes[:, :topk]
        results.append(near_indexes)
    near_indexes = torch.cat(results, dim=0)
    if input_indexes is not None:
        near_indexes = input_indexes[near_indexes]
    torch.save(near_indexes, filename)
    return near_indexes

# 如果不存在一个obs有很多act的情况，可以不用这一个函数。
def similar_psi(obs_batch, obs_all, psi, input_indexes=None, eval_batch_size=2500, topk=64):
    i = 0
    all_siamese_distance = None
    psi_batch = psi(obs_batch)
    if i < obs_all.shape[0]:
        if i + eval_batch_size < obs_all.shape[0]:
            siamese_distance = torch.linalg.vector_norm(psi_batch.unsqueeze(dim=1).expand(-1, eval_batch_size, -1) - psi(obs_all[i: i + eval_batch_size]).unsqueeze(dim=0).expand(obs_batch.shape[0], -1, -1), dim=2)
        else:
            siamese_distance = torch.linalg.vector_norm(psi_batch.unsqueeze(dim=1).expand(-1, obs_all.shape[0] - i, -1) - psi(obs_all[i:]).unsqueeze(dim=0).expand(obs_batch.shape[0], -1, -1), dim=2)
        if all_siamese_distance is not None:
            all_siamese_distance = torch.cat([all_siamese_distance, siamese_distance], dim=1)
        else:
            all_siamese_distance = siamese_distance
        i += i + eval_batch_size
    _, near_indexes = torch.sort(all_siamese_distance)
    near_indexes = near_indexes[:, :topk]
    if input_indexes is not None:
        near_indexes = input_indexes[near_indexes]
    return near_indexes

def similar_phi(obs_batch, act_batch, obs_near, act_near, phi, task_id, input_indexes=None, eval_batch_size=2500, distance_threshold=1):
    print(f'obs_batch: {obs_batch.shape}')
    print(f'act_batch: {act_batch.shape}')
    print(f'task_id: {task_id.shape}')
    phi_batch = phi(obs_batch, act_batch, task_id)
    b, n, o = obs_near.shape
    obs_near = obs_near.reshape(b * n, -1)
    act_near = act_near.reshape(b * n, -1)
    task_id = task_id.unsqueeze(dim=1).expand(-1, n, -1).reshape(b * n, -1)
    near_batch = phi(obs_near, act_near, task_id)
    near_batch = near_batch.reshape(b, n, -1)
    siamese_distance = torch.linalg.vector_norm(phi_batch.unsqueeze(dim=1).expand(-1, near_batch.shape[1], -1) - near_batch, dim=2)
    near_indexes = torch.nonzero(torch.where(siamese_distance < distance_threshold, 1, 0))
    smallest_distance, smallest_index = torch.min(siamese_distance, dim=1)
    if input_indexes is not None:
        near_indexes = input_indexes[near_indexes]
        smallest_index = input_indexes[smallest_index]
    return near_indexes, smallest_index, smallest_distance
