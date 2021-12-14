import time
import itertools
import torch
from torch.multiprocessing import Pool
import numpy as np
from myd3rlpy.models.torch.siamese import Phi, Psi


device = torch.device('cuda')
def compute_vector_norm(tensors):
    tensor1, tensor2, numbers = tensors
    tensor1.half().to(device)
    tensor2.half().to(device)
    ret = torch.linalg.vector_norm(tensor1 - tensor2, dim = 2, ord=1)
    return ret.cpu()
def similar_euclid(obs_all, input_indexes=None, eval_batch_size=256, topk=256):
    tensor1s = []
    tensor2s = []
    numbers = []
    i = 0
    while i < obs_all.shape[0]:
        j = 0
        if i + eval_batch_size < obs_all.shape[0]:
            while j < obs_all.shape[0]:
                if j + eval_batch_size < obs_all.shape[0]:
                    tensor1s.append(obs_all[i: i + eval_batch_size].unsqueeze(dim=1).expand(-1, eval_batch_size, -1))
                    tensor2s.append(obs_all[j: j + eval_batch_size].unsqueeze(dim=0).expand(eval_batch_size, -1, -1))
                    numbers.append((i, j))
                else:
                    tensor1s.append(obs_all[i: i + eval_batch_size].unsqueeze(dim=1).expand(-1, obs_all.shape[0] - j, -1))
                    tensor2s.append(obs_all[j:].unsqueeze(dim=0).expand(eval_batch_size, -1, -1))
                    numbers.append((i, j))
                j += eval_batch_size
        else:
            while j < obs_all.shape[0]:
                if j + eval_batch_size < obs_all.shape[0]:
                    tensor1s.append(obs_all[i:].unsqueeze(dim=1).expand(-1, eval_batch_size, -1))
                    tensor2s.append(obs_all[j: j + eval_batch_size].unsqueeze(dim=0).expand(obs_all.shape[0] - i, -1, -1))
                    numbers.append((i, j))
                else:
                    tensor1s.append(obs_all[i:].unsqueeze(dim=1).expand(-1, obs_all.shape[0] - j, -1))
                    tensor2s.append(obs_all[j:].unsqueeze(dim=0).expand(obs_all.shape[0] - i, -1, -1))
                    numbers.append((i, j))
                j += eval_batch_size
        i += eval_batch_size
    all_siamese_distance = None
    multi_inputs = list(itertools.product(tensor1s, tensor2s, numbers))
    # print('start multiprocessing')
    # with Pool(processes=16) as pool:
    #     results = pool.map(compute_vector_norm, multi_input)
    results = []
    for multi_input in multi_inputs:
        print(f'input_nums: {len(multi_inputs)}')
        start_time = time.time()
        results.append(compute_vector_norm(multi_input))
        print(f'one cost time: {time.time() - start_time}')
        assert False
    i = 0; j = 0; k = 0
    siamese_distance = None
    while i < obs_all.shape[0]:
        if i + eval_batch_size < obs_all.shape[0]:
            while j < obs_all.shape[0]:
                if j + eval_batch_size < obs_all.shape[0]:
                    siamese_distance = results[k]
                    k += 1
                else:
                    siamese_distance = results[k]
                    k += 1
                j += eval_batch_size
            if all_siamese_distance is None:
                all_siamese_distance = siamese_distance
            else:
                all_siamese_distance = torch.cat([all_siamese_distance, siamese_distance], dim=1)
        else:
            if j + eval_batch_size < obs_all.shape[0]:
                siamese_distance = results[k]
                k += 1
            else:
                siamese_distance = results[k]
                k += 1
            if all_siamese_distance is None:
                all_siamese_distance = siamese_distance
            else:
                all_siamese_distance = torch.cat([all_siamese_distance, siamese_distance], dim=1)
        i += eval_batch_size
        if all_siamese_distance is None:
            all_siamese_distance = siamese_distance
        else:
            all_siamese_distance = torch.cat([all_siamese_distance, siamese_distance], dim=0)
    torch.save(all_siamese_distance, 'all_siamese_distance.pt')
    _, near_indexes = torch.sort(all_siamese_distance)
    near_indexes = near_indexes[:, :topk]
    if input_indexes is not None:
        near_indexes = input_indexes[near_indexes]
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

def similar_phi(obs_batch, act_batch, obs_all, act_all, phi, input_indexes=None, eval_batch_size=2500, distance_threshold=1):
    i = 0
    all_siamese_distance = None
    phi_batch = phi(obs_batch, act_batch)
    if i < obs_all.shape[0]:
        if i + eval_batch_size < obs_all.shape[0]:
            siamese_distance = torch.linalg.vector_norm(phi_batch.unsqueeze(dim=1).expand(-1, eval_batch_size, -1) - phi(obs_all[i: i + eval_batch_size], act_all[i: i + eval_batch_size]).unsqueeze(dim=0).expand(obs_batch.shape[0], -1, -1), dim=2)
        else:
            siamese_distance = torch.linalg.vector_norm(phi_batch.unsqueeze(dim=1).expand(-1, obs_all.shape[0] - i, -1) - phi(obs_all[i:], act_all[i:]).unsqueeze(dim=0).expand(obs_batch.shape[0], -1, -1), dim=2)
        if all_siamese_distance is not None:
            all_siamese_distance = torch.cat([all_siamese_distance, siamese_distance], dim=1)
        else:
            all_siamese_distance = siamese_distance
        i += i + eval_batch_size
    _, near_indexes = torch.non_zero(torch.where(all_siamese_distance < distance_threshold, 1, 0))
    smallest_index = torch.argmin(all_siamese_distance, dim=1)
    smallest_distance = torch.min(all_siamese_distance, dim=1)
    if input_indexes is not None:
        near_indexes = input_indexes[near_indexes]
        smallest_index = input_indexes[smallest_index]
    return near_indexes, smallest_index, smallest_distance
