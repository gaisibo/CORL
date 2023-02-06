import h5py
import tqdm
import numpy as np
import pickle


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys
def get_dataset(h5path):
    data_dict = {}
    data_dict_old = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in get_keys(dataset_file):
            try:  # first try loading as an array
                data_dict_old[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict_old[k] = dataset_file[k][()]
    data_dict['observations'] = np.flip(data_dict_old['obs'])
    data_dict['next_observations'] = np.flip(data_dict_old['next_obs'])
    data_dict['actions'] = np.flip(data_dict_old['actions'])
    data_dict['rewards'] = np.flip(data_dict_old['rewards'])
    data_dict['terminals'] = np.flip(data_dict_old['terminals'])
    for index, (reward, terminal) in enumerate(zip(data_dict['rewards'], data_dict['terminals'])):
        middle_reward = []
        if terminal == 1:
            middle_reward.append(reward)
    middle_reward = max(middle_reward) / 3
    expert_start_indexes = []
    expert_end_indexes = []
    replay_start_indexes = []
    replay_end_indexes = []
    start_index = 0
    for index, (reward, terminal) in enumerate(zip(data_dict['rewards'], data_dict['terminals'])):
        if terminal == 1:
            if reward >= middle_reward * 2:
                expert_start_indexes.append(start_index)
                expert_end_indexes.append(index)
            elif reward < middle_reward:
                replay_start_indexes.append(start_index)
                replay_end_indexes.append(index)
            start_index = index + 1
    expert_data_dict = dict()
    expert_data_dict['observations'] = np.concatenate([data_dict['observations'][start_index: end_index + 1] for start_index, end_index in zip(expert_start_indexes, expert_end_indexes)], axis=0)
    expert_data_dict['next_observations'] = np.concatenate([data_dict['next_observations'][start_index: end_index + 1] for start_index, end_index in zip(expert_start_indexes, expert_end_indexes)], axis=0)
    expert_data_dict['actions'] = np.concatenate([data_dict['actions'][start_index: end_index + 1] for start_index, end_index in zip(expert_start_indexes, expert_end_indexes)], axis=0)
    expert_data_dict['rewards'] = np.concatenate([data_dict['rewards'][start_index: end_index + 1] for start_index, end_index in zip(expert_start_indexes, expert_end_indexes)], axis=0)
    expert_data_dict['terminals'] = np.concatenate([data_dict['terminals'][start_index: end_index + 1] for start_index, end_index in zip(expert_start_indexes, expert_end_indexes)], axis=0)
    replay_data_dict = dict()
    replay_data_dict['observations'] = np.concatenate([data_dict['observations'][start_index: end_index + 1] for start_index, end_index in zip(replay_start_indexes, replay_end_indexes)], axis=0)
    replay_data_dict['next_observations'] = np.concatenate([data_dict['next_observations'][start_index: end_index + 1] for start_index, end_index in zip(replay_start_indexes, replay_end_indexes)], axis=0)
    replay_data_dict['actions'] = np.concatenate([data_dict['actions'][start_index: end_index + 1] for start_index, end_index in zip(replay_start_indexes, replay_end_indexes)], axis=0)
    replay_data_dict['rewards'] = np.concatenate([data_dict['rewards'][start_index: end_index + 1] for start_index, end_index in zip(replay_start_indexes, replay_end_indexes)], axis=0)
    replay_data_dict['terminals'] = np.concatenate([data_dict['terminals'][start_index: end_index + 1] for start_index, end_index in zip(replay_start_indexes, replay_end_indexes)], axis=0)
    expert_new_data = dict(
        observations=np.array(expert_data_dict['observations']).astype(np.float32),
        actions=np.array(expert_data_dict['actions']).astype(np.float32),
        next_observations=np.array(expert_data_dict['next_observations']).astype(np.float32),
        rewards=np.array(expert_data_dict['rewards']).astype(np.float32),
        terminals=np.array(expert_data_dict['terminals']).astype(np.bool),
    )
    replay_new_data = dict(
        observations=np.array(replay_data_dict['observations']).astype(np.float32),
        actions=np.array(replay_data_dict['actions']).astype(np.float32),
        next_observations=np.array(replay_data_dict['next_observations']).astype(np.float32),
        rewards=np.array(replay_data_dict['rewards']).astype(np.float32),
        terminals=np.array(replay_data_dict['terminals']).astype(np.bool),
    )
    return expert_new_data, replay_new_data

if __name__ == '__main__':
    # for task_id in range(5):
    #     expert_data, replay_data = get_dataset(f'dataset/macaw/ant_dir/buffers_ant_dir_train_{task_id}_sub_task_0.hdf5')
    #     hfile = h5py.File(f'dataset/macaw/sac_ant_dir_{task_id}/expert.hdf5', 'w')
    #     for k in expert_data:
    #         hfile.create_dataset(k, data=expert_data[k], compression='gzip')
    #     hfile.close()
    #     hfile = h5py.File(f'dataset/macaw/sac_ant_dir_{task_id}/replay.hdf5', 'w')
    #     for k in replay_data:
    #         hfile.create_dataset(k, data=replay_data[k], compression='gzip')
    #     hfile.close()
    # for task_id in range(5):
    #     expert_data, replay_data = get_dataset(f'dataset/macaw/walker_dir/buffers_walker_param_train_{task_id}_sub_task_0.hdf5')
    #     hfile = h5py.File(f'dataset/macaw/sac_walker_dir_{task_id}/expert.hdf5', 'w')
    #     for k in expert_data:
    #         hfile.create_dataset(k, data=expert_data[k], compression='gzip')
    #     hfile.close()
    #     hfile = h5py.File(f'dataset/macaw/sac_walker_dir_{task_id}/replay.hdf5', 'w')
    #     for k in replay_data:
    #         hfile.create_dataset(k, data=replay_data[k], compression='gzip')
    #     hfile.close()
    # for task_id in range(2):
    #     expert_data, replay_data = get_dataset(f'dataset/macaw/cheetah_dir/buffers_cheetah_dir_train_{task_id}_sub_task_0.hdf5')
    #     hfile = h5py.File(f'dataset/macaw/sac_cheetah_dir_{task_id}/expert.hdf5', 'w')
    #     for k in expert_data:
    #         hfile.create_dataset(k, data=expert_data[k], compression='gzip')
    #     hfile.close()
    #     hfile = h5py.File(f'dataset/macaw/sac_cheetah_dir_{task_id}/replay.hdf5', 'w')
    #     for k in replay_data:
    #         hfile.create_dataset(k, data=replay_data[k], compression='gzip')
    #     hfile.close()
    for task_id in range(5):
        expert_data, replay_data = get_dataset(f'dataset/macaw/cheetah_vel/buffers_cheetah_vel_train_{task_id}_sub_task_0.hdf5')
        hfile = h5py.File(f'dataset/macaw/sac_cheetah_vel_{task_id}/expert.hdf5', 'w')
        for k in expert_data:
            hfile.create_dataset(k, data=expert_data[k], compression='gzip')
        hfile.close()
        hfile = h5py.File(f'dataset/macaw/sac_cheetah_vel_{task_id}/replay.hdf5', 'w')
        for k in replay_data:
            hfile.create_dataset(k, data=replay_data[k], compression='gzip')
        hfile.close()
    print('finish')
