import d3rlpy


online_algos = ['td3', 'sac', 'iql_online', 'iqln_online']
offline_algos = ['iql', 'iqln', 'cql', 'cal', 'td3_plus_bc']
def get_o2o_dict(algo, quality):
    o2o_dict = dict()
    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
    o2o_dict["actor_encoder_factory"] = encoder;
    o2o_dict["critic_encoder_factory"] = encoder;
    o2o_dict['batch_size'] = 256
    o2o_dict['n_action_samples'] = 10
    if algo in ['cql', 'cal']:
        if "medium" in quality:
            conservative_weight = 10.0
        else:
            conservative_weight = 5.0
        o2o_dict["actor_learning_rate"] = 3e-5
        o2o_dict["critic_learning_rate"] = 3e-4
        o2o_dict["temp_learning_rate"] = 1e-4
        o2o_dict["alpha_learning_rate"] = 0.0
        o2o_dict['conservative_weight'] = conservative_weight
    elif algo == 'sac':
        o2o_dict["actor_learning_rate"] = 3e-4
        o2o_dict["critic_learning_rate"] = 3e-4
        o2o_dict["temp_learning_rate"] = 1e-4
    elif algo in ['td3', 'td3_plus_bc']:
        o2o_dict["actor_learning_rate"] = 3e-4
        o2o_dict["critic_learning_rate"] = 3e-4
        o2o_dict["target_smooth_sigma"] = 0.2
        o2o_dict["target_smooth_clip"] = 0.5
        o2o_dict["update_actor_interval"] = 2
        o2o_dict['policy_noise'] = 0.2
        if algo == 'td3_plus_bc':
            o2o_dict['alpha'] = 2.5
            #o2o_dict['scaler'] = 'standard'
    elif algo in ['iql', 'iqln', 'iql_online', 'iqln_online']:
        if algo in ['iql', 'iqln']:
            reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(multiplier=1000.0)
            o2o_dict["reward_scaler"] = reward_scaler
        o2o_dict["actor_learning_rate"] = 3e-4
        o2o_dict["critic_learning_rate"] = 3e-4
        o2o_dict["weight_temp"] = 3.0
        o2o_dict["max_weight"] = 100.0
        if algo in ['iqln', 'iqln_online']:
            o2o_dict["n_ensemble"] = 10
    elif algo in ['ppo', 'bppo']:
        reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(multiplier=1000.0)
        o2o_dict["actor_learning_rate"] = 3e-4
        o2o_dict["critic_learning_rate"] = 3e-4
        o2o_dict["weight_temp"] = 3.0
        o2o_dict["max_weight"] = 100.0
        o2o_dict["reward_scaler"] = reward_scaler

        o2o_dict['const_eps'] = 1e-10
        o2o_dict['is_clip_decay'] = True
        o2o_dict['is_lr_decay'] = True
        o2o_dict['update_critic'] = False
        o2o_dict['update_critic_interval'] = 2
        o2o_dict['update_old_policy'] = True
        o2o_dict['update_old_policy_interval'] = 10
    else:
        raise NotImplementedError
    return o2o_dict
