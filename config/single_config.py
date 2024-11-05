import d3rlpy


def get_st_dict(args, dataset, algo, quality):
    st_dict = dict()
    st_dict['impl_name'] = algo
    st_dict['critic_replay_type'] = args.critic_replay_type
    st_dict['critic_replay_lambda'] = args.critic_replay_lambda
    st_dict['actor_replay_type'] = args.actor_replay_type
    st_dict['actor_replay_lambda'] = args.actor_replay_lambda
    st_dict['vae_replay_lambda'] = args.vae_replay_lambda
    st_dict['vae_replay_type'] = args.vae_replay_type
    st_dict['clone_actor'] = args.clone_actor
    st_dict['clone_critic'] = args.clone_critic
    st_dict['fine_tuned_step'] = args.fine_tuned_step
    st_dict['experience_type'] = args.experience_type
    # st_dict['merge'] = args.merge
    st_dict['use_gpu'] = True
    st_dict['use_vae'] = args.use_vae
    online_st_dict = dict()
    online_st_dict['n_steps'] = 0
    online_st_dict['n_steps_per_epoch'] = 1000
    online_st_dict['buffer_size'] = 1000000
    step_dict = dict()
    if dataset == 'd4rl':
        st_dict['batch_size'] = 256
        if args.algo_kind == 'cql':
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 1e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['alpha_learning_rate'] = 0.0
            st_dict['n_action_samples'] = 10
            if 'medium' in quality:
                st_dict['conservative_weight'] = 10.0
            else:
                st_dict['conservative_weight'] = 5.0
        if args.algo_kind in ['td3_plus_bc', 'td3']:
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 3e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['target_smoothing_sigma'] = 0.2
            st_dict['target_smoothing_clip'] = 0.5
            st_dict['update_actor_interval'] = 2
        elif algo in ['sac', 'sacn', 'edac']:
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 3e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['temp_learning_rate'] = 3e-4
        elif algo in ['iql', 'iqln', 'iqln2', 'iqln3', 'iqln4', 'sql', 'sqln']:
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['value_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 3e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['weight_temp'] = 3.0
            st_dict['max_weight'] = 100.0
            st_dict['expectile'] = 0.7
            reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
                multiplier=1000.0)
            st_dict['reward_scaler'] = reward_scaler
            if algo in ['iqln', 'iqln2', 'iqln3', 'iqln4', 'sqln']:
                st_dict['n_ensemble'] = 10
                st_dict['entropy_time'] = 0.2
                st_dict['std_time'] = 1
                st_dict['std_type'] = 'clamp'
                if algo == 'iqln2':
                    st_dict['update_ratio'] = 0.3
    elif dataset == 'block':
        step_dict['n_steps'] = 500000
        step_dict['coldstart_steps'] = 500000
        # step_dict['merge_n_steps'] = 200000
        step_dict['n_steps_per_epoch'] = 1000
        st_dict['critic_update_step'] = 0
        st_dict['batch_size'] = 256
        st_dict['vae_learning_rate'] = 1e-3
        st_dict['n_action_samples'] = args.n_action_samples
        if args.algo_kind == 'cql':
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 1e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['temp_learning_rate'] = 1e-4
            st_dict['alpha_learning_rate'] = 0.0
            st_dict['conservative_weight'] = 5.0
            st_dict['conservative_threshold'] = 0.1
            st_dict['std_time'] = 1
            st_dict['std_type'] = 'clamp'
        elif algo in ['sacn', 'edac', 'n_critics']:
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 1e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['temp_learning_rate'] = 1e-4
            st_dict['n_critics'] = args.n_critics
        elif algo in ['iql', 'iqln', 'iqln2', 'iqln3', 'iqln4']:
            st_dict['actor_encoder_factory'] = "default"
            st_dict['critic_encoder_factory'] = "default"
            st_dict['value_encoder_factory'] = "default"
            st_dict['actor_learning_rate'] = 3e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['weight_temp'] = 3.0
            st_dict['max_weight'] = 100.0
            st_dict['expectile'] = 0.7
            reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
                multiplier=1000.0)
            st_dict['reward_scaler'] = reward_scaler
            if algo in ['iqln', 'iqln2', 'iqln3', 'iqln4']:
                st_dict['n_ensemble'] = 10
                st_dict['std_lamdba'] = 0.01
                st_dict['std_bias'] = 0
    elif dataset == 'antmaze':
        step_dict['n_steps'] = 1000000
        step_dict['coldstart_steps'] = 1000000
        # step_dict['merge_n_steps'] = 1000000
        step_dict['n_steps_per_epoch'] = 100000
        st_dict['critic_update_step'] = 0
        st_dict['batch_size'] = 256
        st_dict['vae_learning_rate'] = 1e-3
        st_dict["n_critics"] = n_critics
        if args.algo_kind == 'cql':
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 1e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['temp_learning_rate'] = 1e-4
            st_dict['alpha_learning_rate'] = 0.0
            st_dict['conservative_weight'] = 5.0
            st_dict['conservative_threshold'] = 0.1
            st_dict['std_time'] = 1
            st_dict['std_type'] = 'clamp'
        if args.algo_kind in ['td3_plus_bc', 'td3']:
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 1e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['temp_learning_rate'] = 1e-4
            st_dict['alpha_learning_rate'] = 0.0
            st_dict['conservative_weight'] = 5.0
            st_dict['conservative_threshold'] = 0.1
            st_dict['std_time'] = 1
            st_dict['std_type'] = 'clamp'
        elif algo in ['sacn', 'edac', 'sac', 'td3']:
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 1e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['temp_learning_rate'] = 1e-4
            st_dict['n_critics'] = args.n_critics
        elif algo in ['iql', 'iqln', 'iqln2', 'iqln3', 'iqln4']:
            st_dict['actor_encoder_factory'] = "default"
            st_dict['critic_encoder_factory'] = "default"
            st_dict['value_encoder_factory'] = "default"
            st_dict['actor_learning_rate'] = 3e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['weight_temp'] = 10.0
            st_dict['max_weight'] = 100.0
            st_dict['expectile'] = 0.9
            reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(multiplier=1000.0)
            st_dict['reward_scaler'] = reward_scaler
            if algo in ['iqln', 'iqln2', 'iqln3', 'iqln4']:
                st_dict['n_ensemble'] = 10
                st_dict['std_time'] = 1
                st_dict['std_type'] = 'clamp'
    return st_dict, online_st_dict, step_dict
