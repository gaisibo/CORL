import d3rlpy
from d3rlpy.preprocessing.reward_scalers import ConstantShiftRewardScaler


def get_st_dict(args, dataset, algo):
    st_dict = dict()
    st_dict['impl_name'] = args.algo
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
    st_dict['n_action_samples'] = args.n_action_samples
    st_dict['merge'] = args.merge
    st_dict['use_gpu'] = True
    st_dict['use_vae'] = args.use_vae
    online_st_dict = dict()
    online_st_dict['n_steps'] = 0
    online_st_dict['n_steps_per_epoch'] = 1000
    online_st_dict['buffer_size'] = 1000000
    if args.dataset_kind == 'd4rl':
        st_dict['n_steps'] = 500000
        st_dict['critic_update_step'] = 0
        st_dict['coldstart_steps'] = 500000
        st_dict['merge_n_steps'] = 200000
        st_dict['n_steps_per_epoch'] = 1000
        st_dict['batch_size'] = 256
        st_dict['vae_learning_rate'] = 1e-3
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
        if args.algo == 'sacn':
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 1e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['temp_learning_rate'] = 1e-4
            st_dict['n_critics'] = 10
        elif args.algo in ['iql', 'iqln']:
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
            if args.algo == 'iqln':
                st_dict['n_ensemble'] = 10
                st_dict['std_lamdba'] = 0.01
                st_dict['std_bias'] = 0
    elif args.dataset_kind == 'block':
        st_dict['n_steps'] = 500000
        st_dict['critic_update_step'] = 0
        st_dict['coldstart_steps'] = 500000
        st_dict['merge_n_steps'] = 200000
        st_dict['n_steps_per_epoch'] = 1000
        st_dict['batch_size'] = 256
        st_dict['vae_learning_rate'] = 1e-3
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
        if args.algo == 'sacn':
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 1e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['temp_learning_rate'] = 1e-4
            st_dict['n_critics'] = 10
        elif args.algo in ['iql', 'iqln']:
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
            if args.algo == 'iqln':
                st_dict['n_ensemble'] = 10
                st_dict['std_lamdba'] = 0.01
                st_dict['std_bias'] = 0
    elif args.dataset_kind == 'antmaze':
        st_dict['n_steps'] = 1000000
        st_dict['critic_update_step'] = 1000
        st_dict['coldstart_steps'] = 1000000
        st_dict['merge_n_steps'] = 1000000
        st_dict['n_steps_per_epoch'] = 10000
        st_dict['batch_size'] = 256
        st_dict['vae_learning_rate'] = 1e-3
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
        elif args.algo == 'sacn':
            encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
            st_dict['actor_encoder_factory'] = encoder
            st_dict['critic_encoder_factory'] = encoder
            st_dict['actor_learning_rate'] = 1e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['temp_learning_rate'] = 1e-4
            st_dict['n_critics'] = 10
        elif args.algo in ['iql', 'iqln']:
            st_dict['actor_encoder_factory'] = "default"
            st_dict['critic_encoder_factory'] = "default"
            st_dict['value_encoder_factory'] = "default"
            st_dict['actor_learning_rate'] = 3e-4
            st_dict['critic_learning_rate'] = 3e-4
            st_dict['weight_temp'] = 10.0
            st_dict['max_weight'] = 100.0
            st_dict['expectile'] = 0.9
            reward_scaler = ConstantShiftRewardScaler(shift=-1)
            st_dict['reward_scaler'] = reward_scaler
            if args.algo == 'iqln':
                st_dict['n_ensemble'] = 10
                st_dict['std_lamdba'] = 0.01
                st_dict['std_bias'] = 0
    return st_dict, online_st_dict
