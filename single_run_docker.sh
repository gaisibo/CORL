export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin;

bash single_d4rl_run_file.sh --algo=iqln --n_ensemble=100 --expectile=0.99 --dataset=ant-random-v0 --dataset_nums=0_0_0_300_300_300_0_0_0 --actor_replay_type=orl --critic_replay_type=orl --max_save_num=30 --mix_type=q --n_steps=50000 --clone_actor --gpu=0 &
bash single_d4rl_run_file.sh --algo=iql --n_ensemble=2 --expectile=0.9 --dataset=ant-random-v0 --dataset_nums=0_0_0_300_300_300_0_0_0 --actor_replay_type=orl --critic_replay_type=orl --max_save_num=30 --mix_type=q --n_steps=50000 --clone_actor --gpu=1 &
bash single_d4rl_run_file.sh --algo=td3_plus_bc --n_ensemble=2 --dataset=ant-random-v0 --dataset_nums=0_0_0_300_300_300_0_0_0 --actor_replay_type=orl --critic_replay_type=orl --max_save_num=30 --mix_type=q --n_steps=50000 --clone_actor --gpu=2 &
bash single_d4rl_run_file.sh --algo=sacn --n_ensemble=100 --dataset=ant-random-v0 --dataset_nums=0_0_0_300_300_300_0_0_0 --actor_replay_type=orl --critic_replay_type=orl --max_save_num=30 --mix_type=q --n_steps=50000 --clone_actor --gpu=3 &
bash single_d4rl_run_file.sh --algo=edac --n_ensemble=10 --dataset=ant-random-v0 --dataset_nums=0_0_0_300_300_300_0_0_0 --actor_replay_type=orl --critic_replay_type=orl --max_save_num=30 --mix_type=q --n_steps=50000 --clone_actor --gpu=4 &

wait
