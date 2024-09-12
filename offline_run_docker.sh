export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin;

bash offline_d4rl_run_file.sh --experience=coverage --dataset=mix --quality=medium_random --clone --device=0 &
bash offline_d4rl_run_file.sh --experience=single --dataset=mix --quality=medium_random --clone --device=1 &
bash offline_d4rl_run_file.sh --experience=max_reward_mean --dataset=mix --quality=medium_random --clone --device=2 &
bash offline_d4rl_run_file.sh --experience=random_transition --dataset=mix --quality=medium_random --clone --device=3 &
bash offline_d4rl_run_file.sh --experience=max_model_mean --dataset=mix --quality=medium_random --clone --device=4 &
bash offline_d4rl_run_file.sh --experience=max_match_mean --dataset=mix --quality=medium_random --clone --device=5 &
bash offline_d4rl_run_file.sh --experience=max_supervise_mean --dataset=mix --quality=medium_random --clone --device=6 &
bash offline_d4rl_run_file.sh --experience=model_next --dataset=mix --quality=medium_random --clone --device=7 &

wait
