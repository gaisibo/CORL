declare -a algo=('mrcql')
declare -a experience=('random_episode')
declare -a sample_type=('none')
declare -a dataset=("halfcheetah-random-v0") 
declare -a dataset_nums=("500-0")
declare -a critic_replay_type=("bc")
declare -a critic_replay_lambda=("100")
declare -a actor_replay_type=("orl")
declare -a actor_replay_lambda=("1")
declare -a max_save_num=('10000')
declare -a seed=('0')
for i in "${!dataset[@]}"
do
        for j in "${!dataset_nums[@]}"
        do
                for k in ${!algo[@]}
                do
                        for l in "${!critic_replay_type[@]}"
                        do
                                for m in "${!max_save_num[@]}"
                                do
                                        ~/run_gpu_task.beta.1 -w compute$1 -g gpu:1 -c 1 -j ${algo[$k]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${max_save_num[$m]}_${seed[$s]} -o single_log_files/output_${algo[$k]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${max_save_num[$m]}_${seed[$s]}_$(date +%Y%m%d%H%M%S).txt -i compute1:5000/co:v8 -v /home/gaisibo/Continual-Offline/CCQL:/work "cd /work; /root/miniconda3/bin/python3.7 continual_single.py --algo ${algo[$k]} --experience_type random_episode --dataset ${dataset[$i]} --dataset_nums ${dataset_nums[$j]} --max_save_num ${max_save_num[$m]} --critic_replay_type ${critic_replay_type[$l]} --critic_replay_lambda ${critic_replay_lambda[$l]} --actor_replay_type ${actor_replay_type[$l]} --actor_replay_lambda ${actor_replay_lambda[$l]} --read_policies -1 --seed ${seed[$s]} > single_output_files/output_${algo[$k]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${max_save_num[$m]}_${seed[$s]}_$(date +%Y%m%d%H%M%S).txt"
                                done
                        done
                done
        done
done