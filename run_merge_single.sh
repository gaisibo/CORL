declare -a algo=('iql')
declare -a sample_type=('none')
declare -a dataset=("halfcheetah-random-v0") 
# declare -a dataset_nums=("500-0" "0-500")
declare -a dataset_nums=("500-0")
declare -a critic_replay_type=("bc")
declare -a critic_replay_lambda=("100")
declare -a actor_replay_type=("bc")
declare -a actor_replay_lambda=("100")
declare -a clone=("none")
declare -a max_save_num=('100000')
declare -a seed=('0')
for i in "${!dataset[@]}"
do
        for j in "${!dataset_nums[@]}"
        do
                for k in ${!algo[@]}
                do
                        for l in "${!critic_replay_type[@]}"
                        do
                                echo "
                                python continual_single.py --algo ${algo[$k]} --experience_type random_episode --dataset ${dataset[$i]} --dataset_nums ${dataset_nums[$j]} --max_save_num ${max_save_num[$l]} --critic_replay_type ${critic_replay_type[$l]} --critic_replay_lambda ${critic_replay_lambda[$l]} --actor_replay_type ${actor_replay_type[$l]} --actor_replay_lambda ${actor_replay_lambda[$l]} --clone ${clone[$l]} --read_policy 0 --read_merge_policy 1 --merge --seed ${seed[$s]} > single_output_files/merge_${algo[$k]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${clone[$l]}_${max_save_num[$l]}_${seed[$s]}_\$(date +%Y%m%d%H%M%S).txt --gpu \$1" > single_run_files/merge_${algo[$k]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${clone[$l]}_${max_save_num[$l]}_${seed[$s]}.sh
                        done
                done
        done
done
