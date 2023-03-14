declare -a algo=('iql' 'cql' 'sacn' 'iqln')
declare -a sample_type=('none')
declare -a dataset=("antmaze-large-diverse-v2" "antmaze-large-play-v2" "antmaze-medium-diverse-v2" "antmaze-medium-play-v2" "antmaze-umaze-diverse-v2" "antmaze-umaze-v2") 
declare -a dataset_nums=("d4rl_2_1-d4rl_2_0" "d4rl_2_0-d4rl_2_1" "d4rl_5_4-d4rl_5_3-d4rl_5_2-d4rl_5_1-d4rl_5_0" "d4rl_5_0-d4rl_5_1-d4rl_5_2-d4rl_5_3-d4rl_5_4")
declare -a critic_replay_type=("orl" "orl" "orl")
declare -a critic_replay_lambda=("1" "1" "1")
declare -a actor_replay_type=("orl" "orl" "orl")
declare -a actor_replay_lambda=("1" "1" "1")
declare -a clone=("none" "none" "none")
declare -a max_save_num=("100000" "10000" "1000")
declare -a mix_type=("q" "random")
declare -a seed=('0')
for i in "${!dataset[@]}"
do
        for j in "${!dataset_nums[@]}"
        do
                for k in ${!algo[@]}
                do
                        for l in "${!critic_replay_type[@]}"
                        do
                                for m in "${!mix_type[@]}"
                                do
                                        echo "
                                        python continual_single.py --algo ${algo[$k]} --experience_type random_episode --dataset ${dataset[$i]} --dataset_nums ${dataset_nums[$j]} --max_save_num ${max_save_num[$l]} --critic_replay_type ${critic_replay_type[$l]} --critic_replay_lambda ${critic_replay_lambda[$l]} --actor_replay_type ${actor_replay_type[$l]} --actor_replay_lambda ${actor_replay_lambda[$l]} --clone ${clone[$l]} --mix_type ${mix_type[$m]} --seed ${seed[$s]} > single_output_files/output_${algo[$k]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${clone[$l]}_${max_save_num[$l]}_${seed[$s]}_\$(date +%Y%m%d%H%M%S).txt --gpu \$1" > single_run_files/run_${algo[$k]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${clone[$l]}_${mix_type[$m]}_${max_save_num[$l]}_${seed[$s]}.sh
                                done
                        done
                done
        done
done
