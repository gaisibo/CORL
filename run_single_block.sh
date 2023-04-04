declare -a algo=('iqln' 'iql' 'iql')
declare -a expectile=('0.9' '0.9' '0.99')
declare -a expectile_min=('0.9' '0.9' '0.99')
declare -a expectile_max=('0.99' '0.9' '0.99')
declare -a sample_type=('none')
declare -a dataset=("HalfCheetahBlock-v2" "HopperBlock-v4" "Walker2dBlock-v4") 
# declare -a dataset_nums=("500-0" "0-500")
declare -a dataset_nums=("3000-0-0-0-0" "400-0-0-0-0" "0-3000-0-0-0" "0-400-0-0-0")
declare -a critic_replay_type=("orl" "orl" "orl" "orl")
declare -a critic_replay_lambda=("1" "1" "1" "1")
declare -a actor_replay_type=("orl" "orl" "orl" "none")
declare -a actor_replay_lambda=("1" "1" "1" "0")
declare -a clone=("clone" "clone" "clone" "clone")
declare -a max_save_num=("100000" "50000" "10000" "10000")
declare -a mix_type=("random" "vq_diff")
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
                                        echo "python continual_single.py --algo ${algo[$k]} --experience_type random_episode --dataset ${dataset[$i]} --dataset_nums ${dataset_nums[$j]} --max_save_num ${max_save_num[$l]} --critic_replay_type ${critic_replay_type[$l]} --critic_replay_lambda ${critic_replay_lambda[$l]} --actor_replay_type ${actor_replay_type[$l]} --actor_replay_lambda ${actor_replay_lambda[$l]} --clone ${clone[$l]} --mix_type ${mix_type[$m]} --expectile ${expectile[$k]} --expectile_min ${expectile_min[$k]} --expectile_max ${expectile_max[$k]} --seed ${seed[$s]} > single_output_files/output_${algo[$k]}_${expectile[$k]}_${expectile_min[$k]}_${expectile_max[$k]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${clone[$l]}_${mix_type[$m]}_${max_save_num[$l]}_${seed[$s]}_\$(date +%Y%m%d%H%M%S).txt --gpu \$1" > single_run_files/run_${algo[$k]}_${expectile[$k]}_${expectile_min[$k]}_${expectile_max[$k]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${clone[$l]}_${mix_type[$m]}_${max_save_num[$l]}_${seed[$s]}.sh
                                done
                        done
                done
        done
done
