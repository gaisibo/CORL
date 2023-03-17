declare -a algo=('iqln' 'iql')
declare -a sample_type=('none')
declare -a dataset=("halfcheetah-random-v0" "hopper-random-v0" "walker2d-random-v0") 
# declare -a dataset_nums=("500-0" "0-500")
declare -a dataset_nums=("0_0_2_500_1_2-500_0_2_0_1_2" "500_0_2_0_1_2-0_0_2_500_1_2" "0_0_2_100_1_2-100_0_2_0_1_2" "100_0_2_0_1_2-0_0_2_100_1_2" "100_0_2_500_1_2-500_0_2_100_1_2" "500_0_2_100_1_2-100_0_2_500_1_2" "100-500" "500-100" "500-0" "0-500" "100-0" "0-100" "500-0-0-0-0" "100-0-0-0-0" "0-500-0-0-0" "0-100-0-0-0")
declare -a critic_replay_type=("orl" "orl" "orl" "generate_orl" "orl")
declare -a critic_replay_lambda=("1" "1" "1" "1" "1")
declare -a actor_replay_type=("orl" "orl" "orl" "generate_orl" "none")
declare -a actor_replay_lambda=("1" "1" "1" "1" "0")
declare -a clone=("clone" "clone" "clone" "clone" "clone")
declare -a max_save_num=("100000" "10000" "1000" "0" "10000")
declare -a mix_type=("q" "random" "vq_diff")
declare -a expectile=("0.7" "0.9")
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
                                        for n in "${!expectile[@]}"
                                        do
                                                echo "
                                                python continual_single.py --algo ${algo[$k]} --experience_type random_episode --dataset ${dataset[$i]} --dataset_nums ${dataset_nums[$j]} --max_save_num ${max_save_num[$l]} --critic_replay_type ${critic_replay_type[$l]} --critic_replay_lambda ${critic_replay_lambda[$l]} --actor_replay_type ${actor_replay_type[$l]} --actor_replay_lambda ${actor_replay_lambda[$l]} --clone ${clone[$l]} --mix_type ${mix_type[$m]} --expectile ${expectile[$n]} --read_policy 0 --seed ${seed[$s]} > single_output_files/output_${algo[$k]}_${expectile[$n]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${clone[$l]}_${mix_type[$m]}_${max_save_num[$l]}_${seed[$s]}_\$(date +%Y%m%d%H%M%S).txt --gpu \$1" > single_run_files/run_${algo[$k]}_${expectile[$n]}_${dataset[$i]}_${dataset_nums[$j]}_${critic_replay_type[$l]}_${critic_replay_lambda[$l]}_${actor_replay_type[$l]}_${actor_replay_lambda[$l]}_${clone[$l]}_${mix_type[$m]}_${max_save_num[$l]}_${seed[$s]}.sh
                                        done
                                done
                        done
                done
        done
done
