declare -a algo=('td3_plus_bc' 'cql' 'combo')
declare -a short_algo=('t' 'c' 'm')
declare -a replay_type=('bc' 'orl' 'ewc' 'gem' 'agem')
declare -a experience=('model' 'coverage' 'random_episode' 'max_match_mean' 'max_reward_mean' 'max_model_mean' 'random_transition' 'max_match' 'max_reward' 'max_model')
declare -a short_experience=('m__' 'r__' 'c__' 're_' 'mpm' 'mrm' 'mdm' 'rt_' 'mp_' 'mr_' 'md_')
declare -a generate_type=('model')
declare -a short_generate=('m')
declare -a sample_type=('none')
declare -a short_sample_type=('none')
declare -a dataset=("ant_dir_medium" "ant_dir_random" "walker_dir_medium" "walker_dir_random" "cheetah_dir_medium" "cheetah_dir_random" "cheetah_vel_medium" "cheetah_vel_random") 
declare -a short_dataset=("adm" "adr" "wdm" "wdr" "cdm" "cdr" "cvm" "cvr")
declare -a task_nums=("5" "5" "5" "5" "2" "2" "5" "5")
declare -a inner_path=("sac_ant_dir_num/medium.hdf5" "sac_ant_dir_num/random.hdf5" "sac_walker_dir_num/medium.hdf5" "sac_walker_dir_num/random.hdf5" "sac_cheetah_dir_num/medium.hdf5" "sac_cheetah_dir_num/random.hdf5" "sac_cheetah_vel_num/medium.hdf5" "sac_cheetah_vel_num/random.hdf5") 
declare -a env_path=("ant_dir/env_ant_dir_train_tasknum.pkl" "ant_dir/env_ant_dir_train_tasknum.pkl" "walker_dir/env_walker_param_train_tasknum.pkl" "walker_dir/env_walker_param_train_tasknum.pkl" "cheetah_dir/env_cheetah_dir_train_tasknum.pkl" "cheetah_dir/env_cheetah_dir_train_tasknum.pkl" "cheetah_vel/env_cheetah_vel_train_tasknum.pkl" "cheetah_vel/env_cheetah_vel_train_tasknum.pkl")
declare -a replay_alpha=('1')
declare -a max_save_num=('1000' '10000' '100000')
declare -a generate_step=('1' '3')
declare -a computer=("compute3" "compute3" "compute3" "compute3" "compute3" "compute3" "compute3" "compute3") 
for a in "${!dataset[@]}"
do
        for b in ${!algo[@]}
        do
                for j in "${!replay_alpha[@]}"
                do
                        for m in "${!max_save_num[@]}"
                        do
                                for i in "${!experience[@]}"
                                do
                                        for n in "${!replay_type[@]}"
                                        do
                                                for k in "${!sample_type[@]}"
                                                do
                                                        for l in "${!generate_type[@]}"
                                                        do
                                                                echo "
                                                                        python ccql.py --algo ${algo[$b]} --env_path ${env_path[$a]} --inner_path ${inner_path[$a]} --task_nums ${task_nums[$a]} --experience_type ${experience[$i]} --sample_type ${sample_type[$k]} --replay_type ${replay_type[$n]} --dataset ${dataset[$a]} --max_save_num ${max_save_num[$m]} --replay_alpha ${replay_alpha[$j]} --generate_type ${generate_type[$l]} --generate_step 0 --read_policies -1 | tee output_${algo[$b]}_${replay_type[$n]}_${experience[$i]}_${generate_type[$l]}_0_${sample_type[$k]}_${dataset[$a]}_${replay_alpha[$j]}_${max_save_num[$m]}.txt
                                                                        " > "run_files/run_${algo[$b]}_${replay_type[$n]}_${experience[$i]}_${generate_type[$l]}_0_${sample_type[$k]}_${dataset[$a]}_${replay_alpha[$j]}_${max_save_num[$m]}.sh"
                                                        done
                                                done
                                                echo "
                                                        python ccql.py --algo ${algo[$b]} --env_path ${env_path[$a]} --inner_path ${inner_path[$a]} --task_nums ${task_nums[$a]} --experience_type ${experience[$i]} --sample_type none --replay_type ${replay_type[$n]} --dataset ${dataset[$a]} --max_save_num ${max_save_num[$m]} --replay_alpha ${replay_alpha[$j]} --generate_type no --generate_step 0 --read_policies -1 --gpu \$1 | tee output_${algo[$b]}_${replay_type[$n]}_${experience[$i]}_no_0_none_${dataset[$a]}_${replay_alpha[$j]}_${max_save_num[$m]}.txt
                                                        " > "run_files/run_${algo[$b]}_${replay_type[$n]}_${experience[$i]}_no_0_none_${dataset[$a]}_${replay_alpha[$j]}_${max_save_num[$m]}.sh"
                                        done
                                done
                        done
                done
        done
done
