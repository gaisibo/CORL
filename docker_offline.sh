declare -a replay=("none" "all" "orl" "bc" "ewc" "gem" "agem" "rwalk" "si")
declare -a experience_type=("all" "none" "single" "online" "model_next" "coverage" "random_transition" "random_epsisode" "max_reward_mean" "max_match_mean" "max_supervise_mean" "max_model_mean" "max_reward_mean" "max_supervise_mean")
declare -a dataset=("ant_dir" "walker_dir" "cheetah_vel")
declare -a quality=("medium" "medium_random")
declare -a buffer=("10000" "1000")
declare -a lambda=("3" "1" "0.3")
declare -a distance_type=("l2" "feature")
for l in "${!buffer[@]}"
do
        for m in "${!lambda[@]}"
        do
                for n in "${!dataset[@]}"
                do
                        for o in "${!quality[@]}"
                        do
                                for p in "${!distance_type[@]}"
                                do
                                        echo "~/run_gpu_task.beta.1 -w compute\$1 -g gpu:1 -c 1 -j bc_${distance_type[$p]}_model_next_${dataset[$n]}_${quality[$o]}_1_1000_clone_\$2 -o output_files/output_bc_${distance_type[$p]}_model_next_${dataset[$n]}_${quality[$o]}_1_1000_clone_\$2_\$(date +%Y%m%d%H%M%S).txt -i compute1:5000/co:11 -v /home/gaisibo/Continual-Offline/CCQL:/work \"cd /work; /root/miniconda3/bin/python3.7 continual_offline.py --task_nums 5 --experience_type model_next --replay_type bc --distance_type ${distance_type[$p]} --dataset ${dataset[$n]} --quality ${quality[$o]} --max_save_num ${buffer[$l]} --replay_alpha ${lambda[$m]} --read_policies -1 --clone_actor --seed \$2\"" > docker_files/docker_bc_${distance_type[$p]}_model_next_${dataset[$n]}_${quality[$o]}_${lambda[$m]}_${buffer[$l]}_clone.sh
                                done
                                for i in "${!replay[@]}"
                                do
                                        echo "~/run_gpu_task.beta.1 -w compute\$1 -g gpu:1 -c 1 -j ${replay[$i]}_l2_model_next_${dataset[$n]}_${quality[$o]}_1_1000_clone_\$2 -o output_files/output_${replay[$i]}_l2_model_next_${dataset[$n]}_${quality[$o]}_1_1000_clone_\$2_\$(date +%Y%m%d%H%M%S).txt -i compute1:5000/co:11 -v /home/gaisibo/Continual-Offline/CCQL:/work \"cd /work; /root/miniconda3/bin/python3.7 continual_offline.py --task_nums 5 --experience_type model_next --replay_type ${replay[$i]} --distance_type l2 --dataset ${dataset[$n]} --quality ${quality[$o]} --max_save_num ${buffer[$l]} --replay_alpha ${lambda[$m]} --read_policies -1 --clone_actor --seed \$2\"" > docker_files/docker_${replay[$i]}_l2_model_next_${dataset[$n]}_${quality[$o]}_${lambda[$m]}_${buffer[$l]}_clone.sh
                                done
                                echo "~/run_gpu_task.beta.1 -w compute\$1 -g gpu:1 -c 1 -j bc_l2_model_next_${dataset[$n]}_${quality[$o]}_1_1000_\$2 -o output_files/output_bc_l2_model_next_${dataset[$n]}_${quality[$o]}_1_1000_\$2_\$(date +%Y%m%d%H%M%S).txt -i compute1:5000/co:11 -v /home/gaisibo/Continual-Offline/CCQL:/work \"cd /work; /root/miniconda3/bin/python3.7 continual_offline.py --task_nums 5 --experience_type model_next --replay_type bc --distance_type l2 --dataset ${dataset[$n]} --quality ${quality[$o]} --max_save_num ${buffer[$l]} --replay_alpha ${lambda[$m]} --read_policies -1 --seed \$2\"" > docker_files/docker_bc_l2_model_next_${dataset[$n]}_${quality[$o]}_${lambda[$m]}_${buffer[$l]}.sh
                                echo "~/run_gpu_task.beta.1 -w compute\$1 -g gpu:1 -c 1 -j fix_l2_model_next_${dataset[$n]}_${quality[$o]}_1_1000_\$2 -o output_files/output_fix_l2_model_next_${dataset[$n]}_${quality[$o]}_1_1000_\$2_\$(date +%Y%m%d%H%M%S).txt -i compute1:5000/co:11 -v /home/gaisibo/Continual-Offline/CCQL:/work \"cd /work; /root/miniconda3/bin/python3.7 continual_offline.py --task_nums 5 --experience_type model_next --replay_type fix --distance_type l2 --dataset ${dataset[$n]} --quality ${quality[$o]} --max_save_num ${buffer[$l]} --replay_alpha ${lambda[$m]} --read_policies -1 --seed \$2\"" > docker_files/docker_fix_l2_model_next_${dataset[$n]}_${quality[$o]}_${lambda[$m]}_${buffer[$l]}.sh
                        done
                done
        done
done
for i in "${!experience_type[@]}"
do
        for l in "${!buffer[@]}"
        do
                for m in "${!lambda[@]}"
                do
                        for n in "${!dataset[@]}"
                        do
                                for o in "${!quality[@]}"
                                do
                                echo "~/run_gpu_task.beta.1 -w compute\$1 -g gpu:1 -c 1 -j bc_l2_${experience_type[$i]}_${dataset[$n]}_${quality[$o]}_1_1000_clone_\$2 -o output_files/output_bc_l2_${experience_type[$i]}_no_0_none_${dataset[$n]}_${quality[$o]}_1_1000_clone_\$2_\$(date +%Y%m%d%H%M%S).txt -i compute1:5000/co:11 -v /home/gaisibo/Continual-Offline/CCQL:/work \"cd /work; /root/miniconda3/bin/python3.7 continual_offline.py --task_nums 5 --experience_type ${experience_type[$i]} --replay_type bc --distance_type l2 --dataset ${dataset[$n]} --quality ${quality[$o]} --max_save_num ${buffer[$l]} --replay_alpha ${lambda[$m]} --read_policies -1 --clone_actor --seed \$2\"" > docker_files/docker_bc_l2_${experience_type[$i]}_${dataset[$n]}_${quality[$o]}_${lambda[$m]}_${buffer[$l]}_clone.sh
                                done
                        done
                done
        done
done
