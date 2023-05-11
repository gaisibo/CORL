declare algo=("iql" "iql" "iqln")
declare expectile=("0.9" "0.7" "0.9")
declare expectile_min=("0.9" "0.7" "0.7")
declare expectile_max=("0.9" "0.7" "0.9")
declare dataset=("antmaze-umaze-v2" "antmaze-medium-play-v2" "antmaze-large-play-v2")
declare dataset_nums=("Ant_2_1-Ant_2_0" "Ant_5_4-Ant_5_3-Ant_5_2-Ant_5_1-Ant_5_0")
declare mix_type=("vq_diff" "random")
declare max_save_num=("100000" "30000")
for i in "${!algo[@]}"
do
        for k in "${!dataset[@]}"
        do
                        for l in "${!dataset_nums[@]}"
                        do
                                for m in "${!mix_type[@]}"
                                do
                                        for n in "${!max_save_num[@]}"
                                        do
echo "~/run_gpu_task.beta.1 -w compute\$1 -g gpu:1 -c 1 -j ${algo[$i]}_${expectile[$i]}_${expectile_min[$i]}_${expectile_max[$i]}_${dataset[$k]}_${dataset_nums[$l]}_orl_1_orl_1_clone_${mix_type[$m]}_${max_save_num[$n]}_0 -o single_output_files/output_${algo[$i]}_${expectile[$i]}_${expectile_min[$i]}_${expectile_max[$i]}_${dataset[$k]}_${dataset_nums[$l]}_orl_1_orl_1_clone_${mix_type[$m]}_${max_save_num[$n]}_0_\$(date +%Y%m%d%H%M%S).txt -i compute1:5000/co:11 -v /home/gaisibo/Continual-Offline/CCQL:/work \"cd /work; /root/miniconda3/bin/python3.7 continual_single.py --algo ${algo[$i]} --experience_type random_episode --dataset ${dataset[$k]} --dataset_nums ${dataset_nums[$l]} --max_save_num ${max_save_num[$n]} --critic_replay_type orl --critic_replay_lambda 1 --actor_replay_type orl --actor_replay_lambda 1 --clone clone --mix_type ${mix_type[$m]} --expectile ${expectile[$i]} --expectile_min ${expectile_min[$i]} --expectile_max ${expectile_max[$i]} --seed 0\"" > single_docker_files/docker_${algo[$i]}_${expectile[$i]}_${expectile_min[$i]}_${expectile_max[$i]}_${dataset[$k]}_${dataset_nums[$l]}_orl_1_orl_1_clone_${mix_type[$m]}_${max_save_num[$n]}_0.sh
                                        done
                                done
                        done
        done
done
