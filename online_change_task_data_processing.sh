#!/usr/bin/bash
#awk -f online_change_task_data_processing.awk online_change_task.$1.log | sed "s/=//g" | sed "s/\}//g" | sed "s/epoch//g" | gnuplot -e "plot '-' w lp; pause 99"
rm logs/*latest*
rm pics/*
bash read_logs.sh
algorithms_offline=( "iql" "iqln" )
algorithms_online=( "iql_online" "iqln_online" )
algorithms_all=( ${algorithms_offline[@]} ${algorithms_online[@]} )
for algorithm1 in "${algorithms_offline[@]}"; do
    for algorithm2 in "${algorithms_online[@]}"; do
        if [[ $algorithm1 == $algorithm2 ]]; then
            continue;
        elif [[ $algorithm1 == "iqln" && $algorithm2 != "iqln_online" ]]; then
            continue;
        elif [[ $algorithm1 != "iqln" && $algorithm2 == "iqln_online" ]]; then
            continue;
        fi
        if [[ $algorithm1 == 'iqln' && $algorithm2 == "iqln_online" ]]; then
            n_ensembles=( "2" )
        else
            n_ensembles=( "2" "10" )
        fi
        for n_ensemble in "${n_ensembles[@]}"; do
            algorithms="${algorithm1}-${algorithm2}_${n_ensemble}"
            datasets=( "halfcheetah" "hopper" "walker2d")
            for dataset in ${datasets[@]}; do
                has_online=0
                quality1_list=( "medium" "expert" "random" "medium_expert" "medium_replay" )
                quality2_list=( "medium" )
                has_online=1
                copy_buffer_list=( "none" "copy" "mix_all" "mix_same" "ewc_same" "ewc_all" )
                buffer_list=( "20000" "2000000" )
                for quality1 in "${quality1_list[@]}" ; do
                    for quality2 in "${quality2_list[@]}" ; do
                        qualities="${quality1}-${quality2}"
                        first="1000000"
                        second="1000000"
                        for buffer in "${buffer_list[@]}"; do
                            for copy_buffer in ${copy_buffer_list[@]}; do
                                if [[ $copy_buffer == "mix_same" || $copy_buffer == "mix_all" ]]; then
                                    buffer_mix_type_list=( "_all" )
                                else
                                    buffer_mix_type_list=( "" )
                                fi
                                for buffer_mix_type in "${buffer_mix_type_list[@]}"; do
                                    copy_buffer_str=$(echo ${copy_buffer} | sed "s/\_/\\\_/g")
                                    buffer_mix_type_str=$(echo ${buffer_mix_type} | sed "s/\_/\\\_/g")
                                    for copy_optim in "_copy_optim"; do
                                        TMPFILE1=$(mktemp) || exit 1
                                        echo "${algorithms_offline[@]}" | grep -wq ${algorithm1} && log_name=logs/online_change_task_${dataset}_${quality1}_${algorithm1}_${n_ensemble}_${first}.latest.log || log_name=logs/online_change_task_${dataset}_${algorithm1}_${first}_${buffer}.latest.log
                                        if [[ -f $log_name ]]; then
                                            #echo $log_name
                                            awk -f online_change_task_data_processing.awk $log_name | sed "s/=//g" | sed "s/\}//g" > TMPFILE1
                                        else
                                            echo $log_name not exist
                                            continue
                                        fi
                                        TMPFILE2=$(mktemp) || exit 1
                                        #log_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}${copy_optim}_${copy_buffer}.latest.log
                                        log_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}${explore}${copy_optim}_${copy_buffer}${buffer_mix_type}.latest.log
                                        if [[ -f $log_name ]]; then
                                            #echo $log_name
                                            awk -f online_change_task_data_processing.awk $log_name | sed "s/=//g" | sed "s/\}//g" > TMPFILE2
                                        else
                                            echo $log_name not exist
                                            continue
                                        fi
                                        pic_name=pics/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}${explore}${copy_optim}_${copy_buffer}${buffer_mix_type}.png
                                        txt_name=pics/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}${explore}${copy_optim}_${copy_buffer}${buffer_mix_type}.txt
                                        #cat TMPFILE1 TMPFILE2 | gnuplot -e "set terminal png; set title '${dataset}\_${qualities}\_${algorithms}\_${first}\_${second}\_${buffer}\_${copy_optim_str}\_${copy_buffer_str}'; plot '-' using 1 with lines;" > $pic_name
                                        algorithms_str=$(echo ${algorithms} | sed "s/\_/\\\_/g")
                                        copy_optim_str=$(echo ${copy_optim} | sed "s/\_/\\\_/g")
                                        explore_str=$(echo ${explore} | sed "s/\_/\\\_/g")
                                        cat TMPFILE1 TMPFILE2 | gnuplot -e "set terminal png; set title '${qualities}\_${algorithms_str}\_${buffer}${explore_str}${copy_optim_str}\_${copy_buffer_str}${buffer_mix_type_str}'; plot '-' using 1 with lines;" > $pic_name
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
rm TMPFILE1
rm TMPFILE2
# | sed "s/,//g"
