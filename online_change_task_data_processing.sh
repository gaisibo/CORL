#!/usr/bin/bash
#awk -f online_change_task_data_processing.awk online_change_task.$1.log | sed "s/=//g" | sed "s/\}//g" | sed "s/epoch//g" | gnuplot -e "plot '-' w lp; pause 99"
rm logs/*latest*
rm pics/*
bash read_logs.sh
algorithms_offline=( 'iql' 'cql' 'cal' 'td3_plus_bc' )
algorithms_online=( 'td3' 'sac' )
algorithms=( ${algorithms_offline[@]} ${algorithms_online[@]} )
for algorithm1 in "${algorithms[@]}"; do
    for algorithm2 in "${algorithms[@]}"; do
        if [[ $algorithm1 == $algorithm2 ]]; then
            continue;
        fi
        algorithms="${algorithm1}-${algorithm2}"
        for dataset in 'halfcheetah'; do
            has_online=0
            if [[ "${algorithms_offline[@]}" =~ ${algorithm1} ]]; then
                quality1_list=( 'medium' 'expert' )
            else
                has_online=1
                quality1_list=( 'medium' )
            fi
            if [[ "${algorithms_offline[@]}" =~ ${algorithm1} ]]; then
                quality2_list=( 'medium' 'expert' )
            else
                has_online=1
                quality2_list=( 'medium' )
            fi
            if [[ $has_online -eq 1 ]]; then
                buffer_list=( '20000' '2000000' )
            else
                buffer_list=( '20000' )
            fi
            for quality1 in "${quality1_list[@]}" ; do
                for quality2 in "${quality2_list[@]}" ; do
                    qualities="${quality1}-${quality2}"
                    first='1000000'
                    second='1000000'
                    for buffer in "${buffer_list[@]}"; do
                        for copy_buffer in 'none' 'copy' 'mix_all' 'mix_same'; do
                            if [[ $copy_buffer == 'none' || $copy_buffer == 'copy' ]]; then
                                buffer_mix_type_list=( '' )
                            else
                                buffer_mix_type_list=( '_all' '_policy' '_value' )
                            fi
                            for buffer_mix_type in "${buffer_mix_type_list[@]}"; do
                                copy_buffer_str=$copy_buffer
                                if [[ ${copy_buffer} == 'mix_all' ]]; then
                                    copy_buffer_str='mix\_all'
                                fi
                                if [[ ${copy_buffer} == 'mix_same' ]]; then
                                    copy_buffer_str='mix\_same'
                                fi
                                for copy_optim in "" "_copy_optim"; do
                                    #for copy_optim in '' '_copy_optim'; do
                                    #    if [[ $copy_optim == '_copy_optim' ]]; then
                                    #        copy_optim_str='copy\_optim'
                                    #        copy_optim_path='copy_optim'
                                    #    else
                                    #        copy_optim_str='no\_copy\_optim'
                                    #        copy_optim_path='no_copy_optim'
                                    #    fi
                                    TMPFILE1=$(mktemp) || exit 1
                                    if [[ "${algorithms_offline[@]}" =~ ${algorithm1} ]]; then
                                        log_name=logs/online_change_task_${dataset}_${quality1}_${algorithm1}_${first}.latest.log
                                    else
                                        log_name=logs/online_change_task_${dataset}_${algorithm1}_${first}_${buffer}.latest.log
                                    fi
                                    if [[ -f $log_name ]]; then
                                        echo $log_name
                                        awk -f online_change_task_data_processing.awk $log_name | sed "s/=//g" | sed "s/\}//g" > TMPFILE1
                                    else
                                        echo $log_name not exist
                                        exit 1
                                    fi
                                    TMPFILE2=$(mktemp) || exit 1
                                    #log_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}${copy_optim}_${copy_buffer}.latest.log
                                    log_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}${copy_optim}_${copy_buffer}${buffer_mix_type}.latest.log
                                    if [[ -f $log_name ]]; then
                                        echo $log_name
                                        awk -f online_change_task_data_processing.awk $log_name | sed "s/=//g" | sed "s/\}//g" > TMPFILE2
                                    else
                                        :
                                        #echo $log_name not exist
                                    fi
                                    pic_name=pics/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}_${copy_buffer}.png
                                    txt_name=pics/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}_${copy_buffer}.txt
                                    #cat TMPFILE1 TMPFILE2 | gnuplot -e "set terminal png; set title '${dataset}\_${qualities}\_${algorithms}\_${first}\_${second}\_${buffer}\_${copy_optim_str}\_${copy_buffer_str}'; plot '-' using 1 with lines;" > $pic_name
                                    cat TMPFILE1 TMPFILE2 | gnuplot -e "set terminal png; set title '${dataset}\_${qualities}\_${algorithms}\_${first}\_${second}\_${buffer}\_${copy_buffer_str}'; plot '-' using 1 with lines;" > $pic_name
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
