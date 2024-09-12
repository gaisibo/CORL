#!/usr/bin/bash
#awk -f online_change_task_data_processing.awk online_change_task.$1.log | sed "s/=//g" | sed "s/\}//g" | sed "s/epoch//g" | gnuplot -e "plot '-' w lp; pause 99"
rm logs/*latest*
rm pics/*
bash read_logs.sh
for algorithm1 in 'iql' 'td3'; do
    for algorithm2 in 'sac'; do
        if [[ $algorithm1 == $algorithm2 ]]; then
            continue;
        fi
        algorithms="${algorithm1}-${algorithm2}"
        for dataset in 'halfcheetah'; do
            for quality1 in 'medium' 'expert'; do
                if [[ ($algorithm1 == 'sac' || $algorithm1 == 'td3') && $quality1 == 'expert' ]]; then
                    continue;
                fi
                for quality2 in 'medium' 'expert'; do
                    if [[ ($algorithm2 == 'sac' || $algorithm2 == 'td3') && $quality2 == 'expert' ]]; then
                        continue;
                    fi
                    qualities="${quality1}-${quality2}"
                    first='1000000'
                    second='1000000'
                    for buffer in '20000' '2000000'; do
                        for copy_buffer in 'none' 'copy' 'mix_all' 'mix_same'; do
                            copy_buffer_str=$copy_buffer
                            if [[ ${copy_buffer} == 'mix_all' ]]; then
                                copy_buffer_str='mix\_all'
                            fi
                            if [[ ${copy_buffer} == 'mix_same' ]]; then
                                copy_buffer_str='mix\_same'
                            fi
                            #for copy_optim in '' '_copy_optim'; do
                            #    if [[ $copy_optim == '_copy_optim' ]]; then
                            #        copy_optim_str='copy\_optim'
                            #        copy_optim_path='copy_optim'
                            #    else
                            #        copy_optim_str='no\_copy\_optim'
                            #        copy_optim_path='no_copy_optim'
                            #    fi
                            TMPFILE1=$(mktemp) || exit 1
                            if [[ $algorithm1 == 'iql' ]]; then
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
                            log_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}_${copy_buffer}.latest.log
                            if [[ -f $log_name ]]; then
                                echo $log_name
                                awk -f online_change_task_data_processing.awk $log_name | sed "s/=//g" | sed "s/\}//g" > TMPFILE2
                            else
                                echo $log_name not exist
                                exit 1
                            fi
                            pic_name=pics/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}_${copy_buffer}.png
                            txt_name=pics/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}_${copy_buffer}.txt
                            #cat TMPFILE1 TMPFILE2 | gnuplot -e "set terminal png; set title '${dataset}\_${qualities}\_${algorithms}\_${first}\_${second}\_${buffer}\_${copy_optim_str}\_${copy_buffer_str}'; plot '-' using 1 with lines;" > $pic_name
                            cat TMPFILE1 TMPFILE2 | gnuplot -e "set terminal png; set title '${dataset}\_${qualities}\_${algorithms}\_${first}\_${second}\_${buffer}\_${copy_buffer_str}'; plot '-' using 1 with lines;" > $pic_name
                            #done
                        done
                    done
                done
            done
        done
    done
done
# | sed "s/,//g"
