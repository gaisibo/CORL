#!/usr/bin/bash
\rm pics_short/*
dataset="halfcheetah"
qualities="medium-medium"
first=1000000
second=1000000
buffer=20000
copy_optim="_copy_optim"
copy_buffer="mix_same"
buffer_mix_type="all"

algorithms_list=( "iql-sac" "cql-sac" "cal-sac" "td3_plus_bc-sac" )

for algorithms in "${algorithms_list[@]}"; do
    pic_name=/mnt/d/work/continual_offline/pics/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}${copy_optim}_${copy_buffer}_${buffer_mix_type}.png
    if [[ -f $pic_name ]]; then
        pic_short_name=/mnt/d/work/continual_offline/pics_short/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}${copy_optim}_${copy_buffer}_${buffer_mix_type}.png
        ln -s ${pic_name} ${pic_short_name}
    else
        echo ${pic_name} do not exist.
    fi
done

algorithms="td3-sac"
qualities="medium-medium"
pic_name=/mnt/d/work/continual_offline/pics/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}${copy_optim}_${copy_buffer}_${buffer_mix_type}.png
if [[ -f $pic_name ]]; then
    pic_short_name=/mnt/d/work/continual_offline/pics_short/online_change_task_${dataset}_${qualities}_${algorithms}_${first}_${second}_${buffer}${copy_optim}_${copy_buffer}_${buffer_mix_type}.png
    ln -s ${pic_name} ${pic_short_name}
else
    echo ${pic_name} do not exist.
fi
