#!/usr/bin/bash

expand_str=`date +"%Y%m%d%H%M%S"`
test_arg=""
algorithms=""
copy_optim_arg=""
copy_optim_str=""
explore_arg=""
explore_str=""
copy_buffer=""
buffer_mix_type="all"
dataset=''
qualities=''
first_n_steps=1000000
second_n_steps=1000000
n_buffer=0
seed=0
gpu=0

#ARGS=`getopt -o tc --long test,copy_optim -n 'online_change_task.sh' -- "$@"`
<<<<<<< HEAD
<<<<<<< HEAD
ARGS=`getopt -o +tecb:m:s:g: --long test,copy_optim,explore,copy_buffer:,buffer_mix_type:,algorithms:,qualities:,dataset:,first_n_steps:,second_n_steps:,n_buffer:,seed:,gpu: -n "$0" -- "$@"`
=======
ARGS=`getopt +tcb --long test,copy_optim,copy_buffer:,algorithms:,qualities:,dataset:,first_n_steps:,second_n_steps:,n_buffer: -n "$0" -- "$@"`
>>>>>>> f9f73bb (完成多种类o2o，包括iql、td3和sac三种算法。)
=======
ARGS=`getopt -o +tcbmsg --long test,copy_optim,copy_buffer:,buffer_mix_type:,algorithms:,qualities:,dataset:,first_n_steps:,second_n_steps:,n_buffer:,seed:,gpu: -n "$0" -- "$@"`
>>>>>>> 1e851d1 (完成版。)
if [ $? != 0 ]; then
    echo "Terminating..."
    exit 1
fi
eval set -- "${ARGS}"
while true; do
    case "$1" in
        -t|--test)
            expand_str="test";
            test_arg="--test";
            shift
            ;;
        -c|--copy_optim)
            copy_optim_arg="--copy_optim";
            copy_optim_str="_copy_optim";
            shift
            ;;
        -e|--explore)
            explore_arg="--explore";
            explore_str="_explore";
            shift
            ;;
        -b|--copy_buffer)
            copy_buffer=$2;
            shift 2
            ;;
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 1e851d1 (完成版。)
        -m|--buffer_mix_type)
            buffer_mix_type=$2;
            shift 2
            ;;
<<<<<<< HEAD
=======
>>>>>>> f9f73bb (完成多种类o2o，包括iql、td3和sac三种算法。)
=======
>>>>>>> 1e851d1 (完成版。)
        --algorithms)
            algorithms=$2;
            shift 2
            ;;
        --dataset)
            dataset=$2;
            shift 2
            ;;
        --qualities)
            qualities=$2;
            shift 2
            ;;
        --first_n_steps)
            first_n_steps=$2;
            shift 2
            ;;
        --second_n_steps)
            second_n_steps=$2;
            shift 2
            ;;
        --n_buffer)
            n_buffer=$2;
            shift 2
            ;;
        -s|--seed)
            seed=$2;
            shift 2
            ;;
        -g|--gpu)
            gpu=$2;
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo $1
            echo "Internal error!";
            exit 1
            ;;
    esac
done

must_args="n_buffer"
for must_arg_str in $must_args; do
    must_arg=`eval echo '$'$must_arg_str`
    if [ $must_arg -eq 0 ]; then
        echo "$must_arg_str must be set!"
        exit 1
    fi
done

for must_arg_str in "algorithms" "dataset" "qualities" "copy_buffer"; do
    must_arg=`eval echo '$'$must_arg_str`
    if [[ $must_arg == '' ]]; then
        echo "$must_arg_str must be set!"
        exit 1
    fi
done

if [[ $copy_buffer == "mix_all" || $copy_buffer == "mix_same" ]]; then
    output_file_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${first_n_steps}_${second_n_steps}_${n_buffer}${explore}${copy_optim_str}_${copy_buffer}_${buffer_mix_type}.${expand_str}_${seed}.log
else
    output_file_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${first_n_steps}_${second_n_steps}_${n_buffer}${explore}${copy_optim_str}_${copy_buffer}.${expand_str}_${seed}.log
fi
echo $output_file_name
echo "python online_change_task.py --dataset ${dataset} --algorithms=${algorithms} --qualities=${qualities} $copy_optim_arg $explore_arg $explore_arg --copy_buffer $copy_buffer --buffer_mix_type ${buffer_mix_type} --first_n_steps ${first_n_steps} --second_n_steps ${second_n_steps} --n_buffer ${n_buffer} --seed ${seed} ${test_arg}" > ${output_file_name}
echo "" >> ${output_file_name}
python online_change_task.py ${copy_optim_arg} ${explore_arg} --copy_buffer ${copy_buffer} --buffer_mix_type ${buffer_mix_type} --dataset ${dataset} --algorithms ${algorithms} --qualities ${qualities} --first_n_steps ${first_n_steps} --second_n_steps ${second_n_steps} --n_buffer ${n_buffer} --gpu ${gpu} --seed ${seed} ${test_arg} | tee ${output_file_name}
