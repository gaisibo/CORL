#!/usr/bin/bash

expand_str=`date +"%Y%m%d%H%M%S"`
test_arg=""
algorithms=""
dataset="halfcheetah"
qualities=''
first_n_steps=1000000
n_buffer=0
gpu=0

#ARGS=`getopt -o tc --long test,copy_optim -n 'online_change_task.sh' -- "$@"`
ARGS=`getopt -o +tg --long test,algorithms:,qualities:,dataset:,first_n_steps:,n_buffer:,gpu: -n "$0" -- "$@"`
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
        --n_buffer)
            n_buffer=$2;
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

#must_args="n_buffer"
#for must_arg_str in $must_args; do
#    must_arg=`eval echo '$'$must_arg_str`
#    if [ $must_arg -eq 0 ]; then
#        echo "$must_arg_str must be set!"
#        exit 1
#    fi
#done

if [[ $algorithms == 'iql' || $algorithms == 'cql' || $algorithms == 'cal' || $algorithms=='td3_plus_bc' ]]; then
    output_file_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${first_n_steps}.${expand_str}.log
    must_arg_strs=( "algorithms" "dataset" "qualities" )
else
    output_file_name=logs/online_change_task_${dataset}_${algorithms}_${first_n_steps}_${n_buffer}.${expand_str}.log
    must_arg_strs=( "algorithms" "dataset" "n_buffer" )
fi
for must_arg_str in "algorithms" "dataset" "qualities"; do
must_arg=`eval echo '$'$must_arg_str`
if [[ $must_arg == '' ]]; then
    echo "$must_arg_str must be set!"
    exit 1
fi
done
echo $output_file_name
echo "python online_change_task.py --dataset ${dataset} --algorithms=${algorithms} --qualities=${qualities} --first_n_steps ${first_n_steps} --n_buffer ${n_buffer} ${test_arg}" > ${output_file_name}
echo "" >> ${output_file_name}
python online_change_task_pretrain.py --dataset ${dataset} --algorithms ${algorithms} --qualities ${qualities} --first_n_steps ${first_n_steps} --n_buffer ${n_buffer} --gpu ${gpu} ${test_arg} | tee ${output_file_name}
