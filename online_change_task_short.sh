#!/usr/bin/bash

expand_str=`date +"%Y%m%d%H%M%S"`
test_arg=""
algorithms=""
copy_optim_arg=""
copy_optim_str=""
dataset="halfcheetah"
qualities=''
first_n_steps=1000000
second_n_steps=1000000
n_buffer=0
seed=0

#ARGS=`getopt -o tc --long test,copy_optim -n 'online_change_task.sh' -- "$@"`
ARGS=`getopt -o +tcn:a:q: --long test,copy_optim,algorithms:,qualities:,dataset:,n_buffer: -n "$0" -- "$@"`
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
        -a|--algorithms)
            if [[ $2 == "i" ]]; then
                algorithms="iql-sac";
            elif [[ $2 == "c" ]]; then
                algorithms="cql-sac";
            elif [[ $2 == "s" ]]; then
                algorithms="sac-td3";
            elif [[ $2 == "t" ]]; then
                algorithms="td3-sac";
            fi
            shift 2
            ;;
        -q|--qualities)
            if [[ $2 == "e" ]]; then
                qualities="expert-medium";
            elif [[ $2 == "m" ]]; then
                qualities="medium-medium";
            fi
            shift 2
            ;;
        -n|--n_buffer)
            if [[ $2 == "s" ]]; then
                n_buffer="20000";
            elif [[ $2 == "b" ]]; then
                n_buffer="2000000";
            fi
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

export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin;

bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=none --gpu 0 ${test_arg} &
#bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=copy --gpu 1 ${test_arg} &
#bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_all --gpu 2 ${test_arg} &
#bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_same --gpu 3 ${test_arg} &
#bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_all --buffer_mix_type=value --gpu 4 ${test_arg} &
#bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_all --buffer_mix_type=policy --gpu 5 ${test_arg} &
#bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_same --buffer_mix_type=value --gpu 6 ${test_arg} &
#bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_same --buffer_mix_type=policy --gpu 7 ${test_arg} &

wait
