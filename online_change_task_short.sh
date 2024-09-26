#!/usr/bin/bash

expand_str=`date +"%Y%m%d%H%M%S"`
test_arg=""
algorithms=""
copy_optim_arg=""
copy_optim_str=""
dataset=''
qualities=''
first_n_steps=1000000
second_n_steps=1000000
n_buffer=0
seed=0

#ARGS=`getopt -o tc --long test,copy_optim -n 'online_change_task.sh' -- "$@"`
ARGS=`getopt -o +tcn:a:q:d: --long test,copy_optim,algorithms:,qualities:,dataset:,n_buffer: -n "$0" -- "$@"`
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
            if [[ $2 == "is" ]]; then
                algorithms="iql-sac";
            elif [[ $2 == "it" ]]; then
                algorithms="iql-td3";
            elif [[ $2 == "itb" ]]; then
                algorithms="iql-td3_plus_bc";
            elif [[ $2 == "ic" ]]; then
                algorithms="iql-cql";
            elif [[ $2 == "ica" ]]; then
                algorithms="iql-cal";
            elif [[ $2 == "cs" ]]; then
                algorithms="cql-sac";
            elif [[ $2 == "ct" ]]; then
                algorithms="cql-td3";
            elif [[ $2 == "ctb" ]]; then
                algorithms="cql-td3_plus_bc";
            elif [[ $2 == "ci" ]]; then
                algorithms="cql-iql";
            elif [[ $2 == "cca" ]]; then
                algorithms="cql-cal";
            elif [[ $2 == "cas" ]]; then
                algorithms="cal-sac";
            elif [[ $2 == "cat" ]]; then
                algorithms="cal-td3";
            elif [[ $2 == "catb" ]]; then
                algorithms="cal-td3_plus_bc";
            elif [[ $2 == "cac" ]]; then
                algorithms="cal-cql";
            elif [[ $2 == "cai" ]]; then
                algorithms="cal-ial";
            elif [[ $2 == "ts" ]]; then
                algorithms="td3-sac";
            elif [[ $2 == "st" ]]; then
                algorithms="sac-td3";
            elif [[ $2 == "tbs" ]]; then
                algorithms="td3_plus_bc-sac";
            elif [[ $2 == "tbt" ]]; then
                algorithms="td3_plus_bc-td3";
            elif [[ $2 == "tbi" ]]; then
                algorithms="td3_plus_bc-iql";
            elif [[ $2 == "tbc" ]]; then
                algorithms="td3_plus_bc-cql";
            elif [[ $2 == "tbca" ]]; then
                algorithms="td3_plus_bc-cal";
            elif [[ $2 == "to" ]]; then
                algorithms1="td3-td3_plus_bc";
                algorithms2="td3-iql";
                algorithms3="td3-cql";
                algorithms4="td3-cal";
                algorithms="other";
            elif [[ $2 == "so" ]]; then
                algorithms1="sac-td3_plus_bc";
                algorithms2="sac-iql";
                algorithms3="sac-cql";
                algorithms4="sac-cal";
                algorithms="other";
            fi
            shift 2
            ;;
        -q|--qualities)
            if [[ $2 == "e" || $2 == "em" ]]; then
                qualities="expert-medium";
            elif [[ $2 == "m" || $2 == "mm" ]]; then
                qualities="medium-medium";
            elif [[ $2 == "ee" ]]; then
                qualities="expert-expert";
            elif [[ $2 == "me" ]]; then
                qualities="medium-expert";
            fi
            shift 2
            ;;
        -d|--dataset)
            if [[ $2 == "c" ]]; then
                dataset="halfcheetah";
            elif [[ $2 == "h" ]]; then
                dataset="hopper";
            elif [[ $2 == "w" ]]; then
                dataset="walker2d";
            elif [[ $2 == "a" ]]; then
                dataset="ant";
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

if [[ ${algorithms} != "other" ]]; then
        export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin;

        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=none --gpu 0 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=copy --gpu 1 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_all --gpu 2 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_same --gpu 3 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_all --buffer_mix_type=value --gpu 4 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_all --buffer_mix_type=policy --gpu 5 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_same --buffer_mix_type=value --gpu 6 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=mix_same --buffer_mix_type=policy --gpu 7 ${test_arg} &

        wait
else
        export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin;
        if [[ $qualities=='expert-medium' ]]; then
                qualities='medium-expert'
        fi

        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms1} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=none --gpu 0 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms1} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=copy --gpu 1 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms2} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=none --gpu 2 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms2} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=copy --gpu 3 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms3} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=none --gpu 4 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms3} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=copy --gpu 5 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms4} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=none --gpu 6 ${test_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms4} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --copy_buffer=copy --gpu 7 ${test_arg} &

        wait
fi
