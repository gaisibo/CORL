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

#ARGS=`getopt -o +tcn:a:q:d: --long test,copy_optim,algorithms:,qualities:,dataset:,n_buffer: -n "$0" -- "$@"`
ARGS=`getopt -o +tcea:m:q:d:l: --long test,explore,copy_optim,algorithms:,qualities:,dataset: -n "$0" -- "$@"`
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
        -a|--algorithms)
            if [[ $2 == "is" ]]; then
                algorithms="iql-sac";
            elif [[ $2 == "it" ]]; then
                algorithms="iql-td3";
            elif [[ $2 == "ii" ]]; then
                algorithms="iql-iql_online";
            elif [[ $2 == "io" ]]; then
                algorithms1="iql-td3_plus_bc";
                algorithms2="iql-iql";
                algorithms3="iql-cql";
                algorithms4="iql-cal";
                algorithms="other";
            elif [[ $2 == "ins" ]]; then
                algorithms="iqln-sac";
            elif [[ $2 == "int" ]]; then
                algorithms="iqln-td3";
            elif [[ $2 == "ini" ]]; then
                algorithms="iqln-iqln_online";
            elif [[ $2 == "inie" ]]; then
                algorithms="iqln-iqlne_online";
            elif [[ $2 == "iie" ]]; then
                algorithms="iql-iqle_online";
            elif [[ $2 == "cs" ]]; then
                algorithms="cql-sac";
            elif [[ $2 == "ct" ]]; then
                algorithms="cql-td3";
            elif [[ $2 == "ci" ]]; then
                algorithms="cql-iql_online";
            elif [[ $2 == "co" ]]; then
                algorithms1="cql-td3_plus_bc";
                algorithms2="cql-iql";
                algorithms3="cql-cql";
                algorithms4="cql-cal";
                algorithms="other";
            elif [[ $2 == "cat" ]]; then
                algorithms="cal-td3";
            elif [[ $2 == "cas" ]]; then
                algorithms="cal-sac";
            elif [[ $2 == "cai" ]]; then
                algorithms="cal-iql_online";
            elif [[ $2 == "cao" ]]; then
                algorithms1="cal-td3_plus_bc";
                algorithms2="cal-iql";
                algorithms3="cal-cql";
                algorithms4="cal-cal";
                algorithms="other";
            elif [[ $2 == "ts" ]]; then
                algorithms="td3-sac";
            elif [[ $2 == "ti" ]]; then
                algorithms="td3-iql_online";
            elif [[ $2 == "st" ]]; then
                algorithms="sac-td3";
            elif [[ $2 == "si" ]]; then
                algorithms="sac-iql_online";
            elif [[ $2 == "tbs" ]]; then
                algorithms="td3_plus_bc-sac";
            elif [[ $2 == "tbt" ]]; then
                algorithms="td3_plus_bc-td3";
            elif [[ $2 == "tbi" ]]; then
                algorithms="td3_plus_bc-iql_online";
            elif [[ $2 == "tbo" ]]; then
                algorithms1="td3_plus_bc-td3_plus_bc";
                algorithms2="td3_plus_bc-iql";
                algorithms3="td3_plus_bc-cql";
                algorithms4="td3_plus_bc-cal";
                algorithms="other";
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
            elif [[ $2 == "r" || $2 == "rr" ]]; then
                qualities="random-medium";
            elif [[ $2 == "l" || $2 == "ll" ]]; then
                qualities="medium_replay-medium";
            elif [[ $2 == "h" || $2 == "hh" ]]; then
                qualities="medium_expert-medium";
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
        #-c|--copy_optim)
        #    copy_optim_arg="--copy_optim"
        #    copy_optim_str="_copy_optim"
        #    shift
        #    ;;
        -e|--explore)
            explore_arg="--explore"
            explore_str="_explore"
            shift
            ;;
        #-n|--n_buffer)
        #    if [[ $2 == "s" ]]; then
        #        n_buffer="20000";
        #    elif [[ $2 == "b" ]]; then
        #        n_buffer="2000000";
        #    fi
        #    shift 2
        #    ;;
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

if [[ ${algorithms} != "other" && ${algorithms:0:4} != "iqln" ]]; then
        export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin;

        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=20000 --continual_type=none --gpu 0 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=20000 --continual_type=copy --gpu 1 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=20000 --continual_type=mix --buffer_replay_type=all --gpu 2 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=20000 --continual_type=mix --buffer_replay_type=same --gpu 3 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=2000000 --continual_type=none --gpu 4 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=2000000 --continual_type=copy --gpu 5 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=2000000 --continual_type=mix --buffer_replay_type=all --gpu 6 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=2000000 --continual_type=mix --buffer_replay_type=same --gpu 7 ${test_arg} ${explore_arg} &

        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --n_critics=10 --qualities=${qualities} --copy_optim --n_buffer=20000 --continual_type=none --gpu 0 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --n_critics=10 --qualities=${qualities} --copy_optim --n_buffer=20000 --continual_type=copy --gpu 1 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --n_critics=10 --qualities=${qualities} --copy_optim --n_buffer=20000 --continual_type=mix --buffer_replay_type=all --gpu 2 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --n_critics=10 --qualities=${qualities} --copy_optim --n_buffer=20000 --continual_type=mix --buffer_replay_type=same --gpu 3 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --n_critics=10 --qualities=${qualities} --copy_optim --n_buffer=2000000 --continual_type=none --gpu 4 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --n_critics=10 --qualities=${qualities} --copy_optim --n_buffer=2000000 --continual_type=copy --gpu 5 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --n_critics=10 --qualities=${qualities} --copy_optim --n_buffer=2000000 --continual_type=mix --buffer_replay_type=all --gpu 6 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --n_critics=10 --qualities=${qualities} --copy_optim --n_buffer=2000000 --continual_type=mix --buffer_replay_type=same --gpu 7 ${test_arg} ${explore_arg} &

        #bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} ${explore} --n_buffer=${n_buffer} --continual_type=mix --buffer_replay_type=all --buffer_mix_type=value --gpu 4 ${test_arg} &
        #bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} ${explore} --n_buffer=${n_buffer} --continual_type=mix --buffer_replay_type=all --buffer_mix_type=policy --gpu 5 ${test_arg} &
        #bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} ${explore} --n_buffer=${n_buffer} --continual_type=mix --buffer_replay_type=same --buffer_mix_type=value --gpu 6 ${test_arg} &
        #bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} ${copy_optim_arg} ${explore} --n_buffer=${n_buffer} --continual_type=mix --buffer_replay_type=same --buffer_mix_type=policy --gpu 7 ${test_arg} &

        wait
elif [[ ${algorithms} != "other" ]]; then
        export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin;

        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=20000 --continual_type=none --gpu 0 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=20000 --continual_type=copy --gpu 1 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=20000 --continual_type=mix --buffer_replay_type=all --gpu 2 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=20000 --continual_type=mix --buffer_replay_type=same --gpu 3 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=2000000 --continual_type=none --gpu 4 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=2000000 --continual_type=copy --gpu 5 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=2000000 --continual_type=mix --buffer_replay_type=all --gpu 6 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=2 --copy_optim --n_buffer=2000000 --continual_type=mix --buffer_replay_type=same --gpu 7 ${test_arg} ${explore_arg} &

        wait
else
        export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin;
        if [[ $qualities=='expert-medium' ]]; then
                qualities='medium-expert'
        fi

        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms1} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --continual_type=none --gpu 0 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms1} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --continual_type=copy --gpu 1 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms2} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --continual_type=none --gpu 2 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms2} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --continual_type=copy --gpu 3 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms3} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --continual_type=none --gpu 4 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms3} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --continual_type=copy --gpu 5 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms4} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --continual_type=none --gpu 6 ${test_arg} ${explore_arg} &
        bash online_change_task.sh --dataset=${dataset} --algorithms=${algorithms4} --qualities=${qualities} ${copy_optim_arg} --n_buffer=${n_buffer} --continual_type=copy --gpu 7 ${test_arg} ${explore_arg} &

        wait
fi
