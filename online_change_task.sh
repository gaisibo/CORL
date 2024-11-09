#!/usr/bin/bash

expand_str=`date +"%Y%m%d%H%M%S"`
test_arg=""
algorithms=""
continual_type="diff"
actor_replay_type="none"
critic_replay_type="none"
copy_optim_arg=""
copy_optim_str=""
explore_arg=""
explore_str=""
buffer_replay_type="all"
dataset=''
qualities=''
first_n_steps=1000000
second_n_steps=1000000
n_buffer=0
n_critics=2
seed=0
gpu=0

#ARGS=`getopt -o tc --long test,copy_optim -n 'online_change_task.sh' -- "$@"`
ARGS=`getopt -o +tecb:m:s:g: --long test,copy_optim,explore,continual_type:,critic_replay_type:,actor_replay_type:,buffer_replay_type:,algorithms:,qualities:,dataset:,first_n_steps:,second_n_steps:,n_buffer:,n_critics:,seed:,gpu: -n "$0" -- "$@"`
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
        -b|--continual_type)
            continual_type=$2;
            shift 2
            ;;
        --critic_replay_type)
            critic_replay_type=$2;
            shift 2
            ;;
        --actor_replay_type)
            actor_replay_type=$2;
            shift 2
            ;;
        -m|--buffer_replay_type)
            buffer_replay_type=$2;
            shift 2
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
        --second_n_steps)
            second_n_steps=$2;
            shift 2
            ;;
        --n_buffer)
            n_buffer=$2;
            shift 2
            ;;
        --n_critics)
            n_critics=$2;
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

for must_arg_str in "algorithms" "dataset" "qualities"; do
    must_arg=`eval echo '$'$must_arg_str`
    if [[ $must_arg == '' ]]; then
        echo "$must_arg_str must be set!"
        exit 1
    fi
done

if [[ $continual_type == "ewc" || $continual_type == "mix" ]]; then
    output_file_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${n_critics}_${first_n_steps}_${second_n_steps}_${n_buffer}${explore}${copy_optim_str}_${continual_type}_${buffer_replay_type}.${expand_str}_${seed}.log
elif [[ $continual_type == "diff" ]]; then
    output_file_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${n_critics}_${first_n_steps}_${second_n_steps}_${n_buffer}${explore}${copy_optim_str}_${critic_replay_type}_${actor_replay_type}_${buffer_replay_type}.${expand_str}_${seed}.log
else
    output_file_name=logs/online_change_task_${dataset}_${qualities}_${algorithms}_${n_critics}_${first_n_steps}_${second_n_steps}_${n_buffer}${explore}${copy_optim_str}_${continual_type}.${expand_str}_${seed}.log
fi
echo $output_file_name
echo "python online_change_task.py --dataset ${dataset} --algorithms=${algorithms} --qualities=${qualities} --n_critics=${n_critics} $copy_optim_arg $explore_arg $explore_arg --continual_type $continual_type --critic_replay_type ${critic_replay_type} --actor_replay_type ${actor_replay_type} --buffer_replay_type ${buffer_replay_type} --first_n_steps ${first_n_steps} --second_n_steps ${second_n_steps} --n_buffer ${n_buffer} --seed ${seed} ${test_arg}" > ${output_file_name}
echo "" >> ${output_file_name}
python online_change_task.py ${copy_optim_arg} ${explore_arg} --continual_type ${continual_type} --critic_replay_type ${critic_replay_type} --actor_replay_type ${actor_replay_type} --buffer_replay_type ${buffer_replay_type} --dataset ${dataset} --algorithms ${algorithms} --qualities ${qualities} --first_n_steps ${first_n_steps} --second_n_steps ${second_n_steps} --n_buffer ${n_buffer} --n_critics ${n_critics} --gpu ${gpu} --seed ${seed} ${test_arg} | tee ${output_file_name}
