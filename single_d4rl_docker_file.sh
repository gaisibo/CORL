algo="iqln"
n_ensemble="10"
n_steps="5000"
expectile="0.7"
expectile_min="0.7"
expectile_max="0.7"
update_ratio="2"
alpha="2"
dataset="halfcheetah"
dataset_nums="3000-0"
mix_type="random"
clone=""
max_save_num="10"
actor_replay_type="orl"
actor_replay_lambda="1"
critic_replay_type="orl"
critic_replay_lambda="1"
entropy_time="0.2"
test_=""
device=2

echo original parameters=[$@]

ARGS=`getopt -o a:s:d:m: --long algo:,n_ensemble:,n_steps:,expectile:,expectile_min:,expectile_max:,update_ratio:,alpha:,dataset:,dataset_nums:,mix_type:,clone,max_save_num:,actor_replay_type:,actor_replay_lambda:,critic_replay_type:,critic_replay_lambda:,entropy_time:,device:,test_ -n "$0" -- "$@"`
if [ $? != 0 ]; then
    echo "Terminating..."
    exit 1
fi

echo ARGS=[$ARGS]
#将规范化后的命令行参数分配至位置参数（$1,$2,...)
eval set -- "${ARGS}"
echo formatted parameters=[$@]

while true
do
    case "$1" in
        -a|--algo) 
	    algo=$2
	    shift 2  
	    ;;
        --n_ensemble) 
	    n_ensemble=$2
	    shift 2  
	    ;;
        -s|--n_steps)
	    n_steps=$2
	    shift 2  
	    ;;
        --expectile)
	    expectile=$2
	    shift 2  
	    ;;
        --expectile_min)
	    expectile_min=$2
	    shift 2  
	    ;;
        --expectile_max)
	    expectile_max=$2
	    shift 2  
	    ;;
        --update_ratio)
	    update_ratio=$2
	    shift 2  
	    ;;
        --alpha)
	    alpha=$2
	    shift 2  
	    ;;
        --dataset)
	    dataset=$2
	    shift 2  
	    ;;
        --dataset_nums)
	    dataset_nums=$2
	    shift 2  
	    ;;
        --mix_type)
	    mix_type=$2
	    shift 2  
	    ;;
        --max_save_num)
	    max_save_num=$2
	    shift 2  
	    ;;
        --actor_replay_type)
	    actor_replay_type=$2
	    shift 2  
	    ;;
        --actor_replay_lambda)
	    actor_replay_lambda=$2
	    shift 2  
	    ;;
        --critic_replay_type)
	    critic_replay_type=$2
	    shift 2  
	    ;;
        --critic_replay_lambda)
	    critic_replay_lambda=$2
	    shift 2  
	    ;;
        --entropy_time)
	    entropy_time=$2
	    shift 2  
	    ;;
        --device)
	    device=$2
	    shift 2  
	    ;;
        --clone)
        clone="--clone"
        shift;
        ;;
        --test_)
        test_="--test"
        shift;
        ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

if [[ $algo == "sql" || $algo == "sqln" ]]
then
        if [[ $test_ == "--test" ]]
        then
                task_id=${algo}_${n_ensemble}_${alpha}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_0_test
                output_file=single_output_files/output_${algo}_${n_ensemble}_${alpha}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_test.txt
        else
                task_id=${algo}_${n_ensemble}_${alpha}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_0
                output_file=single_output_files/output_${algo}_${n_ensemble}_${alpha}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_0_$(date +%Y%m%d%H%M%S).txt
        fi
elif [[ $algo == "iqln2" ]]
then
        if [[ $test_ == "--test" ]]
        then
                task_id=${algo}_${n_ensemble}_${alpha}_${update_ratio}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_0_test
                output_file=single_output_files/output_${algo}_${n_ensemble}_${alpha}_${update_ratio}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_test.txt
        else
                task_id=${algo}_${n_ensemble}_${alpha}_${update_ratio}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_0
                output_file=single_output_files/output_${algo}_${n_ensemble}_${alpha}_${update_ratio}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_0_$(date +%Y%m%d%H%M%S).txt
        fi
else
        if [[ $test_ == "--test" ]]
        then
                task_id=${algo}_${n_ensemble}_${expectile}_${expectile_min}_${expectile_max}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_0_test
                output_file=single_output_files/output_${algo}_${n_ensemble}_${expectile}_${expectile_min}_${expectile_max}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_test.txt
        else
                task_id=${algo}_${n_ensemble}_${expectile}_${expectile_min}_${expectile_max}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_0
                output_file=single_output_files/output_${algo}_${n_ensemble}_${expectile}_${expectile_min}_${expectile_max}_${dataset}_${dataset_nums}_${n_steps}_${actor_replay_type}_${actor_replay_lambda}_${critic_replay_type}_${critic_replay_lambda}_${entropy_time}_${clone}_${mix_type}_${max_save_num}_0_$(date +%Y%m%d%H%M%S).txt
        fi
fi
~/run_gpu_task.beta.1 -w compute${device} -g gpu:1 -c 1 -j $task_id -o $output_file -i compute1:5000/co:11 -v /home/gaisibo/Continual-Offline/CCQL:/work "cd /work; /root/miniconda3/bin/python3.7 continual_single.py --algo ${algo} --alpha ${alpha} --update_ratio ${update_ratio} --n_ensemble ${n_ensemble} --experience_type random_episode --dataset ${dataset} --dataset_nums ${dataset_nums} --max_save_num ${max_save_num} --critic_replay_type ${critic_replay_type} --critic_replay_lambda ${critic_replay_lambda} --actor_replay_type ${actor_replay_type} --actor_replay_lambda ${actor_replay_lambda} ${clone} --mix_type ${mix_type} --expectile ${expectile} --expectile_min ${expectile_min} --expectile_max ${expectile_max} --entropy_time ${entropy_time} --n_steps=${n_steps} --seed 0 --read_policy -1"
