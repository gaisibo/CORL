declare -a gene=("no_mb_generate")
declare -a short_gene=("no_ge")
declare -a replay=("mb_replay")
declare -a short_replay=("re")
declare -a orl=("orl")
declare -a short_orl=("or")
declare -a dataset=("ant_maze")
declare -a short_dataset=("am")
for i in "${!gene[@]}"
do
        for j in "${!replay[@]}"
        do
                for k in "${!orl[@]}"
                do
                        for l in "${!dataset[@]}"
                        do
                                ~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j "${short_gene[$i]}${short_replay[$j]}${short_orl[$k]}${short_dataset[$l]}" -o output_${gene[$i]}_${replay[$j]}_${orl[$k]}_${dataset[$l]}.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --${gene[$i]} --${replay[$j]} --${orl[$k]} --n_epochs 200 --dataset ${dataset[$l]}"
                        done
                done
        done
done