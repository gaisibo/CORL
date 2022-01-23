declare -a gene=("siamese" "random")
declare -a short_gene=("s" "r")
declare -a replay=("siamese" "random")
declare -a short_replay=("s" "r")
declare -a orl=("orl" "no_orl")
declare -a short_orl=("o" "n")
declare -a dataset=("maze")
declare -a short_dataset=("m")
declare -a reduce=('retrain' "no_retrain")
declare -a short_reduce=('r' 'n')
for i in "${!gene[@]}"
do
        for j in "${!replay[@]}"
        do
                for k in "${!orl[@]}"
                do
                        for l in "${!dataset[@]}"
                        do
                                for m in "${!reduce[@]}"
                                do
                                        ~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j "${short_gene[$i]}${short_replay[$j]}${short_orl[$k]}${short_dataset[$l]}${short_reduce[$m]}" -o output_${gene[$i]}_${replay[$j]}_${orl[$k]}_${dataset[$l]}_${reduce[$m]}.txt -i "compute1:5000/comrl:v13" -v "/home/gaisibo/Continual-Offline/CCQL:/work" "cd /work; /root/miniconda3/bin/python3.7 ccql_siamese.py --generate_type ${gene[$i]} --replay_type ${replay[$j]} --orl ${orl[$k]} --reduce_replay ${reduce[$m]} --n_epochs 1000 --dataset ${dataset[$l]}"
                                done
                        done
                done
        done
done