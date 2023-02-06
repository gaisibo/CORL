declare -a dataset=("ant_maze" "maze")
declare -a short_dataset=("am" "ma")
declare -a epochs=("1000" "1000")
declare -a lr=("1e-4" "1e-5")
for i in "${!dataset[@]}"
do
        ~/run_gpu_task -g gpu:v100:1 -c 1 -j "dy${short_dataset[$i]}" -o dynamics_"${dataset[$i]}".txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 pretrain_dynamics.py --dataset ${dataset[$i]} --n_epochs=${epochs[$i]} --learning_rate=${lr[$i]}"
done
