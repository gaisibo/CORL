~/run_gpu_task.beta.1 -w compute3 -g gpu:rtx2080:1 -c 1 -j coyy -o out_siamese_update_replay_eval.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --use_phi_update --use_phi_replay --eval"
~/run_gpu_task.beta.1 -w compute3 -g gpu:rtx2080:1 -c 1 -j coyn -o out_siamese_update_noreplay_eval.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --use_phi_update --no_use_phi_replay --eval"
~/run_gpu_task.beta.1 -w compute2 -g gpu:rtx2080:1 -c 1 -j cony -o out_siamese_noupdate_replay_eval.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --use_phi_replay --eval"
~/run_gpu_task.beta.1 -w compute2 -g gpu:rtx2080:1 -c 1 -j conn -o out_siamese_noupdate_noreplay_eval.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --no_use_phi_replay --eval"


~/run_gpu_task.beta.1 -w compute3 -g gpu:rtx2080:1 -c 1 -j coyyp -o out_siamese_update_replay_pretrain_eval.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --use_phi_update --use_phi_replay --pretrain_phi_epoch 10 --eval"
~/run_gpu_task.beta.1 -w compute3 -g gpu:rtx2080:1 -c 1 -j coynp -o out_siamese_update_noreplay_pretrain_eval.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --use_phi_update --no_use_phi_replay --pretrain_phi_epoch 10 --eval"
~/run_gpu_task.beta.1 -w compute2 -g gpu:rtx2080:1 -c 1 -j conyp -o out_siamese_noupdate_replay_pretrain_eval.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --use_phi_replay --pretrain_phi_epoch 10 --eval"
~/run_gpu_task.beta.1 -w compute2 -g gpu:rtx2080:1 -c 1 -j connp -o out_siamese_noupdate_noreplay_pretrain_eval.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --no_use_phi_replay --pretrain_phi_epoch 10 --eval"