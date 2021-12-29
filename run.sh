~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j coyy -o out_siamese_update_replay.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --use_phi_update --use_phi_replay "
~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j coyn -o out_siamese_update_noreplay.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --use_phi_update --no_use_phi_replay "
~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j cony -o out_siamese_noupdate_replay.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --use_phi_replay "
~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j conn -o out_siamese_noupdate_noreplay.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --no_use_phi_replay "


~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j coyyp -o out_siamese_update_replay_pretrain.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --use_phi_update --use_phi_replay --pretrain_phi_epoch 10"
~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j coynp -o out_siamese_update_noreplay_pretrain.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --use_phi_update --no_use_phi_replay --pretrain_phi_epoch 10"
~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j conyp -o out_siamese_noupdate_replay_pretrain.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --use_phi_replay --pretrain_phi_epoch 10"
~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j connp -o out_siamese_noupdate_noreplay_pretrain.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --no_use_phi_replay --pretrain_phi_epoch 10"
