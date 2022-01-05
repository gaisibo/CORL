~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j coyy -o out_siamese_update_replay_orl.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --use_phi_update --use_phi_replay --orl"
~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j cony -o out_siamese_noupdate_replay_orl.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --use_phi_replay --orl"
~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j conn -o out_siamese_noupdate_noreplay_orl.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --no_use_phi_replay --orl"



~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j coyy -o out_siamese_update_replay_noorl.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --use_phi_update --use_phi_replay --no_orl"
~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j cony -o out_siamese_noupdate_replay_noorl.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --use_phi_replay --no_orl"
~/run_gpu_task -g gpu:rtx2080:1 -c 1 -j conn -o out_siamese_noupdate_noreplay_noorl.txt -i "compute1:5000/comrl:v11" -v "/home/gaisibo/Continual-Offline/CCQL:/app" "cd /app; /root/miniconda3/bin/python3.7 ccql_siamese.py --no_use_phi_update --no_use_phi_replay --no_orl"
