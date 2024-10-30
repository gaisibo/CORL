#!/usr/bin/sh

#sudo chown -R gaisibo:gaisibo logs
rm logs_save/*
eza --long --git logs | awk '$8 ~ /online/ && $8 !~ /latest/ && $8 !~ /test/ {print $8}' - | awk -F "." '{print $0" "$1" "$2}' | awk '{if($3>x[$2]){x[$2]=$3; y[$2]=$1}} END {for(i in x) {system("ln -sf \"/gaisibo/algo/Continual-Offline/Continual-Offline/logs/"y[i]"\" \"/gaisibo/algo/Continual-Offline/Continual-Offline/logs/"i".latest.log\""); system("cp \"logs/"y[i]"\" \"logs_save\"")}}'

