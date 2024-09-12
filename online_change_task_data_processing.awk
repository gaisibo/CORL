$6 ~ /epoch=/{
    for (i=1; i<=NF; i++){
        if ($i ~ /rollout/ || $i ~ /evaluation/){
            print $(i+1)
        }
    }
}
