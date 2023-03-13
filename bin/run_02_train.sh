# Models stage 1 - run two seeds for each model. If you want to run on folds, uncomment the folds below
for FOLD in -1 -1 # 0 1 2 3 4
do 
    for CONFIG in cfg_dh_4c_aux14B_v2_s cfg_dh_4c_aux14B_b3 cfg_dh_4c_aux14B_b4 cfg_dh_4c_aux14B_b5 # cfg_ip_4c_aux14B_v2_m 
    do
	echo "Running stage 1 model for config "$CONFIG
        python train.py -C $CONFIG --fold $FOLD
    done
done

# Models stage 2 - use the weights from stage 1 model
FOLD=-1
for CONFIG in cfg_dh_4c_aux14B_v2_s cfg_dh_4c_aux14B_b3 cfg_dh_4c_aux14B_b4 cfg_dh_4c_aux14B_b5 # cfg_ip_4c_aux14B_v2_m 
do
    search_dir="weights/${CONFIG}/fold" 
    search_dir+=$FOLD
    for WEIGHTS_NAME in "$search_dir"/check*
    do
        echo "Loading weights file "$WEIGHTS_NAME
        python train.py -C "${CONFIG}_agg1" --fold $FOLD --pretrained_weights $WEIGHTS_NAME
    done
done


