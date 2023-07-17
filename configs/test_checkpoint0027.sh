#!/usr/bin/env bash

set -x

EXP_DIR=./run/test_checkpoint0027
PY_ARGS=${@:1}

python3 -u inference.py \
    --output_dir ${EXP_DIR} \
    --data_mode '15frames' \
    --num_global_frames 3 \
    --num_support_frames 2 \
    --num_frames 6 \
    --num_support_frames_testing 2 \
    --enc_temporal_window 5 \
    --num_feature_levels 4 \
    --batch_size 1 \
    --lr 5e-5 \
    --dist_url tcp://127.0.0.1:50001 \
    --enc_connect_all_frames \
    --shuffled_aug "centerCrop" \
    --num_workers 12 \
    --data_coco_lite_path '/home/qinc/Dataset/BUV' \
    --with_box_refine \
    --resume '/home/qinc/Downloads/checkpoint0027.pth' \
    --eval \
    ${PY_ARGS}