#!/bin/bash

MODEL_PATH="Meta-Llama-3-8B-Instruct"
SAVE_DIR="output"
TEST_MAX_LENGTH=131072
PRETRAIN_LENGTH=8192
SELECTED_DIM_PATH="/path/to/weights.pt"
DATA_FILE="${DATA_DIR}/niah_test_data.json"
SUBSET="suffix"
WINDOW_SIZE=1024
TOPK=48
MAX_TOKENS=128
USE_TOPK="--use_topk"  # Add this flag to enable topk


for detect_num in 1 2 3 4 5 6 7 8;
do
    # Here, we define the scale of other dimensions as 'factor'. We only vary the scale of 'detect_num' for detection.
    factor=$((TEST_MAX_LENGTH / PRETRAIN_LENGTH * 2))
    # detecting_factor=(1 2 4 8 16 32 64 128)
    # Generate integer scale factors from TEST_MAX_LENGTH down to 1024
    detecting_factor=()
    length=$TEST_MAX_LENGTH
    
    while [ $length -ge 1024 ]; do
        ratio=$((TEST_MAX_LENGTH / length))
        detecting_factor+=( "$ratio" )
        length=$((length / 2))
    done
    
    num_parts=8
    GROUP_SIZES=()
    for val in "${detecting_factor[@]}"; do
        current_group=""
        separator=""
        for ((i=1; i<=num_parts; i++)); do
            part_val=""
            if [ $i -eq $detect_num ]; then
                part_val="$val"
            else
                part_val="$factor"
            fi
            current_group="${current_group}${separator}${part_val}"
            separator="-"
        done
        GROUP_SIZES+=( "$current_group" )
    done

    # Conduct the NIAH detection experiment here.
    for GROUP_SIZE in "${GROUP_SIZES[@]}"; do
        
        CUDA_VISIBLE_DEVICES=0 \
        python inference.py \
            --model_path ${MODEL_PATH} \
            --data_file ${DATA_FILE} \
            --output_dir ${OUTPUT_DIR} \
            --subset ${SUBSET}_${GROUP_SIZE} \
            --window_size ${WINDOW_SIZE} \
            --group_sizes ${GROUP_SIZE} \
            ${USE_TOPK} \
            --topk ${TOPK} \
            --max_tokens ${MAX_TOKENS} \
            --selected_dim_path ${SELECTED_DIM_PATH}
    done
done
