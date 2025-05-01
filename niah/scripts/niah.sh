#!/bin/bash

DATA_DIR="generated_data"
OUTPUT_DIR="output"
LOG_DIR="logs"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

MODEL_PATH="/path/to/model"
DATA_FILE="${DATA_DIR}/niah_test_data.json"
SUBSET=""
TOPK=48
MAX_TOKENS=128
USE_TOPK="--use_topk"  # Add this flag to enable topk


CUDA_VISIBLE_DEVICES=0 \
python run_niah.py \
    --model_path ${MODEL_PATH} \
    --data_file ${DATA_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --subset ${SUBSET} \
    ${USE_TOPK} \
    --topk ${TOPK} \
    --max_tokens ${MAX_TOKENS}
