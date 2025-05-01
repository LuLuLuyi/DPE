#!/bin/bash

DATA_DIR="niah/generated_data"
LOG_DIR="niah/logs"

mkdir -p ${DATA_DIR}
mkdir -p ${LOG_DIR}

MODEL_PATH="/path/to/model"
NUM_TEST=100
MAX_CONTEXT_LENGTH=131072
OUTPUT_FILE="niah_test_data.json"

python data_generator.py \
    --model_path ${MODEL_PATH} \
    --num_test ${NUM_TEST} \
    --output_dir ${DATA_DIR} \
    --output_file ${OUTPUT_FILE} \
    --max_context_length ${MAX_CONTEXT_LENGTH} \
    > ${LOG_DIR}/data_generation.log 2>&1

echo "Data generation completed. The results are saved in ${DATA_DIR}/${OUTPUT_FILE}"
echo "Logs are saved in ${LOG_DIR}/data_generation.log"