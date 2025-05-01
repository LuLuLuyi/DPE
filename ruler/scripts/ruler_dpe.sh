#!/bin/bash

root_dir=data-jsonl-100
CUDA_DEVICE=0

MODEL_NAME="llama3-8b-Instruct"
MODEL_PATH="Meta-Llama-3-8B-Instruct"

USE_TOPK=true
TOPK=32
OUTPUT_DIR="output"
tasks=(
  cwe 
  fwe 
  niah_multikey_1 
  niah_multikey_2 
  niah_multikey_3 
  niah_multiquery 
  niah_multivalue 
  niah_single_1 
  niah_single_2 
  niah_single_3 
  qa_1 
  qa_2 
  vt
)

SEQ_LENGTHS=(131072)

for task in "${tasks[@]}"; do
  task_dir="${root_dir}/${task}"
  
  for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
    file="${task_dir}/${MODEL_NAME}-${SEQ_LEN}.jsonl"
    
    echo "Evaluating task: ${task}, model: ${MODEL_NAME}, seq_len: ${SEQ_LEN}"
    
    TOPK_ARG=""
    if [ "$USE_TOPK" = true ]; then
      TOPK_ARG="--use_topk --topk ${TOPK}"
    fi
    
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
    python run_ruler.py \
      --model_path ${MODEL_PATH} \
      --task ${task} \
      --data_dir ${file} \
      --seq_len ${SEQ_LEN} \
      --output_dir_base ${OUTPUT_DIR} \
      --model_type llama \
      ${TOPK_ARG}
      
    echo "Completed ${task} with seq_len ${SEQ_LEN}"
  done
done

echo "All evaluations complete!"
