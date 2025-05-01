import os
import argparse
from concurrent.futures import ProcessPoolExecutor

tasks = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"
]

template = {
    "llama3.1": "llama3-chat",
    "llama2": "meta-chat",
    "qwen2": "qwen2"
}

def execute_command(task):
    cmd = f"""
    python data/prepare.py \
    --save_dir ./data-jsonl \
    --benchmark synthetic \
    --task {task} \
    --tokenizer_path {args.model_path} \
    --tokenizer_type hf \
    --max_seq_length {args.max_length} \
    --model_template_type {temp} \
    --num_samples 500
    """
    print(f"Executing: {cmd}")
    result = os.system(cmd)
    if result != 0:
        print(f"Command for task {task} failed with exit code {result}")


# pip install pyyaml, tqdm, wonderwords, tenacity, nltk, filelock
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_path", type=str, required=True)
    parse.add_argument("--max_length", type=int, default=None)
    parse.add_argument("--temp", default="llama3.1", choices=["llama3.1", "llama2", "qwen2"])
    args = parse.parse_args()
    temp = template[args.temp]

    with ProcessPoolExecutor() as executor:
        for task in tasks:
            executor.submit(execute_command, task)
