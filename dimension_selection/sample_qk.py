import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from sample_qk_atten_forward import LlamaFlashAttention2
import transformers

# Load data
json_path = "./dimension_selection/data/PaulGrahamEssays.json"
text_list = json.load(open(json_path))["text"].split(".")

# Sampling lengths
lengths = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
num_samples = 10

# Output directory for saved query/key tensors
output_dir = "./dimension_selection/sampled_qk_results/llama3-8b-Instruct"
os.makedirs(output_dir, exist_ok=True)

# Load model and tokenizer
model_name = "path/to/your/model"  # update with actual path or HF name
transformers.models.llama.modeling_llama.LlamaAttention.forward = LlamaFlashAttention2
transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = LlamaFlashAttention2
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",attn_implementation="flash_attention_2")

# Loop over lengths and sample indices
for length in lengths:
    os.makedirs(os.path.join(output_dir, str(length)), exist_ok=True)
    for sample_idx in range(num_samples):
        model.config.prompt_key = sample_idx  # Used in custom attention module
        model.config.output_dir = os.path.join(output_dir, str(length))

        # Get enough text to tokenize to desired length
        text = ".".join(text_list[sample_idx * 100 :])  # Ensure enough context
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=length).input_ids.to(model.device)

        if input_ids.shape[1] < length:
            print(f"Skipping sample {sample_idx} at length {length} due to insufficient tokens ({input_ids.shape[1]})")
            continue

        input_ids = input_ids[:, :length]

        with torch.no_grad():
            _ = model(input_ids)

print("✅ Sampling completed and query/key states saved.")