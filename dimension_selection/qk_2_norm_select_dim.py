import torch
import os
from tqdm import tqdm
# run sample_qk.py to generate sampled query/key states before running this script
data_dir = "./dimension_selection/sampled_qk_results/llama3-8b-Instruct" # Directory of sampled query/key states: shape [sample_idx, layers, batch_size, num_heads, seq_len, head_dim]
output_dir = "./dimension_selection/result/llama3-8b-Instruct"

num_samples = 10
pretrain_length = 8192 # set the pre-trained length of the model, llama3-8b-Instruct:8192, llama3.1-8b-Instruct:131072
# Automatically generate lengths from pretrain_length down to 2048
lengths = [] # llama3-8b-Instruct: lengths = [2048, 4096, 8192],  llama3.1-8b-Instruct: lengths = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
length = pretrain_length
while length >= 2048:
    lengths.append(length)
    length = length // 2
layers = [f"layer{i}" for i in range(32)]  # "layer0" to "layer31"

def calculate_2_norm(tensor):
    """Compute 2-norm over sequence length dimension."""
    return torch.norm(tensor.reshape(tensor.size(0), tensor.size(1), 2, -1).transpose(-1, -2), p=2, dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match number of query heads."""
    if n_rep == 1:
        return hidden_states
    slen, kv_heads, head_dim = hidden_states.shape
    return hidden_states[:, :, None, :].expand(slen, kv_heads, n_rep, head_dim).reshape(slen, kv_heads * n_rep, head_dim)

# Preallocate result tensor: [num_lengths, num_samples, num_layers, num_heads, head_dim]
qk_2_norm_tensor = torch.zeros(len(lengths), num_samples, len(layers), 32, 64)

for length_idx, length in enumerate(lengths):
    for sample_idx in range(num_samples):
        for layer_idx, layer in enumerate(tqdm(layers, desc=f"Length {length}", ncols=100)): 
            query_path = os.path.join(data_dir, f"{length}/query_states_prompt_key{sample_idx}_{layer}.pt")
            key_path = os.path.join(data_dir, f"{length}/key_states_prompt_key{sample_idx}_{layer}.pt")
            if os.path.exists(query_path) and os.path.exists(key_path):
                query = torch.load(query_path).transpose(1, 2).squeeze(0)
                key = torch.load(key_path).transpose(1, 2).squeeze(0)
                q_norm = calculate_2_norm(query)
                k_norm = repeat_kv(calculate_2_norm(key), 7)
                qk_norm = (q_norm * k_norm).mean(dim=0)
                qk_2_norm_tensor[length_idx, sample_idx, layer_idx] = qk_norm

os.makedirs(output_dir, exist_ok=True)
torch.save(qk_2_norm_tensor, os.path.join(output_dir, "qk_2_norm_selected_dim.pt"))
