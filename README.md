<h1 align="center">
<!-- <img src="./fig/.png" width="100" alt="" /> -->
<br>
DPE: Effective Length Extrapolation via Dimension-Wise Positional Embeddings Manipulation
</h1>

<!-- <p align="center">
  <a href=" "><b>[📜 Paper]</b></a> •
  <a href=" "><b>[🤗 HF HUB]</b></a> 
</p> -->

This is the official repository for 📜 [Effective Length Extrapolation via Dimension-Wise Positional Embeddings Manipulation]().

<img src="" width="1000" alt="" />

## TABLE OF CONTENTS
1. [Introdution](#Introdution)
2. [Quick Start](#quick-start)
3. [Apply DPE to New Models](#apply-dpe-to-new-models)
4. [Experiments](#experiments)
5. [Citation](#citation)

## 📖Introduction

### Dimension-Wise Positional Embeddings Manipulation

<div align=center><img src="./fig/method_fig.pdf" width="90%" /></div>



## 🚀 Quick Start
### 1. Requirements
```bash
pip install transformers==4.47.0
# We use flash-attn==2.7.0
pip install flash-attn --no-build-isolation
```
### 2. Load model with DPE (Support Llama, Qwen, Mistral)
```python
model_name = "model_name in ./config/dpe_config.yaml" # support llama, qwen, mistral
model_path = "/path/to/your/model" 
config = AutoConfig.from_pretrained(model_path)
# DPE config, best settings for all the models can be found in ./config
with open('./config/dpe_config.yaml', 'r') as f:
    dpe_config = yaml.safe_load(f)
model_config = dpe_config[model_name]

local_window_size = model_config['local_window_size']
scale_factors = model_config['scale_factors']
dimension_groups_range = model_config['dimension_groups_range']
# If select topk dimensions, set select_topk_dim=True
select_topk_dim = False
topk_dim=48 # number of selected dimensions
selected_dim_path = model_config['selected_dim_path'] # result of identified key dimensions
# apply dpe
replace_with_dpe(scale_factors, local_window_size, dimension_groups_range, select_topk_dim, topk_dim, selected_dim_path)
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
```
### 3. Full inference code
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dpe_for_llama import replace_with_dpe

model_path = "meta-llama/Meta-Llama-3-8B-Instruct" # support llama, qwen, mistral
config = AutoConfig.from_pretrained(model_path)

# DPE
# Settings for Llama-3-8b-Instruct, best settings for all the models can be found in ./config
local_window_size = 1024
scale_factors = [4, 4, 2, 16, 32, 32, 16, 128] # scale size for each dimension group
dimension_groups_range = [0, 8, 16, 24, 32, 40, 48, 56, 64] # dimension range for each group
# If select topk dimensions, set select_topk_dim=True
select_topk_dim = False
topk_dim=48 # number of selected dimensions
selected_dim_path = "dim_select_result/llama3-8b-Instruct/qk_2_norm_selected_dim.pt" # result of identified key dimensions
replace_with_dpe(scale_factors, local_window_size, args.all_dims, select_topk_dim, topk_dim, selected_dim_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
prompt = f"There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 123456. Remember it. 123456 is the pass key.\n " + \
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 4000 + \
    "\nWhat is the pass key?\nThe passkey is "
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
prompt_length = inputs["input_ids"].shape[1]
output = tokenizer.decode(model.generate(**inputs, max_new_tokens=128)[0][prompt_length:], skip_special_tokens=True)
print("[input length]: ",prompt_length)
print("[model output:]", output)

```

## 🪄 Apply DPE to New Models
### 1. Detecting the effective relative distance for different dimensions

We use NIAH (4-needle) to detect our pretrained model

### 2. Identify key dimensions for context extension
#### step 1: sample query and key's hidden states for your model
```bash
python ./dimension_selection/sample_qk.py
```
#### step 2: calulate 2-norm attention contribution for sampled hidden states
```bash
python ./dimension_selection/qk_2_norm_select_dim.py
```
After this step, we can get ./dimension_selection/result/llama3-8b-Instruct/qk_2_norm_selected_dim.pt
#### step 3: Use qk_2_norm_selected_dim.pt for top-k dimension selection
```
selected_dim = torch.load(path/to/qk_2_norm_selected_dim.pt)
selected_dim = torch.topk(selected_dim, dim=-1, k=topk)[1]
```
You can also directly pass the "path/to/qk_2_norm_selected_dim.pt" to dpe_for_llama.py and set select_topk_dim=True.

### 3. Add Your Results to DPE config

# wanxu todo:
## 🔬Experiments 
This section contains the data and code for validating STRING in our paper.

#### Needle In A HayStack (4-needle)
We use NIAH (4-needle) to test our pretrained model, Tinyllama-1.3B. We also evaluate base models (without SFT) from the open-source community using STRING, RoPE, and extrapolation baselines on these tasks. The haystack consists of Paul Graham's essays, and the needles are 6-digit numbers. We report the accuracy of successfully retrieving at least two of the needles, following the Llama 3 report.
```python
cd niah
CUDA_VISIBLE_DEVICES=0 python test_niah_llama.py --test_max_length 131072 --model_path /path/to/llama --shifted_ratio 0.33 (default) --local_value 128 (default)
```

#### RULER
The test and evaluation code is from the official release of [RULER](https://github.com/hsiehjackson/RULER) which contains diverse sythetic tasks to test the long-context ability of LLMs. In this repo, we remove the engineering code from their offical code base but keep all config the same as them. We test Llama3.1 by setting the `max_length` in RULER to `128K`. 
```python
# step 1: generate the test data
cd ruler
python auto_prepare_data.py --model_path /path/to/model --max_length 131072 --temp llama3.1 (or qwen2)

# All tasks
[
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"
]
```
This will generate the processed data for the 13 tasks in RULER and save it to the `data-jsonl` folder.

To test the model on RULER, run the following command:
```python
# step 2: test and evaluate the model
# variable tracking
CUDA_VISIBLE_DEVICES=0 python test_ruler_llama.py --model_path /path/to/llama3 --task vt --data_dir data-jsonl/vt/llama3.1-8b-instruct-131072.jsonl  --shifted_ratio 0.33 (default) --local_value 128 (default)
# niah_multikey_3
CUDA_VISIBLE_DEVICES=0 python test_ruler_llama.py --model_path /path/to/llama3 --task niah_multikey_3 --data_dir data-jsonl/niah_multikey_3/llama3.1-8b-instruct-131072.jsonl  --shifted_ratio 0.33 (default) --local_value 128 (default)

# Qwen2: CUDA_VISIBLE_DEVICES=0 python test_ruler_qwen2.py --model_path /path/to/qwen2 --task vt --data_dir data-jsonl/vt/qwen2-72b-instruct-131072.jsonl --shifted_ratio 0.33 (default) --local_value 128 (default)
```
This command will generate a prediction file in the `Predictions/task_name/model_name/directory`. You can view your generation results and scores in this file and in your stdout. We release the predictions from Llama3.1-STRING 8B/70B [here](https://github.com/HKUNLP/STRING/tree/main/ruler/Predictions). You can also test string with the offical code from RULER by adding one line: `replace_with_string`.

## 🔎Citation

If you find our work helpful or relevant to your research, please kindly cite our paper:
```
```
