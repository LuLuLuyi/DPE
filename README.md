<h1 align="center">
<!-- <img src="./fig/.png" width="100" alt="" /> -->
<br>
DPE: Effective Length Extrapolation via Dimension-Wise Positional Embeddings Manipulation
</h1>

<!-- <p align="center">
  <a href=" "><b>[📜 Paper]</b></a> •
  <a href=" "><b>[🤗 HF HUB]</b></a> 
</p> -->

This is the official repository for 📜 [Effective Length Extrapolation via Dimension-Wise Positional Embeddings Manipulation](https://arxiv.org/abs/2504.18857).

<img src="./fig/method_fig.pdf" width="1000" alt="" />

## TABLE OF CONTENTS
1. [Introdution](#Introdution)
2. [Quick Start](#quick-start)
3. [Apply DPE to New Models](#apply-dpe-to-new-models)
4. [Experiments](#experiments)
5. [Citation](#citation)

## 📖Introduction

### Dimension-Wise Positional Embeddings Manipulation
We introduce **Dimension-Wise Positional Embeddings Manipulation (DPE)**, a **training-free** and **plug-and-play** framework for context extension. DPE identifies and selectively manipulates the **most influential dimensions** of the rotary positional embeddings by analyzing their **attention contribution** to attention scores. This strategy optimizes the model’s dimensional adaptation with minimal modifications to the pretrained model, enabling each dimension to extrapolate in an optimal manner.

DPE proceeds in three steps:
a. **Detecting the Effective Relative Distance**:  We partitioning the model’s hidden dimensions into several groups (in practice, we divide them evenly into eight groups). For each group, we detect the most effective relative distances via a detecting task using NIAH.
b. **Identifying the Key Dimensions**: The key dimensions are identified using their attention contribution. 
c. **Scaling the Position Indices**: For each selected dimension, we apply a customized scaling factor derived from its most effective relative distance.

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

<!-- ## 🪄 Apply DPE to New Models
### 1. Detecting the effective relative distance for different dimensions

We use NIAH (4-needle) to detect our pretrained model

After this step, we can get scale_factors for different dimension groups. -->

## 🪄 Apply DPE to New Models
### 1. Detecting the effective relative distance for different dimensions

We use NIAH (4-needle) to detect the optimal scale factors for our pretrained model.

#### step 1: Generate data of the appropriate length
Generate the test data by running the data generator script. You can modify the `MAX_CONTEXT_LENGTH` parameter to adjust the data length.

```bash
bash ./niah/scripts/run_data_generator.sh
```

#### step 2: Run the detection process

Execute the NIAH detection script with your specific parameters:
- Set `TEST_MAX_LENGTH` to your desired test sequence length
- Set `PRETRAIN_LENGTH` to your model's pretraining context length

```bash
bash ./effective_length_detection/run_niah_detect.sh
```

After this step, you'll obtain results for different detection groups. By sorting these results, you can identify the optimal dimensions for your model and their corresponding scale factors.
### 2. Identify key dimensions for context extension
#### step 1: Sample query and key's hidden states for your model
```bash
python ./dimension_selection/sample_qk.py
```
#### step 2: Calulate 2-norm attention contribution for sampled hidden states
```bash
python ./dimension_selection/qk_2_norm_select_dim.py
```
After this step, we can get ./dimension_selection/result/llama3-8b-Instruct/qk_2_norm_selected_dim.pt
<!-- #### step 3: Use qk_2_norm_selected_dim.pt for top-k dimension selection
```
selected_dim = torch.load(path/to/qk_2_norm_selected_dim.pt)
selected_dim = torch.topk(selected_dim, dim=-1, k=topk)[1]
```
You can also directly pass the "path/to/qk_2_norm_selected_dim.pt" to dpe_for_llama.py and set select_topk_dim=True. -->

### 3. Add the Results to DPE config
```yaml
"llama3-8b-instruct":
  local_window_size: 1024
  scale_factors: [4, 4, 2, 16, 32, 32, 16, 128]
  dimension_groups_range: [0, 8, 16, 24, 32, 40, 48, 56, 64]
  selected_dim_path : "dim_select_result/llama3-8b-Instruct/qk_2_norm_selected_dim.pt"
```

## 🔬Experiments 
This section contains the data and code for validating DPE in our paper.

#### Needle In A HayStack (4-needle)
We use NIAH (4-needle) to detect effective relative distance and test the latest models from the open-source community using DPE and RoPE. The haystack consists of Paul Graham's essays, and the needles are 6-digit numbers.
To run the evaluation:
```bash
cd niah/scripts
bash niah.sh
```

#### RULER
The test and evaluation code is from the official release of [RULER](https://github.com/hsiehjackson/RULER) which contains diverse sythetic tasks to test the long-context ability of LLMs.
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
cd scripts
bash ruler_dpe.sh
```
This command will generate a prediction file in the `Predictions/task_name/model_name/directory`. You can view your generation results and scores in this file and in your stdout. We release the predictions from Llama3.1-STRING 8B/70B [here](https://github.com/HKUNLP/STRING/tree/main/ruler/Predictions). You can also test string with the offical code from RULER by adding one line: `replace_with_string`.

## 🔎Citation

If you find our work helpful or relevant to your research, please kindly cite our paper:
```
@misc{lu2025effectivelengthextrapolationdimensionwise,
      title={Effective Length Extrapolation via Dimension-Wise Positional Embeddings Manipulation}, 
      author={Yi Lu and Wanxu Zhao and Xin Zhou and Chenxin An and Chenglong Wang and Shuo Li and Yuming Yang and Jun Zhao and Tao Ji and Tao Gui and Qi Zhang and Xuanjing Huang},
      year={2025},
      eprint={2504.18857},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.18857}, 
}
```
