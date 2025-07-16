import importlib
import re
import yaml
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import sys
import json
from utils import read_manifest
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class InferenceConfig:
    model_path: str
    data_file: str
    output_dir: str
    task_type: str
    max_new_tokens: int
    window_size: int
    group_sizes: str
    topk: int
    use_topk: bool
    selected_dim_path: str
    seq_len: int
    model_type: Optional[str] = None
    all_dims: Optional[List[int]] = None


class NeedleEvaluator:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.test_data = None
        self.results = []
        self.dpe_config = {}
        try:
            with open("dpe_config.yaml", "r") as f:
                self.dpe_config = yaml.safe_load(f)
        except FileNotFoundError:
            pass
        if self.config.all_dims is None:
            self.config.all_dims = [0, 8, 16, 24, 32, 40, 48, 56, 64]

    def _detect_model_type(self):
        if self.config.model_type is not None:
            return self.config.model_type.lower()

        model_path_lower = self.config.model_path.lower()
        if "qwen" in model_path_lower:
            return "qwen"
        elif "llama" in model_path_lower:
            return "llama"
        elif "mistral" in model_path_lower:
            return "mistral"

        try:
            config = AutoConfig.from_pretrained(self.config.model_path)
            model_type_from_config = config.model_type.lower()
            if "qwen" in model_type_from_config:
                return "qwen"
            elif "llama" in model_type_from_config:
                return "llama"
            elif "mistral" in model_type_from_config:
                return "mistral"
        except Exception:
            pass

        print("Warning: Could not auto-detect model type from path or config. Defaulting to 'qwen'.")
        return "qwen"

    def _load_dpe_module(self, model_type):
        if model_type == "qwen":
            from dpe_for_qwen import replace_with_dpe
        elif model_type == "llama":
            from dpe_for_llama import replace_with_dpe
        elif model_type == "mistral":
            from dpe_for_mistral import replace_with_dpe
        else:
            raise ValueError(f"Unsupported model type for DPE: {model_type}")
        return replace_with_dpe

    def load_model(self):
        print(f"Loading model: {self.config.model_path}")
        model_name = os.path.basename(os.path.normpath(self.config.model_path))
        dpe_params = self.dpe_config.get(model_name, None)
        if dpe_params:
            if "local_window_size" in dpe_params:
                self.config.window_size = dpe_params["local_window_size"]
            if "scale_factors" in dpe_params:
                self.config.group_sizes = dpe_params["scale_factors"]
            if "dimension_groups_range" in dpe_params:
                self.config.all_dims = dpe_params["dimension_groups_range"]
            if "selected_dim_path" in dpe_params:
                self.config.selected_dim_path = dpe_params["selected_dim_path"]
        else:
            raise ValueError(f"No DPE parameters found for model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        model_type = self._detect_model_type()
        self.config.model_type = model_type
        print(f"Detected/Confirmed model type: {model_type}")

        replace_with_dpe = self._load_dpe_module(model_type)

        print(
            f"Applying Dimensional DPE: group_sizes={self.config.group_sizes}, window_size={self.config.window_size}, "
            f"dims_range={self.config.all_dims}, use_topk={self.config.use_topk}, topk={self.config.topk}, "
            f"selected_dim_path={self.config.selected_dim_path}"
        )

        replace_with_dpe(
            self.config.group_sizes,
            self.config.window_size,
            self.config.all_dims,
            self.config.use_topk,
            self.config.topk,
            self.config.selected_dim_path
        )

        if model_type == "qwen":
            from transformers import Qwen2ForCausalLM as ModelClass
        elif model_type == "llama":
            from transformers import LlamaForCausalLM as ModelClass
        elif model_type == "mistral":
            from transformers import MistralForCausalLM as ModelClass
        else:
             raise ValueError(f"Unsupported model type: {model_type}")

        self.model = ModelClass.from_pretrained(
            self.config.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        print("Model loading completed")

    def load_test_data(self):
        print(f"Loading test data: {self.config.data_file}")
        try:
            self.test_data = read_manifest(self.config.data_file)
            print(f"Successfully loaded {len(self.test_data)} test examples")
        except Exception as e:
            print(f"Failed to load test data from {self.config.data_file}: {e}")
            raise

    def evaluate(self, infer_size: int = -1) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Please call load_model() first")
        if self.test_data is None:
            raise ValueError("Please call load_test_data() first")

        model_name_for_path = os.path.basename(os.path.normpath(self.config.model_path))
        pred_save_path = os.path.join(self.config.output_dir, self.config.task_type, model_name_for_path)
        os.makedirs(pred_save_path, exist_ok=True)

        group_str = self.config.group_sizes.replace('-', '')
        filename = f"seqlen{self.config.seq_len}_w{self.config.window_size}_g{group_str}_t{self.config.topk}.jsonl"
        results_file = os.path.join(pred_save_path, filename)

        print(f"Evaluation results will be saved to: {results_file}")

        with open(results_file, "a") as fw:
            start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"--------------------- <New RUN {start_time_str}> -----------------------"
            print(header)
            fw.write(f"{header}\n")

            scores = []
            save_ds = []
            num_processed = 0
            for i, data in enumerate(tqdm(self.test_data, desc="Evaluation Progress")):
                if infer_size != -1 and i >= infer_size:
                    print(f"Reached inference size limit: {infer_size}")
                    break

                text_inputs = data["input"]
                expected_answers = data["outputs"]

                result = self._generate_and_evaluate(text_inputs, expected_answers)
                scores.append(result["score"])
                save_ds.append(result)

                current_avg_score = sum(scores) / len(scores) if scores else 0.0

                print(f"\n----------------- Sample {i+1}/{len(self.test_data)} -----------------")
                print(f"Input length: {result['ctx_len']}")
                print(f"[Model Prediction] {result['pred']}")
                print(f"[Ground Truth] {result['needle']}")
                print(f"[Score]: {result['score']:.4f}")
                print(f"Current average score: {current_avg_score:.4f}")

                log_line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, step {i}, ctx len {result['ctx_len']}, avg score {current_avg_score:.4f}\n"
                fw.write(log_line)
                fw.flush()
                num_processed += 1

            for save_d in save_ds:
                 fw.write(json.dumps(save_d) + '\n')

            final_avg = sum(scores) / len(scores) if scores else 0.0
            fw.write(f"avg_score:{final_avg:.4f}\n")
            print(f"\nEvaluation completed! Final average score: {final_avg:.4f}")

        summary = {
            "average_score": final_avg,
            "num_samples_processed": num_processed,
            "total_samples_in_file": len(self.test_data),
            "model_path": self.config.model_path,
            "results_file": results_file,
            "config": self.config.__dict__
        }
        return summary

    def _generate_and_evaluate(self, prompt: str, expected_answers: List[str]) -> Dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(self.model.device)
        prompt_length = inputs.input_ids.size()[-1]

        with torch.no_grad():
            sample = self.model.generate(
                **inputs,
                repetition_penalty=1,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens
            )

        prediction = self.tokenizer.decode(sample[0][prompt_length:])
        prediction = " ".join(prediction.split())

        score = self._calculate_score(prediction, expected_answers)

        return {
            "ctx_len": prompt_length,
            "pred": prediction,
            "needle": expected_answers,
            "score": score,
        }

    def _calculate_score(self, prediction: str, expected_answers: List[str]) -> float:
         if "qa" in self.config.task_type:
             score = float(max([ans.lower() in prediction.lower() for ans in expected_answers]) if expected_answers else 0.0)
         else:
             if not expected_answers:
                 return 0.0
             score_curr = [1.0 if ans.lower() in prediction.lower() else 0.0 for ans in expected_answers]
             score = sum(score_curr) / len(score_curr)
         return score


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate language models with DPE on Needle-in-a-Haystack tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--data_dir", type=str, help="Directory containing the data file")
    parser.add_argument("--benchmark", type=str, default='synthetic', help="Benchmark name")
    parser.add_argument("--task", type=str, required=True, help="Task identifier")
    parser.add_argument('--topk', type=int, default=32, help="Top-k value for DPE attention")
    parser.add_argument('--use_topk', action='store_true', help="Enable top-k attention in DPE")
    parser.add_argument('--max_new_tokens', type=int, default=-1, help="Maximum number of new tokens to generate")
    parser.add_argument('--model_type', type=str, choices=["qwen", "llama", "mistral"], default=None, help="Specify model type explicitly")
    parser.add_argument('--infer_size', type=int, default=-1, help="Limit inference to the first N samples")
    parser.add_argument('--seq_len', type=int, default=-1, help="Sequence length identifier for output filename")
    parser.add_argument('--output_dir_base', type=str, default="eval/ruler/Predictions", help="Base directory for saving prediction files")

    return parser.parse_args()

def main():
    args = parse_arguments()

    seq_len = args.seq_len

    max_new_tokens = args.max_new_tokens
    if max_new_tokens == -1:
        task_lower = args.task.lower()
        if "vt" in task_lower:
            max_new_tokens = 30
        elif "cwe" in task_lower:
            max_new_tokens = 120
        elif "fwe" in task_lower:
            max_new_tokens = 50
        elif "qa" in task_lower:
            max_new_tokens = 32
        elif "niah" in task_lower:
            max_new_tokens = 128
        else:
            print(f"Warning: Unknown task '{args.task}' for default max_new_tokens. Using 128.")
            max_new_tokens = 128
        print(f"Using default max_new_tokens for task '{args.task}': {max_new_tokens}")
    else:
        print(f"Using user-specified max_new_tokens: {max_new_tokens}")

    data_file = args.data_dir

    curr_folder = os.path.dirname(os.path.abspath(__file__))
    benchmark_file = os.path.join(curr_folder, f"{args.benchmark}.yaml")
    try:
        with open(benchmark_file, "r") as f:
            tasks_customized = yaml.safe_load(f)
            if args.task not in tasks_customized:
                 print(f"Warning: Task '{args.task}' not found in '{benchmark_file}'. Proceeding anyway.")
            else:
                 print(f"Task '{args.task}' validated against '{benchmark_file}'.")
    except FileNotFoundError:
        print(f"Warning: Benchmark file '{benchmark_file}' not found. Skipping task validation.")
    except Exception as e:
        print(f"Warning: Error reading benchmark file '{benchmark_file}': {e}. Skipping task validation.")


    config = InferenceConfig(
        model_path=args.model_path,
        data_file=data_file,
        output_dir=args.output_dir_base,
        task_type=args.task,
        max_new_tokens=max_new_tokens,
        window_size=args.window_size,
        group_sizes=args.group_sizes,
        topk=args.topk,
        use_topk=args.use_topk,
        selected_dim_path=args.selected_dim_path,
        seq_len=seq_len,
        model_type=args.model_type,
    )

    evaluator = NeedleEvaluator(config)
    try:
        evaluator.load_model()
        evaluator.load_test_data()
        results = evaluator.evaluate(infer_size=args.infer_size)
        print(f"\nEvaluation Summary:")
        print(f"  Average Score: {results['average_score']:.4f}")
        print(f"  Samples Processed: {results['num_samples_processed']}")
        print(f"  Results File: {results['results_file']}")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
