import argparse
import json
import os
import yaml
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


@dataclass
class InferenceConfig:
    model_path: str
    data_file: str
    output_dir: str
    subset_name: str
    max_new_tokens: int
    window_size: int
    group_sizes: str
    topk: int
    use_topk: bool
    all_dims: List[int]
    save_name: str
    selected_dim_path: str
    model_type: str


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
        if config.all_dims is None:
            self.config.all_dims = [0, 8, 16, 24, 32, 40, 48, 56, 64]

    def _detect_model_type(self):
        if self.config.model_type is not None:
            return self.config.model_type.lower()
        model_path = self.config.model_path.lower()
        if "qwen" in model_path:
            return "qwen"
        elif "llama" in model_path:
            return "llama"
        elif "mistral" in model_path:
            return "mistral"
        try:
            config = AutoConfig.from_pretrained(self.config.model_path)
            model_type = config.model_type.lower()
            if "qwen" in model_type:
                return "qwen"
            elif "llama" in model_type:
                return "llama"
            elif "mistral" in model_type:
                return "mistral"
        except Exception:
            pass
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        model_type = self._detect_model_type()
        print(f"Detected model type: {model_type}")

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
            self.config.selected_dim_path,
        )
        self.model = ModelClass.from_pretrained(
            self.config.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        print("Model loading completed")

    def load_test_data(self):
        print(f"Loading test data: {self.config.data_file}")
        try:
            with open(self.config.data_file, "r") as f:
                self.test_data = json.load(f)
            print(f"Successfully loaded {len(self.test_data)} test examples")
        except Exception as e:
            print(f"Failed to load test data: {e}")
            raise

    def evaluate(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Please call load_model() first")
        if self.test_data is None:
            raise ValueError("Please call load_test_data() first")
        results_file = os.path.join(
            self.config.output_dir,
            f"{self.config.save_name}-{self.config.subset_name}.jsonl",
        )
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "a") as fw:
            start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            header = (
                f"--------------------- <New RUN> {start_time}-----------------------"
            )
            print(header)
            fw.write(f"{header}\n")
            scores = []
            for i, test_sample in enumerate(
                tqdm(self.test_data, desc="Evaluation progress")
            ):
                text_inputs = test_sample["prompt"]
                expected_answer = test_sample["expected_answer"]
                prompt_length = test_sample["prompt_length"]
                result = self._generate_and_evaluate(
                    text_inputs, expected_answer, prompt_length
                )
                scores.append(result["score"])
                print(
                    f"\n----------------- Sample {i+1}/{len(self.test_data)} -----------------"
                )
                print(f"Input length: {prompt_length}")
                print(f"Model prediction: {result['pred']}")
                print(f"Score: {result['score']:.2f}%")
                print(f"Current average score: {sum(scores) / len(scores):.2f}%")
                fw.write(f"avg score: {sum(scores) / len(scores):.2f}\n")
                fw.write(json.dumps(result) + "\n")
                fw.flush()
                self.results.append(result)
            final_avg = sum(scores) / len(scores) if scores else 0
            fw.write(f"Final average score: {final_avg:.2f}%\n")
            print(f"\nEvaluation completed! Final average score: {final_avg:.2f}%")
        summary = {
            "average_score": final_avg,
            "num_samples": len(self.test_data),
            "model_path": self.config.model_path,
            "results_file": results_file,
        }
        return summary

    def _generate_and_evaluate(
        self, prompt: str, expected_answers: List[str], prompt_length: Optional[int] = None
    ) -> Dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
            self.model.device
        )
        if prompt_length is None:
            prompt_length = inputs.input_ids.size()[-1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.config.max_new_tokens
            )
        prediction = self.tokenizer.decode(outputs[0][prompt_length:])
        prediction = " ".join(prediction.split())
        score = self._calculate_score(prediction, expected_answers)
        return {
            "ctx_len": prompt_length,
            "pred": prediction,
            "needle": expected_answers,
            "score": score,
            "prompt": prompt[:10000] + "..." if len(prompt) > 10000 else prompt,
        }

    def _calculate_score(self, prediction: str, expected_answers: List[str]) -> float:
        correct_count = sum(
            1 for ans in expected_answers if ans.lower() in prediction.lower()
        )
        return (correct_count / len(expected_answers)) * 100 if expected_answers else 0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate language models on Needle-in-a-Haystack task"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--data_file", type=str, required=True, help="Test data file path")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--subset", type=str, default="default", help="Model subset name")
    parser.add_argument("--topk", type=int, default=32, help="DPE topk value")
    parser.add_argument("--use_topk", action="store_true", help="Whether to use topk")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen", "llama", "mistral"],
        help="Model architecture type (default: auto-detect)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = InferenceConfig(
        model_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        subset_name=args.subset,
        max_new_tokens=args.max_tokens,
        topk=args.topk,
        use_topk=args.use_topk,
        model_type=args.model_type,
    )
    evaluator = NeedleEvaluator(config)
    try:
        evaluator.load_model()
        evaluator.load_test_data()
        results = evaluator.evaluate()
        print(f"Evaluation completed, results saved to: {results['results_file']}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()