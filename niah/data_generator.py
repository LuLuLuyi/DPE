#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import wonderwords

@dataclass
class DataGeneratorConfig:
    model_path: str
    num_test: int
    output_dir: str
    output_file: str
    max_context_length: int = 129500
    num_values_per_key: int = 4
    template: str = (
        "Some special magic numbers are hidden within the following text. Make sure to memorize it. "
        "I will quiz you about the numbers afterwards.\n{context}\nWhat are all the special magic numbers "
        "for {query} mentioned in the provided text? The special magic numbers for {query} mentioned "
        "in the provided text are"
    )
    essay_path: str = 'data/PaulGrahamEssays.json'


class NeedleDataGenerator:
    def __init__(self, config: DataGeneratorConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        
        print("Loading vocabulary resources...")
        self.nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
        self.adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
        self.words = [f"{adj}-{noun}" for adj in self.adjs for noun in self.nouns]
        self.words = sorted(list(set(self.words)))
        print(f"Vocabulary loading completed, {len(self.words)} combination words in total")
        
        print(f"Loading base text: {config.essay_path}")
        with open(config.essay_path, 'r') as f:
            self.essay = json.load(f)["text"]
            
    def generate_test_data(self) -> List[dict]:
        test_data = []
        
        print(f"Starting to generate {self.config.num_test} test samples...")
        for i in tqdm(range(self.config.num_test)):
            needles, expected_answers, key = self._generate_needles()
            
            text_input = self._create_input_context(needles, key)
            
            tokens = self.tokenizer(text_input, return_tensors="pt", return_token_type_ids=False)
            prompt_length = tokens.input_ids.size()[-1]
            
            test_sample = {
                "prompt": text_input,
                "expected_answer": expected_answers,
                "key": key,
                "prompt_length": int(prompt_length),
                "needle_count": len(needles)
            }
            test_data.append(test_sample)
            
        return test_data
    
    def _generate_random_number(self, num_digits=7) -> str:
        lower_bound = 10**(num_digits - 1)
        upper_bound = 10**num_digits - 1
        return str(random.randint(lower_bound, upper_bound))
    
    def _generate_random_segments(self, count: int) -> List[float]:
        segment_length = 0.99 / count
        return [j * segment_length + random.uniform(0, segment_length) for j in range(count)]
    
    def _generate_random_key(self) -> str:
        return random.choice(self.words)
    
    def _generate_needles(self) -> Tuple[List[str], List[str], str]:
        needle_format = "One of the special magic numbers for {key} is: {value}."
        
        key = self._generate_random_key()
        
        values = []
        needles = []
        for _ in range(self.config.num_values_per_key):
            value = self._generate_random_number()            
            values.append(value)
            needles.append(needle_format.format(key=key, value=value))
        
        return needles, values, key
    
    def _create_input_context(self, needles: List[str], key: str) -> str:
        depths = self._generate_random_segments(len(needles))
        
        text_list = re.sub(r'\s+', " ", self.essay).split(".")
        text_chunks = []
        
        step = 50 if self.config.max_context_length > 64000 else 1
        
        for i in range(0, len(text_list), step):
            text_chunk = ".".join(text_list[i:(i + step)])
            text_chunks.append(text_chunk)
            
            needle_positions = [int(depth * len(text_chunks)) for depth in depths]
            needle_positions = sorted(list(set(needle_positions)))
            
            if not needle_positions:
                needle_positions = [0]
            while len(needle_positions) < len(needles):
                needle_positions.append(len(text_chunks))
            
            context_parts = []
            last_pos = 0
            for j, pos in enumerate(needle_positions):
                if j < len(needles):
                    context_parts.append(".".join(text_chunks[last_pos:pos]))
                    context_parts.append(needles[j])
                    last_pos = pos
            
            context_parts.append(".".join(text_chunks[last_pos:]))
            context = " ".join(context_parts)
            
            formatted_prompt = self.config.template.format(context=context, query=key)
            
            chunk_len = len(self.tokenizer(formatted_prompt, return_tensors="pt").input_ids[0])
            if chunk_len > self.config.max_context_length:
                if context[0] == ".":
                    context = context[1:]
                return self.config.template.format(context=context, query=key)
        
        return self.config.template.format(context=" ".join(text_chunks), query=key)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Needle-in-a-Haystack test data")
    parser.add_argument('--model_path', type=str, default="/path/to/model",
                       help="Model path for tokenization")
    parser.add_argument('--num_test', default=100, type=int,
                       help="Number of test samples to generate")
    parser.add_argument('--output_dir', type=str, default="./generated_data",
                       help="Directory to save output data")
    parser.add_argument('--output_file', type=str, default="niah_test_data.json",
                       help="Output data filename")
    parser.add_argument('--max_context_length', default=131072, type=int,
                       help="Maximum context length in tokens")
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = DataGeneratorConfig(
        model_path=args.model_path,
        num_test=args.num_test,
        output_dir=args.output_dir,
        output_file=args.output_file,
        max_context_length=args.max_context_length-1024
    )
    
    generator = NeedleDataGenerator(config)
    
    test_data = generator.generate_test_data()
    
    output_path = os.path.join(args.output_dir, args.output_file)
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Test data has been saved to {output_path}")
    print(f"Generated {len(test_data)} test samples")


if __name__ == "__main__":
    main()
