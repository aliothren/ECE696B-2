import re
import torch
import random
import logging
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_rel

MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
]


def process_log():
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"ts_guess_log/log_{date_time}.txt"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

def prepare_ts_guessing_aime(dataset, mask_token="[MASK]", max_items=50):
    prepared = []
    for sample in dataset.select(range(max_items)):
        text = sample["problem"]
        tokens = text.split()
        for i, tok in enumerate(tokens):
            if len(tok) > 6:
                tokens[i] = mask_token
                break
        masked = " ".join(tokens)
        prepared.append({
            "original": text,
            "masked": masked,
            "answer": str(sample["answer"])
        })
    return prepared

def prepare_ts_guessing_mmlu(dataset, mask_token="[MASK]", max_items=50):
    prepared = []
    for sample in dataset.select(range(max_items)):
        question = sample["question"]
        choices = sample["choices"]
        correct_ans = sample["answer"]
        correct_index = ord(correct_ans.upper()) - 65
        candidates = [i for i in range(len(choices)) if i != correct_index]
        if not candidates:
            continue
        masked_index = random.choice(candidates)
        masked_choices = choices[:]
        original = masked_choices[masked_index]
        masked_choices[masked_index] = mask_token
        prompt = f"{question}\nChoices:\n" + "\n".join([
            f"{chr(65+i)}. {opt}" for i, opt in enumerate(masked_choices)
        ])
        prepared.append({
            "original": question,
            "masked": prompt,
            "masked_option_letter": chr(65 + masked_index),
            "masked_option_text": original,
            "answer": correct_ans
        })
    return prepared

def get_model_fill_in(model, tokenizer, prompt, mask_token="[MASK]"):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**input_ids, max_new_tokens=20)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # 截取 MASK 后的生成部分
    if mask_token in prompt:
        gen_after_mask = generated_text.split(mask_token)[-1].strip()
        first_word = gen_after_mask.split()[0] if gen_after_mask else ""
    else:
        first_word = generated_text.strip().split()[-1]
    return first_word

def run_ts_guessing(model, tokenizer, prepared_data, is_mmlu=False):
    hits = []
    for item in prepared_data:
        guess = get_model_fill_in(model, tokenizer, item["masked"])
        target = item["masked_option_text"] if is_mmlu else item["original"].split()[0]  # 简化目标匹配
        match = int(guess.lower() == target.lower())
        hits.append(match)
    acc = np.mean(hits)
    return acc, hits

def paired_t_test(hits_orig, hits_masked):
    return ttest_rel(hits_orig, hits_masked)


if __name__ == "__main__":
    process_log()
    
    aime_dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    mmlu_dataset = load_dataset("cais/mmlu", "high_school_mathematics", split="test")

    aime_data = prepare_ts_guessing_aime(aime_dataset)
    mmlu_data = prepare_ts_guessing_mmlu(mmlu_dataset)
    

    for model_name in MODELS:
        print("==============================")
        print(f"Evaluating {model_name} with TS-Guessing...")
        model, tokenizer = load_model(model_name)

        aime_acc, aime_hits = run_ts_guessing(model, tokenizer, aime_data, is_mmlu=False)
        print(f"AIME TS-Guessing EM Accuracy: {aime_acc:.3f}")

        mmlu_acc, mmlu_hits = run_ts_guessing(model, tokenizer, mmlu_data, is_mmlu=True)
        print(f"MMLU TS-Guessing EM Accuracy: {mmlu_acc:.3f}")
