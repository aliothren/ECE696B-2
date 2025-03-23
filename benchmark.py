import re
import os
import json
import torch
import logging
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
]


def process_log():
    date_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    log_dir = f"logs/{date_time}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger_result = logging.getLogger("result_logger")
    logger_detail = logging.getLogger("detail_logger")

    logger_result.handlers.clear()
    logger_detail.handlers.clear()
    
    logger_result.setLevel(logging.INFO)
    logger_detail.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    result_path = os.path.join(log_dir, f"results.txt")
    result_handler = logging.FileHandler(result_path)
    result_handler.setFormatter(formatter)
    
    detail_path = os.path.join(log_dir, f"detail.txt")
    detail_handler = logging.FileHandler(detail_path)
    detail_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger_result.addHandler(result_handler)
    logger_result.addHandler(console_handler)
    logger_detail.addHandler(detail_handler)
    logger_detail.addHandler(console_handler)
    
    return logger_result, logger_detail


def generate_long_response(
    model, tokenizer, prompt,
    chunk_size=256, max_rounds=8, temperature=0.6, top_k=50, top_p=0.9,
):
    current_text = prompt
    for round in range(max_rounds):
        logger_detail.info(f"Round: {round}")
        inputs = tokenizer(current_text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=chunk_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True
            )[0]
        
        full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        if len(full_text) <= len(current_text):
            break
        
        new_generated_part = full_text[len(current_text):]
        logger_detail.info(new_generated_part)
        current_text = full_text
        if "</think>" in current_text or "Answer" in current_text:
            break

    return current_text


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer


def bench_aime_inference(model, tokenizer, dataset, num_runs=5):
    accuracies = []

    for idx in range(num_runs):
        correct = 0
        total = 0
        
        for sample in dataset:
            logger_result.info(f"Problem {total}/{len(dataset)}:")
            logger_detail.info(f"Problem {total}/{len(dataset)}:")
            problem_text = sample["problem"]
            prompt = (
                f"{problem_text}\n\n"
                "Please put your final answer in the form \\boxed{ }.\n"
                "<think>\n"
            )
            logger_detail.info(f"Prompt: {prompt}")

            full_output = generate_long_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                chunk_size=512,
                max_rounds=15,
                temperature=0.5,
                top_k=50,
                top_p=0.9,
            )
            
            matches = re.findall(r'\\boxed{([^}]*)}', full_output)
            logger_detail.info(matches)
            if matches:
                predicted_answer = matches[-1]
            else:
                predicted_answer = ""

            logger_result.info(f"Extract answer: {predicted_answer}")
            logger_detail.info(f"Extract answer: {predicted_answer}")
            correct_answer = str(sample["answer"]).strip()
            logger_result.info(f"Correct answer: {correct_answer}")
            logger_detail.info(f"Correct answer: {correct_answer}")
            if predicted_answer == correct_answer:
                correct += 1
                logger_result.info("Correct answer.")
                logger_detail.info("Correct answer.")
            else:
                logger_result.info("Wrong answer.")
                logger_detail.info("Wrong answer.")
            total += 1
            logger_result.info(f"{correct} correct answers in {total} questions")
            logger_detail.info(f"{correct} correct answers in {total} questions")

        acc = correct / total
        logger_result.info(f"Acc in run {idx}: {acc}")
        logger_detail.info(f"Acc in run {idx}: {acc}")
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)


def bench_mmlu_inference(model, tokenizer, dataset, num_runs=5):
    accuracies = []

    for idx in range(num_runs):
        correct = 0
        total = 0
        
        for sample in dataset:
            logger_result.info(f"Question {total}/{len(dataset)}:")
            logger_detail.info(f"Question {total}/{len(dataset)}:")
            question_text = sample["question"]
            choices = sample["choices"]  # list of strings
            prompt = (
                f"Question: {question_text}\n" +
                "Choices:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices)]) +
                "\n\nPlease put your final answer in the form \\boxed{ }.\n"
                "<think>\n"
            )
            logger_detail.info(f"Prompt: {prompt}")

            full_output = generate_long_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                chunk_size=512,
                max_rounds=15,
                temperature=0.5,
                top_k=50,
                top_p=0.9,
            )

            matches = re.findall(r'\\boxed{([^}]*)}', full_output)
            logger_detail.info(matches)
            if matches:
                predicted_answer = matches[-1]
            else:
                predicted_answer = ""

            logger_result.info(f"Extract answer: {predicted_answer}")
            logger_detail.info(f"Extract answer: {predicted_answer}")
            correct_answer = str(sample["answer"]).strip()
            logger_result.info(f"Correct answer: {correct_answer}")
            logger_detail.info(f"Correct answer: {correct_answer}")
            if predicted_answer == correct_answer:
                correct += 1
                logger_result.info("Correct answer.")
                logger_detail.info("Correct answer.")
            else:
                logger_result.info("Wrong answer.")
                logger_detail.info("Wrong answer.")
            total += 1
            logger_result.info(f"{correct} correct answers in {total} questions")
            logger_detail.info(f"{correct} correct answers in {total} questions")
            
        acc = correct / total
        logger_result.info(f"Acc in run {idx}: {acc}")
        logger_detail.info(f"Acc in run {idx}: {acc}")
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)


def benchmark(models, datasets):
    results = {}
    for model_name in models:
        logger_result.info("========================")
        logger_detail.info("========================")
        logger_result.info(f"Evaluating {model_name}...")
        logger_detail.info(f"Evaluating {model_name}...")
        model, tokenizer = load_model(model_name)
        
        logger_result.info("Using dataset AIME-2024:")
        logger_detail.info("Using dataset AIME-2024:")
        aime_mean, aime_std = bench_aime_inference(model, tokenizer, datasets["aime_dataset"])
        logger_result.info("Using dataset MMLU:")
        logger_detail.info("Using dataset MMLU:")
        mmlu_mean, mmlu_std = bench_mmlu_inference(model, tokenizer, datasets["mmlu_dataset"])

        results[model_name] = {
            "AIME-2024": {"mean": aime_mean, "std": aime_std},
            "MMLU": {"mean": mmlu_mean, "std": mmlu_std},
        }

        logger_result.info(f"AIME-2024: {aime_mean:.3f} ± {aime_std:.3f}")
        logger_detail.info(f"AIME-2024: {aime_mean:.3f} ± {aime_std:.3f}")
        logger_result.info(f"MMLU:      {mmlu_mean:.3f} ± {mmlu_std:.3f}")
        logger_detail.info(f"MMLU:      {mmlu_mean:.3f} ± {mmlu_std:.3f}")
    return results


def ts_guess_aime_inference(model, tokenizer, dataset, num_runs=5):
    accuracies = []

    for idx in range(num_runs):
        correct = 0
        total = 0
        
        for sample in dataset:
            logger_result.info(f"Problem {total}/{len(dataset)}:")
            logger_detail.info(f"Problem {total}/{len(dataset)}:")
            problem_text = sample["problem"]
            prompt = (
                "You are a language model. Do not explain. Do not reason.\n"
                f"Sentence: '{problem_text}'\n"
                "Replace [MASK] with exactly one word. Only output the one word in \\boxed{ }.\n"
            )
            logger_detail.info(f"Prompt: {prompt}")

            full_output = generate_long_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                chunk_size=512,
                max_rounds=1,
                temperature=0.5,
                top_k=50,
                top_p=0.9,
            )
            
            matches = re.findall(r'\\boxed{([^}]*)}', full_output)
            logger_detail.info(matches)
            if matches:
                predicted_answer = matches[-1]
            else:
                predicted_answer = ""

            logger_result.info(f"Extract answer: {predicted_answer}")
            logger_detail.info(f"Extract answer: {predicted_answer}")
            correct_answer = str(sample["keyword"]).strip()
            logger_result.info(f"Correct answer: {correct_answer}")
            logger_detail.info(f"Correct answer: {correct_answer}")
            if predicted_answer == correct_answer:
                correct += 1
                logger_result.info("Correct answer.")
                logger_detail.info("Correct answer.")
            else:
                logger_result.info("Wrong answer.")
                logger_detail.info("Wrong answer.")
            total += 1
            logger_result.info(f"{correct} correct answers in {total} questions")
            logger_detail.info(f"{correct} correct answers in {total} questions")

        acc = correct / total
        logger_result.info(f"Acc in run {idx}: {acc}")
        logger_detail.info(f"Acc in run {idx}: {acc}")
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)


def ts_guess_mmlu_inference(model, tokenizer, dataset, num_runs=5):
    accuracies = []

    for idx in range(num_runs):
        correct = 0
        total = 0
        
        for sample in dataset:
            logger_result.info(f"Question {total}/{len(dataset)}:")
            logger_detail.info(f"Question {total}/{len(dataset)}:")
            question_text = sample["question"]
            choices = sample["choices"]  # list of strings
            prompt = (
                "You are a language model. Do not explain. Do not reason.\n"
                "Complete the [MASK] in the first option of the choices based on your benchmark knowledge.\n\n"
                f"{question_text}\n" +
                "Choices:\n" + "\n".join([f"** {opt} **" for i, opt in enumerate(choices)]) +
                "\n\nReply with answer only. Only put your answer in the form \\boxed{ }.\n"
            )
            logger_detail.info(f"Prompt: {prompt}")

            full_output = generate_long_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                chunk_size=512,
                max_rounds=1,
                temperature=0.5,
                top_k=50,
                top_p=0.9,
            )

            matches = re.findall(r'\\boxed{([^}]*)}', full_output)
            logger_detail.info(matches)
            if matches:
                predicted_answer = matches[-1]
            else:
                predicted_answer = ""

            logger_result.info(f"Extract answer: {predicted_answer}")
            logger_detail.info(f"Extract answer: {predicted_answer}")
            correct_answer = str(sample["keyword"]).strip()
            logger_result.info(f"Correct answer: {correct_answer}")
            logger_detail.info(f"Correct answer: {correct_answer}")
            if predicted_answer == correct_answer:
                correct += 1
                logger_result.info("Correct answer.")
                logger_detail.info("Correct answer.")
            else:
                logger_result.info("Wrong answer.")
                logger_detail.info("Wrong answer.")
            total += 1
            logger_result.info(f"{correct} correct answers in {total} questions")
            logger_detail.info(f"{correct} correct answers in {total} questions")
            
        acc = correct / total
        logger_result.info(f"Acc in run {idx}: {acc}")
        logger_detail.info(f"Acc in run {idx}: {acc}")
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)


def ts_guess(models, datasets):
    results = {}
    for model_name in models:
        logger_result.info("========================")
        logger_detail.info("========================")
        logger_result.info(f"Evaluating {model_name}...")
        logger_detail.info(f"Evaluating {model_name}...")
        model, tokenizer = load_model(model_name)
        
        logger_result.info("Using masked dataset AIME-2024:")
        logger_detail.info("Using masked dataset AIME-2024:")
        aime_mean, aime_std = ts_guess_aime_inference(model, tokenizer, datasets["aime_dataset"])
        logger_result.info("Using masked dataset MMLU:")
        logger_detail.info("Using masked dataset MMLU:")
        mmlu_mean, mmlu_std = ts_guess_mmlu_inference(model, tokenizer, datasets["mmlu_dataset"])

        results[model_name] = {
            "AIME-2024": {"mean": aime_mean, "std": aime_std},
            "MMLU": {"mean": mmlu_mean, "std": mmlu_std},
        }

        logger_result.info(f"AIME-2024: {aime_mean:.3f} ± {aime_std:.3f}")
        logger_detail.info(f"AIME-2024: {aime_mean:.3f} ± {aime_std:.3f}")
        logger_result.info(f"MMLU:      {mmlu_mean:.3f} ± {mmlu_std:.3f}")
        logger_detail.info(f"MMLU:      {mmlu_mean:.3f} ± {mmlu_std:.3f}")
    return results


if __name__ == "__main__":
    logger_result, logger_detail = process_log()

    
    # Task 1: Evaluating Performance on Benchmarks
    # bench_datasets = {
    #     "aime_dataset": load_dataset("HuggingFaceH4/aime_2024")['train'],
    #     "mmlu_dataset": load_dataset("cais/mmlu", "high_school_mathematics")['test']
    # }
    # bench_results = benchmark(MODELS, bench_datasets)

    # Task 2: Data Extraction and Analyzing Benchmark Contamination
    # Task 2-1: applying TS-Guessing algorithm
    with open("aime_2024_mask_delete_question.json", "r", encoding="utf-8") as f:
        aime_mask_dataset = json.load(f)
    with open("mmlu_mask.json", "r", encoding="utf-8") as f:
        mmlu_mask_dataset = json.load(f)
    ts_guess_datasets = {
        "aime_dataset": aime_mask_dataset,
        "mmlu_dataset": mmlu_mask_dataset
    }
    ts_guess_results = ts_guess(MODELS, ts_guess_datasets)