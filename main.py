import os
import json
import logging
from datetime import datetime

from datasets import load_dataset
from benchmark import benchmark
from ts_guess import ts_guess
from pacost import pacost

MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  
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


if __name__ == "__main__":
    logger_result, logger_detail = process_log()

    # Task 1: Evaluating Performance on Benchmarks
    with open("dataset/mmlu_high_school_math_100.json", "r", encoding="utf-8") as f:
        mmlu_bench = json.load(f)
    bench_datasets = {
        "aime": load_dataset("HuggingFaceH4/aime_2024")['train'],
        "mmlu": mmlu_bench
    }
    bench_results = benchmark(MODELS, bench_datasets, logger_result, logger_detail)
    logger_result.info(f"Benchmark results:\n{bench_results}")
    logger_detail.info(f"Benchmark results:\n{bench_results}")

    # Task 2: Data Extraction and Analyzing Benchmark Contamination
    # Task 2-1: applying TS-Guessing algorithm
    with open("dataset/aime_2024_mask_delete_question.json", "r", encoding="utf-8") as f:
        aime_masked = json.load(f)
    with open("dataset/mmlu_mask.json", "r", encoding="utf-8") as f:
        mmlu_masked = json.load(f)
    ts_guess_datasets = {
        "aime": aime_masked,
        "mmlu": mmlu_masked
    }
    ts_guess_results = ts_guess(MODELS, ts_guess_datasets, logger_result, logger_detail)
    logger_result.info(f"TS_Guess results:\n{ts_guess_results}")
    logger_detail.info(f"TS_Guess results:\n{ts_guess_results}")
    
    # Task 2-2: applying PaCoST algorithm
    with open("dataset/aime_2024_paraphrase.json", "r", encoding="utf-8") as f:
        aime_paraphrased = json.load(f)
    with open("dataset/mmlu_paraphrased.json", "r", encoding="utf-8") as f:
        mmlu_paraphrased = json.load(f)
    pacost_datasets = {
        "aime": aime_paraphrased,
        "mmlu": mmlu_paraphrased
    }
    pacost_results = pacost(MODELS, bench_datasets, pacost_datasets, logger_result, logger_detail)
    logger_result.info(f"PaCoST results:\n{pacost_results}")
    logger_detail.info(f"PaCoST results:\n{pacost_results}")
    