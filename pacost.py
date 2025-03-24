import numpy as np
from model import load_model, generate_long_response, extract_clean_answer


def pacost(models, ori_datasets, para_datasets, logger_result, logger_detail):
    results = {}
    for model_name in models:
        logger_result.info("========================")
        logger_detail.info("========================")
        logger_result.info(f"Evaluating PACoST for {model_name}...")
        logger_detail.info(f"Evaluating PACoST for {model_name}...")
        model, tokenizer = load_model(model_name)

        logger_result.info("Using masked dataset AIME-2024 and its paraphrased version:")
        logger_detail.info("Using masked dataset AIME-2024 and its paraphrased version:")
        aime_diff = pacost_aime_inference(
            model, tokenizer, ori_datasets["aime"], para_datasets["aime"], logger_result, logger_detail
        )
        logger_result.info("Using masked dataset MMLU and its paraphrased version:")
        logger_detail.info("Using masked dataset MMLU and its paraphrased version:")
        mmlu_diff = pacost_mmlu_inference(
            model, tokenizer, ori_datasets["mmlu"], para_datasets["mmlu"], logger_result, logger_detail
        )

        results[model_name] = {
            "AIME-2024": aime_diff,
            "MMLU": mmlu_diff,
        }

        logger_result.info(f"AIME-2024 Δ: {aime_diff:.4f}")
        logger_detail.info(f"AIME-2024 Δ: {aime_diff:.4f}")
        logger_result.info(f"MMLU Δ:      {mmlu_diff:.4f}")
        logger_detail.info(f"MMLU Δ:      {mmlu_diff:.4f}")
    return results