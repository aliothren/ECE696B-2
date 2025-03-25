import torch
import numpy as np
from model_generation import load_model
from scipy.stats import ttest_rel


def pacost_score_single_sample(model, tokenizer, sample, keyword, logger_result, logger_detail):
    text = sample.strip()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)

    blacklist_token_ids = {
    220,    # 'Ġ' （表示空格起始 token）
    715,    # 'ĠĊ'
    4710,   # 'ĠĊĊ'
    57960,  # 'Ġ$\''
    2503,   # 'Ġ...'
    1124,   # 'Ġ\''
    23754,  # 'Ġ?ĊĊ'
    9338,   # '...Ċ'
    12236,
    1304,
    508,
    400,
    7436,
    5468,
    2146,
    1112,
    21903,
    17607,
    60353,
    4593,
    2303,
    32671,
    }

    # 关键 token 获取
    keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
    if len(keyword_ids) == 0:
        return 0.0

    first_token_id = keyword_ids[0]
    token_str = tokenizer.decode([first_token_id])

    # 打印 token 信息
    logger_result.info(f"First token: {first_token_id} ({token_str})")
    logger_detail.info(f"Keyword: {keyword} -> First token: '{token_str}' (ID={first_token_id})")

    # 屏蔽黑名单 token 后再 softmax
    logits_vec = logits[0, -1].clone()
    logits_vec[list(blacklist_token_ids)] = float('-inf')
    probs = torch.nn.functional.softmax(logits_vec, dim=-1)

    # 如果 target token 本身就是垃圾，直接跳过
    if first_token_id in blacklist_token_ids:
        logger_result.info(f"Token '{token_str}' is in blacklist. Skipped.")
        logger_detail.info(f"Token '{token_str}' is in blacklist. Skipped.")
        return 0.0

    target_prob = probs[first_token_id].item()

    # 计算排名 & top-k 信息
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    rank = (sorted_indices == first_token_id).nonzero(as_tuple=True)[0].item() + 1

    topk = 5
    topk_ids = sorted_indices[:topk].tolist()
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids)
    topk_probs = sorted_probs[:topk].tolist()

    logger_detail.info(f"Target prob = {target_prob:.4e}, Rank = {rank}")
    logger_detail.info("Top-5 predictions:")
    for i in range(topk):
        logger_detail.info(f"{i+1}. Token: '{topk_tokens[i]}' (ID={topk_ids[i]}), Prob = {topk_probs[i]:.4e}")

    return target_prob


# def pacost_score_single_sample(model, tokenizer, sample, keyword, logger_result, logger_detail):
#     text = sample.strip()
#     inputs = tokenizer(text, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits  # shape: (1, seq_len, vocab_size)
# 
#     keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
#     if len(keyword_ids) == 0:
#         return 0.0 
#     first_token_id = keyword_ids[0] 
#     
#     bad_token_strs = [
#         "Ġ", "Ċ", "ĠĊ", "ĠĊĊ",
#     ]
#     blacklist_token_ids = set()
#     for tok in bad_token_strs:
#         try:
#             ids = tokenizer.encode(tok, add_special_tokens=False)
#             blacklist_token_ids.update(ids)
#         except Exception:
#             pass
#     token_str = tokenizer.decode([first_token_id])
#     logger_result.info(f"First token: {first_token_id} ({token_str})")
#     logger_detail.info(f"First token: {first_token_id} ({token_str})")
#     
#     probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
#     if first_token_id in blacklist_token_ids:
#         logger_result.info(f"Token '{tokenizer.decode([first_token_id])}' in blacklist. Skipped.")
#         return 0.0
#     target_prob = probs[first_token_id].item()
# 
#     sorted_probs, sorted_indices = torch.sort(probs, descending=True)
#     rank = (sorted_indices == first_token_id).nonzero(as_tuple=True)[0].item() + 1
#     
#     topk = 5
#     topk_ids = sorted_indices[:topk].tolist()
#     topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids)
#     topk_probs = sorted_probs[:topk].tolist()
# 
#     logger_detail.info(f"Keyword: {keyword} -> First token: '{tokenizer.decode([first_token_id])}' (ID={first_token_id})")
#     logger_detail.info(f"Target prob = {target_prob:.4e}, Rank = {rank}")
#     logger_detail.info("Top-5 predictions:")
#     for i in range(topk):
#         logger_detail.info(f"{i+1}. Token: '{topk_tokens[i]}' (ID={topk_ids[i]}), Prob = {topk_probs[i]:.4e}")
# 
#     return target_prob


def pacost_aime_inference(
    model, tokenizer, original_dataset, paraphrased_dataset,
    logger_result, logger_detail, num_runs=1
):
    logger_result.info("Evaluating PACoST on AIME...")
    logger_detail.info("Evaluating PACoST on AIME...")
    
    all_deltas = []
    all_p_orig, all_p_para = [], []

    for run in range(num_runs):
        logger_result.info(f"Run {run}/{num_runs}")
        logger_detail.info(f"Run {run}/{num_runs}")
        score_diffs = []
        p_orig_list = []
        p_para_list = []

        for i in range(len(original_dataset)):
            logger_result.info(f"Problem {i}/{len(original_dataset)}")
            logger_detail.info(f"Problem {i}/{len(original_dataset)}")
            # ori_prompt = (
            #     f'"problem": "{original_dataset[i]["problem"]}",\n'
            #     '"answer": "'
            # )
            # para_prompt = (
            #     f'"problem": "{paraphrased_dataset[i]["problem"]}",\n'
            #     '"answer": "'
            # )
            ori_prompt = (
                f'Problem: {original_dataset[i]["problem"]},\n'
                'Answer:\n'
            )
            para_prompt = (
                f'Problem: {paraphrased_dataset[i]["problem"]},\n'
                'Answer:\n'
            )
            answer = original_dataset[i]["answer"].lstrip("0")
            logger_detail.info(f"ori_prompt: {ori_prompt}")
            logger_detail.info(f"para_prompt: {para_prompt}")
            logger_detail.info(f"answer: {answer}")

            p_orig = pacost_score_single_sample(model, tokenizer, ori_prompt, answer, logger_result, logger_detail)
            p_para = pacost_score_single_sample(model, tokenizer, para_prompt, answer, logger_result, logger_detail)

            logger_result.info(f"P(orig)={p_orig:.4e}, P(para)={p_para:.4e}")
            logger_detail.info(f"[{i}] Answer = {answer}, P(orig)={p_orig:.4e}, P(para)={p_para:.4e}")
            score_diffs.append(p_orig - p_para)
            p_orig_list.append(p_orig)
            p_para_list.append(p_para)

        mean_diff = np.mean(score_diffs)
        t_stat, p_value = ttest_rel(p_orig_list, p_para_list)

        logger_result.info(f"Run {run}: Δ = {mean_diff:.4e}, t = {t_stat:.4f}, p = {p_value:.4e}")
        logger_detail.info(f"Run {run}: Δ = {mean_diff:.4e}, t = {t_stat:.4f}, p = {p_value:.4e}")
        all_deltas.append(mean_diff)
        all_p_orig.extend(p_orig_list)
        all_p_para.extend(p_para_list)

    mean_diff = np.mean(all_deltas)
    t_stat, p_value = ttest_rel(all_p_orig, all_p_para)

    logger_result.info(f"[AIME Overall] Avg Δ = {mean_diff:.4e}, t = {t_stat:.4f}, p = {p_value:.4e}")
    logger_detail.info(f"[AIME Overall] Avg Δ = {mean_diff:.4e}, t = {t_stat:.4f}, p = {p_value:.4e}")

    return {
        "mean_delta": mean_diff,
        "t_stat": t_stat,
        "p_value": p_value
    }


def pacost_mmlu_inference(
    model, tokenizer, original_dataset, paraphrased_dataset,
    logger_result, logger_detail, num_runs=1
):
    logger_result.info("Evaluating PACoST on MMLU...")
    logger_detail.info("Evaluating PACoST on MMLU...")
    
    all_deltas = []
    all_p_orig, all_p_para = [], []

    for run in range(num_runs):
        logger_result.info(f"Run {run}/{num_runs}")
        logger_detail.info(f"Run {run}/{num_runs}")
        score_diffs = []
        p_orig_list = []
        p_para_list = []

        for i in range(len(original_dataset)):
            logger_result.info(f"Question {i}/{len(original_dataset)}")
            logger_detail.info(f"Question {i}/{len(original_dataset)}")
            choices = original_dataset[i]["choices"]
            # ori_prompt = (
            #     f'"question": "{original_dataset[i]["question"]}",\n'
            #     '"subject": "high_school_mathematics",\n'
            #     '"choices": [\n'
            #     f'"{choices[0]}",\n'
            #     f'"{choices[1]}",\n'
            #     f'"{choices[2]}",\n'
            #     f'"{choices[3]}"\n'
            #     '],\n'
            #     '"answer": '
            # )
            # para_prompt = (
            #     f'"question": "{paraphrased_dataset[i]["question"]}",\n'
            #     '"subject": "high_school_mathematics",\n'
            #     '"choices": [\n'
            #     f'"{choices[0]}",\n'
            #     f'"{choices[1]}",\n'
            #     f'"{choices[2]}",\n'
            #     f'"{choices[3]}"\n'
            #     '],\n'
            #     '"answer": '
            # )
            choices = original_dataset[i]["choices"]
            ori_prompt = (
                f"Question: {original_dataset[i]['question']}\n"
                "Choices:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices)]) + "\n"
                "Answer:\n"
            )
            para_prompt = (
                f"Question: {paraphrased_dataset[i]['question']}\n"
                "Choices:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices)]) + "\n"
                "Answer:\n"
            )
            # answer = str(original_dataset[i]["answer"])
            index = int(original_dataset[i]["answer"])
            answer = " " + chr(65 + index)
            logger_detail.info(f"ori_prompt: {ori_prompt}")
            logger_detail.info(f"para_prompt: {para_prompt}")
            logger_detail.info(f"answer: {answer}")

            p_orig = pacost_score_single_sample(model, tokenizer, ori_prompt, answer, logger_result, logger_detail)
            p_para = pacost_score_single_sample(model, tokenizer, para_prompt, answer, logger_result, logger_detail)

            logger_result.info(f"P(orig)={p_orig:.4e}, P(para)={p_para:.4e}")
            logger_detail.info(f"[{i}] Answer = {answer}, P(orig)={p_orig:.4e}, P(para)={p_para:.4e}")
            score_diffs.append(p_orig - p_para)
            p_orig_list.append(p_orig)
            p_para_list.append(p_para)

        mean_diff = np.mean(score_diffs)
        t_stat, p_value = ttest_rel(p_orig_list, p_para_list)

        logger_result.info(f"Run {run}: Δ = {mean_diff:.4e}, t = {t_stat:.4f}, p = {p_value:.4e}")
        logger_detail.info(f"Run {run}: Δ = {mean_diff:.4e}, t = {t_stat:.4f}, p = {p_value:.4e}")
        all_deltas.append(mean_diff)
        all_p_orig.extend(p_orig_list)
        all_p_para.extend(p_para_list)

    mean_diff = np.mean(all_deltas)
    t_stat, p_value = ttest_rel(all_p_orig, all_p_para)

    logger_result.info(f"[AIME Overall] Avg Δ = {mean_diff:.4e}, t = {t_stat:.4f}, p = {p_value:.4e}")
    logger_detail.info(f"[AIME Overall] Avg Δ = {mean_diff:.4e}, t = {t_stat:.4f}, p = {p_value:.4e}")

    return {
        "mean_delta": mean_diff,
        "t_stat": t_stat,
        "p_value": p_value
    }


def pacost(models, ori_datasets, para_datasets, logger_result, logger_detail):
    results = {}
    for model_name in models:
        logger_result.info("========================")
        logger_detail.info("========================")
        logger_result.info(f"Evaluating PACoST for {model_name}...")
        logger_detail.info(f"Evaluating PACoST for {model_name}...")
        model, tokenizer = load_model(model_name)

        logger_result.info("Using dataset AIME-2024 and its paraphrased version:")
        logger_detail.info("Using dataset AIME-2024 and its paraphrased version:")
        aime_stats  = pacost_aime_inference(
            model, tokenizer, ori_datasets["aime"], para_datasets["aime"], logger_result, logger_detail
        )
        logger_result.info("Using dataset MMLU and its paraphrased version:")
        logger_detail.info("Using dataset MMLU and its paraphrased version:")
        mmlu_stats  = pacost_mmlu_inference(
            model, tokenizer, ori_datasets["mmlu"], para_datasets["mmlu"], logger_result, logger_detail
        )

        logger_result.info(
            f"AIME-2024: Δ = {aime_stats['mean_delta']:.4e}, "
            f"t = {aime_stats['t_stat']:.4f}, p = {aime_stats['p_value']:.4e}"
        )
        logger_result.info(
            f"MMLU:      Δ = {mmlu_stats['mean_delta']:.4e}, "
            f"t = {mmlu_stats['t_stat']:.4f}, p = {mmlu_stats['p_value']:.4e}"
        )

        logger_detail.info(
            f"AIME-2024: Δ = {aime_stats['mean_delta']:.4e}, "
            f"t = {aime_stats['t_stat']:.4f}, p = {aime_stats['p_value']:.4e}"
        )
        logger_detail.info(
            f"MMLU:      Δ = {mmlu_stats['mean_delta']:.4e}, "
            f"t = {mmlu_stats['t_stat']:.4f}, p = {mmlu_stats['p_value']:.4e}"
        )

        results[model_name] = {
            "AIME-2024": aime_stats,
            "MMLU": mmlu_stats,
        }
        
    return results