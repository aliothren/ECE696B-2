import numpy as np
from model_generation import load_model, generate_long_response, extract_clean_answer


def bench_aime_inference(model, tokenizer, dataset, logger_result, logger_detail, num_runs=5):
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
                logger_detail=logger_detail,
                chunk_size=512,
                max_rounds=15,
                temperature=0.5,
                top_k=50,
                top_p=0.9,
            )
            
            predicted_answer = extract_clean_answer(full_output)
            logger_result.info(f"Extract answer: {predicted_answer}")
            logger_detail.info(f"Extract answer: {predicted_answer}")
            correct_answer = str(sample["answer"]).strip()
            logger_result.info(f"Correct answer: {correct_answer}")
            logger_detail.info(f"Correct answer: {correct_answer}")
            if predicted_answer.lower().zfill(3) == correct_answer.lower():
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


def bench_mmlu_inference(model, tokenizer, dataset, logger_result, logger_detail, num_runs=5):
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
                logger_detail=logger_detail,
                chunk_size=512,
                max_rounds=15,
                temperature=0.5,
                top_k=50,
                top_p=0.9,
            )

            predicted_answer = extract_clean_answer(full_output)
            logger_result.info(f"Extract answer: {predicted_answer}")
            logger_detail.info(f"Extract answer: {predicted_answer}")
            correct_answer = str(sample["answer"]).strip()
            logger_result.info(f"Correct answer: {correct_answer}")
            logger_detail.info(f"Correct answer: {correct_answer}")
            correct_answer = chr(65 + int(correct_answer))
            if predicted_answer.lower() == correct_answer.lower():
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


def benchmark(models, datasets, logger_result, logger_detail):
    results = {}
    for model_name in models:
        logger_result.info("========================")
        logger_detail.info("========================")
        logger_result.info(f"Evaluating {model_name}...")
        logger_detail.info(f"Evaluating {model_name}...")
        model, tokenizer = load_model(model_name)
        
        logger_result.info("Using dataset AIME-2024:")
        logger_detail.info("Using dataset AIME-2024:")
        aime_mean, aime_std = bench_aime_inference(model, tokenizer, datasets["aime"], logger_result, logger_detail)
        logger_result.info("Using dataset MMLU:")
        logger_detail.info("Using dataset MMLU:")
        mmlu_mean, mmlu_std = bench_mmlu_inference(model, tokenizer, datasets["mmlu"], logger_result, logger_detail)

        results[model_name] = {
            "AIME-2024": {"mean": aime_mean, "std": aime_std},
            "MMLU": {"mean": mmlu_mean, "std": mmlu_std},
        }

        logger_result.info(f"AIME-2024: {aime_mean:.3f} ± {aime_std:.3f}")
        logger_detail.info(f"AIME-2024: {aime_mean:.3f} ± {aime_std:.3f}")
        logger_result.info(f"MMLU:      {mmlu_mean:.3f} ± {mmlu_std:.3f}")
        logger_detail.info(f"MMLU:      {mmlu_mean:.3f} ± {mmlu_std:.3f}")
    return results

