import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer


def extract_clean_answer(text):
    pattern = r'\\boxed\s*{((?:[^{}]|{[^{}]*})*)}'
    matches = re.findall(pattern, text)
    
    if not matches:
        return ""
    content = matches[-1]

    while True:
        new_content = re.sub(r'\\(?:text|mathrm|mathit|textbf)\s*{([^{}]*)}', r'\1', content)
        if new_content == content:
            break
        content = new_content

    return content.strip()


def generate_long_response(
    model, tokenizer, prompt, logger_detail,
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
