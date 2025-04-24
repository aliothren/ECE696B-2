# ECE696B Project 2: Benchmark Contamination Detection

This repository contains code and resources for ECE696B Project 2, focused on evaluating benchmark contamination in language models using techniques such as `TS-Guess` and `PaCoST`.

## Project Structure


### Main Scripts

- `report.md`: Experimental report.
- `main.py`: Entry point for evaluation. Runs benchmark scoring, TS-Guess, and PaCoST.
- `benchmark.py`: Logic for standard benchmark accuracy testing.
- `ts_guess.py`: Implements TS-Guessing method to estimate contamination by masking answers.
- `pacost.py`: Implements a modified PaCoST method using log-probabilities.
- `model_generation.py`: Utility for loading models and generating responses.
- `requirements.txt`: Python environment dependencies.


### Datasets
```
dataset/
├── aime_2024.json
├── aime_2024_mask_delete_question.json
├── aime_2024_paraphrase.json
├── mmlu_high_school_math_100.json
├── mmlu_mask.json
└── mmlu_paraphrased.json
```

These are the evaluation datasets:
- `aime_2024.json`: Original AIME 2024 benchmark.
- `aime_2024_mask_delete_question.json`: AIME with masked answers and reduced question text (used in TS-Guess).
- `aime_2024_paraphrase.json`: Paraphrased version of AIME (used in PaCoST).
- `mmlu_high_school_math_100.json`: Filtered MMLU dataset, totally 100 questions.
- `mmlu_mask.json`: MMLU with masked answers (used in TS-Guess).
- `mmlu_paraphrased.json`: Paraphrased version of MMLU (used in PaCoST).


### Outputs
```
logs/
└── <timestamped folders>/
    ├── detail.txt
    └── results.txt
```
Each folder contains output logs for an evaluation run.

## How to Run

Activate your virtual environment and run:

```bash
python main.py
```

Make sure to install dependencies first:

```bash
pip install -r requirements.txt
```

## Output

- Evaluation logs and scores are saved in `logs/YYYY-MM-DD_HH-MM-SS/`.
- TS-Guess logs answer probabilities from masked data.
- PaCoST logs the delta and p-values for contamination significance testing.

---

ECE696B - Project 2
