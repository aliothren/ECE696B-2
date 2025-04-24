# Project 2 - Benchmark Contamination in LLMs

Yuxin Ren  _<yuxinr@arizona.edu>_  

## Introduction

Large Language Models (LLMs) have demonstrated impressive reasoning and problem-solving capabilities across a wide range of benchmarks. However, recent studies have raised concerns regarding the validity of benchmark evaluations, especially in open-source settings where training data is not strictly controlled. Specifically, LLMs may memorize benchmark questions if such data—or closely paraphrased variants—were inadvertently included in the training corpus. This phenomenon, known as **benchmark contamination**, undermines the credibility of reported performance and calls for principled detection methods.

In this project, we investigate the potential benchmark contamination in reasoning-capable open-source LLMs. We focus on two benchmarks:

- **AIME-2024**: A collection of 30 math competition problems requiring structured multi-step reasoning.
- **MMLU (High School Mathematics)**: A 100-question subset from the Massive Multitask Language Understanding benchmark, featuring multiple-choice problems.

We evaluate three models from the DeepSeek-R1 Distilled Qwen family—**1.5B**, **7B**, and **14B** parameters. Our study consists of two major components:

1. **Benchmark Performance Assessment**: We run each model on the original benchmark questions and report accuracy statistics over multiple runs.
2. **Contamination Detection**: We implement and extend two recent contamination detection techniques—**TS-Guess** and **PaCoST**—to assess whether the models exhibit memorization behaviors when queried with original versus paraphrased or masked benchmark inputs.

Through carefully controlled experiments and prompt engineering, we aim to quantify the extent to which benchmark contamination affects model outputs, and to understand how model scale and prompt structure influence these effects.


## 2. Related Work

### 2.1 Benchmarks and Models

In this project, we focus on evaluating potential memorization behavior in open-source LLMs using two reasoning-heavy benchmark datasets and a family of distilled instruction-tuned models:

#### 2.1.1 Benchmarks:

- **AIME-2024**  
  Source: [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)  
  The AIME (American Invitational Mathematics Examination) dataset contains 30 open-ended math problems from the 2024 competition. These problems require multi-step symbolic reasoning and often include equations, variables, and logical deductions. Each problem has a unique short numerical answer, padded to three digits (e.g., `"073"`).

- **MMLU: High School Mathematics**  
  Source: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)  
  The MMLU benchmark includes a wide variety of subjects; we select the *High School Mathematics* subset (100 multiple-choice questions). Each question has four answer choices and a correct answer labeled by index (`0`–`3`). This dataset is widely used to test factual recall and problem-solving skills.

#### 2.1.2 Models:

In this project, we evaluate three instruction-tuned models released by DeepSeek-AI:

- [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)


These models are part of the **DeepSeek R1** release, which is a fully open-source large language model suite trained by [DeepSeek-AI](https://deepseek.com/). The base architecture is derived from **Qwen**, a decoder-only Transformer developed by Alibaba, using multi-stage training that includes pretraining on 2T tokens followed by supervised fine-tuning and reward modeling.

The “Distill” versions are produced through **online distillation**, where the model learns to imitate a larger teacher during the instruction-tuning stage, resulting in smaller and faster models that maintain reasonable performance.

The **DeepSeek-R1 Distill-Qwen** models have the following key properties:

- **Architecture**: All models follow the Qwen decoder-only design, incorporating features such as grouped-query attention, SwiGLU activation, and rotary position embedding. The vocabulary size exceeds 150,000 tokens, using a SentencePiece tokenizer.

- **Training Pipeline**:
  - **Pretraining**: Web-scale corpus of diverse domains (code, math, natural language, etc.)
  - **Instruction fine-tuning**: Based on multiple public instruction datasets including UltraChat, Baize, and Evol-Instruct.
  - **Distillation**: Conducted during supervised fine-tuning from larger Qwen models.

- **Extremely strong thinking-mode**: These models show a *strong inclination to enter "thinking mode"*, often attempting to produce step-by-step reasoning or justification even when the prompt simply requires a short answer. This behavior, while valuable in real-world use cases, introduces complications for contamination detection, particularly in tasks where we rely on models directly emitting final answers without reasoning chains.


By selecting these models, we ensure consistency across scale levels and minimize differences in training objectives, isolating memorization behavior related to parameter size and dataset exposure.

---

## 2.2 Contamination Detection Methods

Recent studies have raised increasing concerns that large language models (LLMs) may memorize and regurgitate benchmark data seen during pretraining. This phenomenon, known as **benchmark contamination**, undermines fair evaluation and poses challenges for both scientific and practical reliability. To address this, a number of methods have been proposed to detect and quantify contamination. These approaches can generally be grouped into three categories:

- **Dataset Scanning**:

These methods directly examine whether benchmark data appears in the model’s pretraining corpus. These approaches typically fall into two subtypes: **exact matching**, which detects verbatim overlaps using string comparison; and **fuzzy matching**, which applies similarity metrics such as MinHash, SimHash, or embedding-based methods like BERTScore to identify paraphrased or near-duplicate entries. While effective in identifying contamination when full access to the training data is available, these methods are fundamentally limited by their dependence on having access to the pretraining corpus—something that is typically unavailable for proprietary models. 


- **Behavioral Probing**:

Behavioral probing measures the model’s **likelihood or confidence behavior** to infer contamination. Two major subtypes include **Token-level probing** (observe if the model can predict the correct answer token when only given the problem statement, often using masked or truncated input) and **Confidence probing under paraphrase** (compare the probability of the correct answer between the original and paraphrased input.) These methods do not require access to training data and are particularly useful for closed-source models.

- **Semantic Perturbation**:

These methods involve modifying the phrasing of benchmark inputs while keeping their meaning unchanged. If the model’s performance degrades significantly on the paraphrased version, it suggests memorization of surface form rather than true generalization.Semantic perturbation can be used in conjunction with behavioral probing to detect **form-sensitive memorization**.

---
### 2.2.1 TS-Guessing

TS-Guessing ([Testset Slot Guessing](https://arxiv.org/abs/2311.09783)), proposed by Deng et al. (2023), investigates contamination by probing whether a model can correctly guess masked elements in benchmark questions, under the assumption that such information should not be inferable without contamination.

The method is evaluated under two complementary settings:

1. **Question-based guessing**: a keyword in the question is masked (e.g., *“Where did [MASK] cookies originate?”*). The model is asked to predict the missing word from an open vocabulary.
2. **Question-multichoice guessing**: one *incorrect* option in a multiple-choice question is masked, while the correct option and two other distractors remain intact. The model is then prompted to fill in the missing option. This design prevents the model from trivially solving the question through reasoning.

These protocols suppress the model's ability to leverage inference and instead isolate memorization behaviors. Deng et al. (2023) show that commercial models like ChatGPT and GPT-4 achieve **surprisingly high exact match (EM) rates** — e.g., **57% EM** in MMLU — despite no reasoning context, suggesting benchmark overlap during training. In contrast, newer benchmarks like GSM8K-v2 or smaller models such as LLaMA 2-7B exhibit significantly lower TS-Guess scores, supporting the method’s validity in distinguishing contaminated from clean settings.

---

### 2.2.2 PaCoST

PaCoST ([Paired Confidence Significance Testing](https://arxiv.org/abs/2406.18326)), introduced by Zhang et al. (2024), proposes a statistical approach to benchmark contamination detection that compares the model’s confidence on original benchmark items versus paraphrased counterparts.

The method follows three steps:

1. **Rephrasing**: For each test instance \((x, y)\), a paraphrased instruction \(x'\) is generated using a separate LLM. The answer \(y\) remains unchanged.
2. **Confidence Estimation**: The target LLM is queried with both \((x, y)\) and \((x', y)\), and its confidence is measured using the **P(True)** method (Kadavath et al., 2022), where the model is asked whether its own output is correct.
3. **Significance Testing**: A **paired sample t-test** is conducted on confidence scores from original vs. paraphrased instances. A statistically significant difference (e.g., \(p < 0.05\)) suggests contamination, as the model exhibits higher confidence on phrasing it has likely seen during training.

PaCoST satisfies several desirable properties: it is threshold-free, does not require access to training data, and is robust to contamination type and prompt format. In controlled experiments, the method successfully identified intentional contamination, and in real-world evaluations, it detected strong signs of contamination across multiple benchmarks (e.g., MMLU, TruthfulQA) and LLMs (e.g., LLaMA-2, DeepSeek, Mistral).

---

These two approaches allow us to probe memorization without requiring access to pretraining corpora. They complement each other: TS-Guess is sensitive to direct recall, while PaCoST detects more subtle statistical biases.


## 3. Methodology and Experiment Setup

### 3.1 Benchmark Evaluation

To evaluate the baseline performance of models on uncontaminated inputs, we begin by benchmarking their accuracy on selected datasets in a zero-shot multiple-choice setting. This provides reference points to help interpret whether strong answer alignment in TS-Guessing or PaCoST can be attributed to contamination.

All models were loaded using the HuggingFace `transformers` API with automatic device mapping (`device_map="auto"`) and quantized to reduce memory usage when applicable.

#### Dataset setups

We select two datasets for evaluation:

- **AIME-2024**: The AMC Intermediate Mathematics Exam (2024 version), accessed via HuggingFace ([`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)), using the full training set.
- **MMLU High School Mathematics**: A subset of the [MMLU benchmark](https://huggingface.co/datasets/cais/mmlu) filtered to the `"high_school_mathematics"` category.

The datasets are stored in `.json` format, with each entry represented as a dictionary containing the following fields:

For AIME-2024:
```json
{
    "id": 61,
    "problem": "Let $ABC$ be a triangle inscribed in circle $\\omega$. Let the tangents to $\\omega$ at $B$ and $C$ intersect at point $D$, and let $\\overline{AD}$ intersect $\\omega$ at $P$. If $AB=5$, $BC=9$, and $AC=10$, $AP$ can be written as the form $\\frac{m}{n}$, where $m$ and $n$ are relatively prime integers. Find $m + n$.",
    "solution": "From the tangency condition we have $\\let\\angle BCD = \\let\\angle CBD = \\let\\angle A$. With LoC we have $\\cos(A) = \\frac{25+100-81}{2*5*10} = \\frac{11}{25}$ and $\\cos(B) = \\frac{81+25-100}{2*9*5} = \\frac{1}{15}$. Then, $CD = \\frac{\\frac{9}{2}}{\\cos(A)} = \\frac{225}{22}$. Using LoC we can find $AD$: $AD^2 = AC^2 + CD^2 - 2(AC)(CD)\\cos(A+C) = 10^2+(\\frac{225}{22})^2 + 2(10)\\frac{225}{22}\\cos(B) = 100 + \\frac{225^2}{22^2} + 2(10)\\frac{225}{22}*\\frac{1}{15} = \\frac{5^4*13^2}{484}$. Thus, $AD = \\frac{5^2*13}{22}$. By Power of a Point, $DP*AD = CD^2$ so $DP*\\frac{5^2*13}{22} = (\\frac{225}{22})^2$ which gives $DP = \\frac{5^2*9^2}{13*22}$. Finally, we have $AP = AD - DP = \\frac{5^2*13}{22} - \\frac{5^2*9^2}{13*22} = \\frac{100}{13} \\rightarrow \\boxed{113}$.\n~angie.\nWe know $AP$ is the symmedian, which implies $\\triangle{ABP}\\sim \\triangle{AMC}$ where $M$ is the midpoint of $BC$. By Appolonius theorem, $AM=\\frac{13}{2}$. Thus, we have $\\frac{AP}{AC}=\\frac{AB}{AM}, AP=\\frac{100}{13}\\implies \\boxed{113}$\n~Bluesoul\nExtend sides $\\overline{AB}$ and $\\overline{AC}$ to points $E$ and $F$, respectively, such that $B$ and $C$ are the feet of the altitudes in $\\triangle AEF$. Denote the feet of the altitude from $A$ to $\\overline{EF}$ as $X$, and let $H$ denote the orthocenter of $\\triangle AEF$. Call $M$ the midpoint of segment $\\overline{EF}$. By the Three Tangents Lemma, we have that $MB$ and $MC$ are both tangents to $(ABC)$ $\\implies$ $M = D$, and since $M$ is the midpoint of $\\overline{EF}$, $MF = MB$. Additionally, by angle chasing, we get that: \n\\[\\angle ABC \\cong \\angle AHC \\cong \\angle EHX\\]\nAlso, \n\\[\\angle EHX = 90 ^\\circ - \\angle HEF = 90 ^\\circ - (90 ^\\circ - \\angle AFE) = \\angle AFE\\] \nFurthermore, \n\\[AB = AF \\cdot \\cos(A)\\]\nFrom this, we see that $\\triangle ABC \\sim \\triangle AFE$ with a scale factor of $\\cos(A)$. By the Law of Cosines, \n\\[\\cos(A) = \\frac{10^2 + 5^2 - 9^2}{2 \\cdot 10 \\cdot 5} = \\frac{11}{25}\\] \nThus, we can find that the side lengths of $\\triangle AEF$ are $\\frac{250}{11}, \\frac{125}{11}, \\frac{225}{11}$. Then, by Stewart's theorem, $AM = \\frac{13 \\cdot 25}{22}$. By Power of a Point, \n\\[\\overline{MB} \\cdot \\overline{MB} = \\overline{MA} \\cdot \\overline{MP}\\]\n\\[\\frac{225}{22} \\cdot \\frac{225}{22} = \\overline{MP} \\cdot \\frac{13 \\cdot 25}{22} \\implies \\overline{MP} = \\frac{225 \\cdot 9}{22 \\cdot 13}\\]\nThus, \n\\[AP = AM - MP = \\frac{13 \\cdot 25}{22} - \\frac{225 \\cdot 9}{22 \\cdot 13} = \\frac{100}{13}\\]\nTherefore, the answer is $\\boxed{113}$.\n~mathwiz_1207\nConnect lines $\\overline{PB}$ and $\\overline{PC}$. From the angle by tanget formula, we have $\\angle PBD = \\angle DAB$. Therefore by AA similarity, $\\triangle PBD \\sim \\triangle BAD$. Let $\\overline{BP} = x$. Using ratios, we have \\[\\frac{x}{5}=\\frac{BD}{AD}.\\] Similarly, using angle by tangent, we have $\\angle PCD = \\angle DAC$, and by AA similarity, $\\triangle CPD \\sim \\triangle ACD$. By ratios, we have \\[\\frac{PC}{10}=\\frac{CD}{AD}.\\] However, because $\\overline{BD}=\\overline{CD}$, we have \\[\\frac{x}{5}=\\frac{PC}{10},\\] so $\\overline{PC}=2x.$ Now using Law of Cosines on $\\angle BAC$ in triangle $\\triangle ABC$, we have \\[9^2=5^2+10^2-100\\cos(\\angle BAC).\\] Solving, we find $\\cos(\\angle BAC)=\\frac{11}{25}$. Now we can solve for $x$. Using Law of Cosines on $\\triangle BPC,$ we have \n\\begin{align*}\n81&=x^2+4x^2-4x^2\\cos(180-\\angle BAC) \\\\ \n&= 5x^2+4x^2\\cos(BAC). \\\\\n\\end{align*}\nSolving, we get $x=\\frac{45}{13}.$ Now we have a system of equations using Law of Cosines on $\\triangle BPA$ and $\\triangle CPA$, \\[AP^2=5^2+\\left(\\frac{45}{13}\\right)^2 -(10) \\left(\\frac{45}{13} \\right)\\cos(ABP)\\]\n\\[AP^2=10^2+4 \\left(\\frac{45}{13} \\right)^2 + (40) \\left(\\frac{45}{13} \\right)\\cos(ABP).\\]\nSolving, we find $\\overline{AP}=\\frac{100}{13}$, so our desired answer is $100+13=\\boxed{113}$.\n~evanhliu2009\nFollowing from the law of cosines, we can easily get $\\cos A = \\frac{11}{25}$, $\\cos B = \\frac{1}{15}$, $\\cos C = \\frac{13}{15}$.\nHence, $\\sin A = \\frac{6 \\sqrt{14}}{25}$, $\\cos 2C = \\frac{113}{225}$, $\\sin 2C = \\frac{52 \\sqrt{14}}{225}$.\nThus, $\\cos \\left( A + 2C \\right) = - \\frac{5}{9}$.\nDenote by $R$ the circumradius of $\\triangle ABC$.\nIn $\\triangle ABC$, following from the law of sines, we have $R = \\frac{BC}{2 \\sin A} = \\frac{75}{4 \\sqrt{14}}$.\nBecause $BD$ and $CD$ are tangents to the circumcircle $ABC$, $\\triangle OBD \\cong \\triangle OCD$ and $\\angle OBD = 90^\\circ$.\nThus, $OD = \\frac{OB}{\\cos \\angle BOD} = \\frac{R}{\\cos A}$.\nIn $\\triangle AOD$, we have $OA = R$ and $\\angle AOD = \\angle BOD + \\angle AOB = A + 2C$.\nThus, following from the law of cosines, we have\n\\begin{align*}\nAD & = \\sqrt{OA^2 + OD^2 - 2 OA \\cdot OD \\cos \\angle AOD} \\\\\n& = \\frac{26 \\sqrt{14}}{33} R.\n\\end{align*}\n\nFollowing from the law of cosines,\n\\begin{align*}\n\\cos \\angle OAD & = \\frac{AD^2 + OA^2 - OD^2}{2 AD \\cdot OA} \\\\\n& = \\frac{8 \\sqrt{14}}{39} .\n\\end{align*}\n\nTherefore,\n\\begin{align*}\nAP & = 2 OA \\cos \\angle OAD \\\\\n& = \\frac{100}{13} .\n\\end{align*}\n\nTherefore, the answer is $100 + 13 = \\boxed{\\textbf{(113) }}$.\n~Steven Chen (Professor Chen Education Palace, www.professorchenedu.com)",
    "answer": "113",
    "url": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems/Problem_10",
    "year": "2024"
  },
```

or for MMLU:

```json
  {
    "question": "At breakfast, lunch, and dinner, Joe randomly chooses with equal probabilities either an apple, an orange, or a banana to eat. On a given day, what is the probability that Joe will eat at least two different kinds of fruit?",
    "subject": "high_school_mathematics",
    "choices": [
      "\\frac{7}{9}",
      "\\frac{8}{9}",
      "\\frac{5}{9}",
      "\\frac{9}{11}"
    ],
    "answer": 1
  },
```

We primarily use the following fields:

- **AIME**:
  - `"problem"`: Used as the main question prompt.
  - `"answer"`: A three-digit string representing the correct answer (e.g., `"073"`). This is the target token for TS-Guess and PaCoST scoring.

- **MMLU**:
  - `"question"`: The textual problem description.
  - `"choices"`: A list of 4 options.
  - `"answer"`: An integer 0–3 indicating the correct choice, which we map to A–D during inference.

These fields are parsed and structured into model-ready prompts, as described in the benchmarking procedure. The dataset processing includes the following setups:

- To ensure consistency across contamination detection experiments, we further filter the MMLU dataset by selecting only the first 100 samples **whose correct answer is not option 'A'**. This subset is saved as `dataset/mmlu_high_school_math_100.json` and reused in both TS-Guess and PaCoST evaluations. This step simplifies TS-Guess analysis by avoiding hardcoded option ordering.

#### Prompt Design

Prompts were carefully crafted to simulate exam conditions and elicit structured reasoning. We primarily use the following prompt to do the benchmark tests:

For AIME-2024:

```
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.

Please put your final answer in the form \boxed{ }.
<think>
```

Or for MMLU:

```
Question: The length of a rectangle is twice its width. Given the length of the diagonal is $5\\sqrt{5}$, find the area of the rectangle.

Choices:
A. 2500
B. 2
C. 50
D. 25

Please put your final answer in the form \boxed{ }.
<think>
```

This format ensures that the model understands the task and produces well-formatted responses by following designs:

- **Explicitly enforce reasoning mode**: we observe that the **DeepSeek R1 Distill Qwen series models have a strong bias toward entering a reasoning phase**, regardless of how strict the prompt instructions are. Attempts such as "Do not explain" or "Only answer" fail to suppress the thought process. As a second choice, we use the `<think>` tag to force each generation into thinking mode and make it consistent across all runs.

- **Boxed answer extraction**: to facilitate reliable answer extraction from free-form generations, we instruct the model to place its final answer inside a LaTeX-style box, e.g., `\boxed{C}` or `\boxed{204}`. This convention is common in math and reasoning benchmarks, and helps distinguish the final answer from intermediate steps, formulas, or distractions in verbose outputs. The use of `\boxed{}` simplifies post-processing and evaluation by enabling a consistent extraction rule, even when the generation includes lengthy explanations.


#### Inference Procedure

To enable stable long-form outputs from the model, we use a custom function `generate_long_response` that repeatedly generates short chunks until termination criteria are met. Parameters are set as:

- `chunk_size = 512`
- `max_rounds = 15`
- `temperature = 0.5`
- `top_k = 50`
- `top_p = 0.9`

This prevents infinite generation loops while allowing sufficient reasoning space for the model to reach a conclusion. We observe that:

- If the model is able to solve the problem, it usually finishes within 10–12 rounds.
- If the model fails or enters invalid mode collapse, the capped length prevents resource waste.

This setting balances completeness and efficiency.

#### Output Handling

The output string is processed by a utility function `extract_clean_answer` to extract the final answer in the form of a letter. The predicted answer is then compared with the correct one, which is converted from numeric label (0-3) to letter (A-D) as:

```python
correct_answer = chr(65 + int(sample["answer"]))
```

This ensures that the string comparison is compatible with the model output format.


#### Summary

This benchmark framework provides the reference performance for each model on clean inputs. Later contamination detection scores (e.g., from TS-Guess or PaCoST) are interpreted in light of these baseline capabilities.



### 3.2 TS-Guessing Evaluation

In this section, we implement a modified version of the **Target String Guessing (TS-Guess)** method. The following methods are applied, based on benchmark setup.

#### Method Modification

In the original paper, TS-Guess is applied across a variety of datasets and settings, and its results are evaluated relative to multiple baselines. However, due to limited compute availability, we simplify the method for practical deployment:

- Based on results form the paper, we hypothesize that for clean datasets, models should not be able to guess the masked answer with any notable accuracy, leading to a score near 0.
- In contrast, if a model demonstrates significantly above-zero accuracy on masked examples, we interpret this as evidence of contamination.
- We therefore run the test on one dataset at a time and compare results across models.

#### Dataset Setups

For the **AIME** dataset, we used GPT-4o to help identify the most semantically central word or phrase in each question, and all replacements are manually verified. The selected word is replaced with a `[MASK]` token, and the original word is stored in the `"keyword"` field for evaluation.

For the **MMLU** dataset, we have extracted the first 100 examples from the "high_school_mathematics" subset where the correct answer is not option A in the **Benchmark** section. Here we replace the **first option** (which is originally incorrect) with `[MASK]`. The original content of option A is saved to the `"keyword"` field.

Example (AIME):
```json
{
  "id": 61,
  "problem": "Let $ABC$ be a [MASK] inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$, and let $\overline{AD}$ intersect $\omega$ at $P$. If $AB=5$, $BC=9$, and $AC=10$, $AP$ can be written as the form $\frac{m}{n}$, where $m$ and $n$ are relatively prime integers.",
  ...
  "answer": "113",
  "keyword": "triangle"
}
```

Example (MMLU):
```json
{
  "question": "John divided his souvenir hat pins into two piles...",
  "choices": ["[MASK]", "72", "66", "36"],
  "answer": 1,
  "keyword": "396"
}
```

#### Prompt Design

To prevent the model from reasoning during inference, we use instructions explicitly discouraging thought. Furthermore, we truncate the question for AIME to its declarative core, eliminating any concluding instructions like "find X".

We further simplify evaluation:
- Only one short output is generated.
- If the model does not output the correct token in the expected format (e.g., `\boxed{triangle}`), we consider it a failed memory recall.

**Prompt (AIME):**
```text
You are a language model. Do not explain. Do not reason.
Sentence: 'Let $ABC$ be a [MASK] inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$...'
Replace [MASK] with exactly one word. Only output the one word in \boxed{ }.
```

**Prompt (MMLU):**
```text
You are a language model. Do not explain. Do not reason.
Complete the [MASK] in the first option of the choices based on your benchmark knowledge.

If $f(x) = 8x^3 - 6x^2 - 4x + 5$, find the value of $f( -2)$.
Choices:
** [MASK] **
** -75 **
** -64 **
** -24 **

Reply with answer only. Only put your answer in the form \boxed{ }.
```

### Evaluation

A correct prediction is determined by matching the model's generated boxed response with the saved `keyword`. Overall accuracy is reported across 100 examples per dataset.

### 3.3 PaCoST Evaluation