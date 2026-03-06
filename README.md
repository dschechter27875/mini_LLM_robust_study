# mini_LLM_robust_study
Fine-tuning FLAN-T5-small on BoolQ and evaluating robustness under paraphrases, typos, and distractors. Includes reproducible training and evaluation pipeline.
# FLAN-T5 Robustness Study on BoolQ

## Overview

This project explores how small instruction-tuned language models behave when fine-tuned on binary question answering tasks.

I fine-tuned **FLAN-T5-small** on the **BoolQ dataset** and evaluated the model under several input perturbations, including typos, paraphrases, and distracting context.

The goal was to understand how well a small transformer adapts to downstream tasks and to identify potential **failure modes during fine-tuning and evaluation**.

---

## Dataset

Experiments use the **BoolQ dataset**, which consists of natural yes/no questions paired with Wikipedia passages.

Example input format:

```
question: Did ethanol production require more energy than it produced?
passage: Ethanol fuel production from biomass involves several energy-intensive steps...
answer yes or no
```

Target output:

```
yes
```

or

```
no
```

The BoolQ validation set is **imbalanced**, with approximately:

| Label | Fraction |
| ----- | -------- |
| Yes   | ~62%     |
| No    | ~38%     |

This imbalance plays an important role when interpreting evaluation results.

---

## Model

Fine-tuned model:

| Property     | Value                          |
| ------------ | ------------------------------ |
| Model        | FLAN-T5-small                  |
| Parameters   | ~80M                           |
| Architecture | Encoder-Decoder Transformer    |
| Pretraining  | Instruction-tuned FLAN mixture |

Training configuration:

| Parameter           | Value                  |
| ------------------- | ---------------------- |
| Train examples      | 6000 (balanced yes/no) |
| Validation examples | 1000                   |
| Learning rate       | 5e-5                   |
| Batch size          | 16                     |
| Epochs              | 2                      |

Training was implemented using **HuggingFace Transformers**.

---

## Robustness Experiments

To evaluate robustness, several perturbed versions of the dataset were created.

| Dataset    | Description                                   |
| ---------- | --------------------------------------------- |
| Clean      | Original BoolQ examples                       |
| Typos      | Character-level noise added to questions      |
| Distractor | Irrelevant sentences appended to passages     |
| Paraphrase | Questions rewritten with alternative phrasing |

These perturbations simulate the kinds of noisy inputs encountered by real-world NLP systems.

---

## Results

Balanced evaluation results (example subset):

| Model            | Accuracy | Yes Predictions | No Predictions |
| ---------------- | -------- | --------------- | -------------- |
| Base FLAN-T5     | ~0.55    | Mixed           | Mixed          |
| Fine-tuned model | ~0.50    | All Yes         | 0              |

---

## Key Findings

### 1. Majority-Class Collapse Can Occur During Fine-Tuning

During training, the model sometimes converged to a degenerate strategy of always predicting **"yes"**, the majority label in BoolQ.

Because the dataset contains ~62% "yes" answers, this trivial strategy achieves:

```
Accuracy ≈ 0.62
```

Balanced evaluation revealed the true behavior: the fine-tuned model collapsed to ~50% accuracy and predicted only the majority class.

---

### 2. Dataset Imbalance Can Mask Model Failures

BoolQ’s natural label imbalance allows trivial classifiers to achieve seemingly strong accuracy.

Evaluating on **balanced subsets** made the model’s behavior much clearer and exposed failure modes that were not visible using the original validation distribution.

---

### 3. Generative Decoding Can Be Misleading for Binary Tasks

When evaluating with standard text generation (`model.generate()`), model outputs appeared reasonable.

However, likelihood-based scoring between candidate answers ("yes" vs "no") revealed that the model strongly preferred a single label.

This suggests that **generative evaluation can obscure degenerate decision boundaries**, especially in small models.

---

### 4. Small Instruction-Tuned Models Are Sensitive to Fine-Tuning Setup

Experiments showed that training dynamics for small encoder-decoder models can be unstable when adapting to binary QA tasks.

Factors such as:

* dataset imbalance
* learning rate
* output tokenization
* evaluation strategy

can significantly influence the final model behavior.

---

### 5. Robustness Experiments Require Careful Evaluation Design

Adding perturbations such as typos, paraphrases, and distracting context helps simulate real-world inputs.

However, robustness metrics can be misleading if the underlying dataset contains structural biases. Careful dataset construction and analysis are necessary for meaningful robustness evaluation.

---

## Repository Structure

```
src/
  train_llm.py
  eval_robustness.py
  make_perturbations.py

data/
  perturbed/

runs/
  model checkpoints
```

---

## Technologies Used

* **Python** – experiment scripting and data processing
* **PyTorch** – neural network training framework
* **HuggingFace Transformers** – loading and fine-tuning FLAN-T5
* **HuggingFace Datasets** – BoolQ dataset handling
* **SentencePiece Tokenization** – tokenization used by the T5 architecture
* **Google Colab GPU (Tesla T4)** – model training and experimentation
* **JSON / Python data pipelines** – generation of perturbed evaluation datasets

---

## Future Work

Possible extensions include:

* training larger models such as **FLAN-T5-base**
* using classification heads instead of generative decoding
* building stronger adversarial perturbations
* exploring retrieval-augmented question answering systems






