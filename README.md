# BART + LoRA Abstractive Summarization

Efficient fine-tuning of **BART-Large-CNN using LoRA (Low Rank Adaptation)** for abstractive text summarization on the CNN/DailyMail dataset.

The goal of this project is to explore **parameter-efficient fine-tuning for large transformer models** and evaluate whether LoRA can achieve competitive summarization performance while dramatically reducing training cost and compute requirements.

---

# Live Demo

Try the summarization model here:

**Hugging Face Space:**  
https://huggingface.co/spaces/kpkanth7/Bart-LoRa-Summarizer

The demo allows users to paste long news articles and generate **abstractive summaries using the fine-tuned LoRA BART model.**

---

# Project Goal

Modern transformer summarization models such as **BART-Large-CNN** achieve strong performance but require expensive full-model fine-tuning, which is often impractical due to:

- extremely high GPU memory requirements  
- long training times  
- large storage overhead  

This project investigates whether **LoRA (Low Rank Adaptation)** can provide an efficient alternative by:

- freezing the original BART model  
- training only small low-rank adapter matrices  
- drastically reducing trainable parameters  

Primary objectives:

- Evaluate how well LoRA performs compared to the baseline model
- Determine whether LoRA preserves summarization quality
- Assess if parameter-efficient fine-tuning is viable for real-world summarization pipelines

---

# Dataset

This project uses the **CNN/DailyMail summarization dataset**.

Dataset:

```
abisee/cnn_dailymail
config: 3.0.0
```

Dataset fields:

| Field | Description |
|------|-------------|
| article | Full news article |
| highlights | Human written summary |

Dataset splits:

- train
- validation
- test

Mapping used in the project:

```
article → input text
highlights → target summary
```

Key preprocessing decisions:

- Minimum article length filtering
- Minimum summary length filtering
- Token-length exploration
- Controlled truncation strategy

---

# Model Architecture

Base model:

```
facebook/bart-large-cnn
```

BART is a **sequence-to-sequence transformer architecture** designed for summarization and generation tasks.

Key characteristics:

- encoder-decoder transformer
- pretrained on large news datasets
- optimized for summarization tasks

---

# Why LoRA

Fine-tuning large language models is computationally expensive.

**LoRA (Low Rank Adaptation)** solves this by:

- freezing original model weights
- injecting low-rank matrices into attention layers
- training only lightweight adapter parameters

Advantages:

- drastically fewer trainable parameters
- lower GPU memory usage
- faster training
- easier deployment

---

# Tokenization Strategy

```
max_input_tokens = 1024
max_target_tokens = 128
padding = max_length
truncation = True
```

These values were selected to balance:

- information retention
- GPU memory limitations
- training stability

---

# Training Process

Training consisted of two stages.

## Step 1 — Baseline Evaluation

The original pretrained BART model was evaluated to establish baseline performance.

| Metric | Score |
|------|------|
| ROUGE-1 | 0.362 |
| ROUGE-2 | 0.164 |
| ROUGE-L | 0.271 |
| ROUGE-Lsum | 0.310 |

---

## Step 2 — LoRA Fine-Tuning

LoRA adapters were introduced into the BART architecture.

Training characteristics:

- adapter-based fine-tuning
- frozen base model weights
- limited training epochs
- parameter-efficient updates

Despite training only a small subset of parameters, the model successfully adapted to the summarization task.

---

# Training Cost and Compute Challenges

Even with LoRA, summarization models remain computationally demanding.

Observed during experimentation:

- GPU training sessions lasting multiple hours
- large sequence lengths increasing memory usage
- careful batch size tuning required

These findings reflect an important industry reality:

**Transformer summarization models remain computationally intensive even with optimization techniques like LoRA.**

---

# Evaluation Results

| Metric | Score |
|------|------|
| ROUGE-1 | 0.351 |
| ROUGE-2 | 0.153 |
| ROUGE-L | 0.254 |
| ROUGE-Lsum | 0.315 |

Evaluation details:

- Epochs: 3
- Evaluation runtime: ~717 seconds
- Samples per second: ~0.69

---

# Baseline vs LoRA Comparison

| Metric | Baseline | LoRA |
|------|------|------|
| ROUGE-1 | 0.362 | 0.351 |
| ROUGE-2 | 0.164 | 0.153 |
| ROUGE-L | 0.271 | 0.254 |
| ROUGE-Lsum | 0.310 | **0.315** |

---

# Key Observations

## Slight Metric Differences

LoRA shows slightly lower ROUGE-1 and ROUGE-2 compared to the baseline.

Possible reasons:

- limited training epochs
- restricted dataset sampling
- reduced parameter updates

## Improved Summary Structure

ROUGE-Lsum improved slightly, suggesting:

- better sentence-level coherence
- improved structural summarization

This indicates LoRA may encourage **more structured summaries rather than phrase copying.**

## Parameter Efficiency Tradeoff

| Full Fine-Tuning | LoRA |
|---|---|
| Higher compute cost | Lower compute cost |
| Full parameter updates | Minimal parameter updates |
| Slightly stronger metrics | Competitive metrics |

LoRA provides a **practical balance between performance and compute efficiency.**

---

# Inference

The model is deployed via **Hugging Face Spaces** allowing real-time summarization.

Users can:

- paste long articles
- generate summaries instantly
- experiment with different inputs

Demo:

https://huggingface.co/spaces/kpkanth7/Bart-LoRa-Summarizer

---

# What This Project Accomplishes

This project demonstrates:

- parameter-efficient fine-tuning of large summarization models
- competitive summarization performance with reduced compute
- a complete NLP pipeline including preprocessing, training, evaluation, and deployment

Pipeline components:

- dataset preprocessing
- transformer fine-tuning
- ROUGE evaluation
- Hugging Face deployment

---

# Limitations

## Limited GPU Compute

Stronger GPUs could enable:

- larger batch sizes
- longer training
- more stable optimization

## Short Training Duration

Training was limited to **3 epochs**, which likely constrained performance.

## Limited Dataset Sampling

Only a subset of the dataset was used due to compute limitations.

Training on the full dataset would likely improve results.

---

# Future Improvements

Potential extensions include:

## Longer Training

10–20 training epochs could improve adapter learning.

## Larger Batch Sizes

Would improve gradient stability.

## Instruction-Tuned Summarization

Instruction datasets could improve summary quality.

## Model Variants

Future experiments could explore:

- T5
- Longformer Encoder-Decoder
- PEGASUS

---

# Running Locally

Clone the repository:

```
git clone <repo-url>
```

Install dependencies:

```
pip install -r space/requirements.txt
```

Run the application:

```
python space/app.py
```

---

# Repository Structure

```
project_root
│
├── notebooks/
│
├── artifacts/
│   ├── evaluation_summary.json
│   ├── baseline_rouge_val_200.json
│   └── lora_eval_metrics_val_500.json
│
├── space/
│   ├── app.py
│   └── requirements.txt
│
├── README.md
```

---

# Conclusion

This project demonstrates an efficient workflow for **parameter-efficient fine-tuning of transformer summarization models using LoRA.**

Despite training only lightweight adapter layers, the model achieves **competitive ROUGE performance while significantly reducing computational requirements.**

The results highlight the growing importance of **parameter-efficient training techniques for scalable NLP development**, especially in environments with limited GPU resources.
