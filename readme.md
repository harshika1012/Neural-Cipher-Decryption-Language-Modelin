# Neural Cipher Decryption & Language Modeling

A PyTorch-based NLP project that combines sequence-to-sequence learning, language modeling, and context-aware error correction to decrypt substitution-style ciphers and reconstruct noisy English text.

## Overview

This project explores three interconnected NLP tasks:

1. **Neural Cipher Decryption:** Train custom RNN and LSTM encoder-decoder models with attention to translate encrypted text into English.
2. **Language Modeling:** Implement a State Space Model (SSM) for next-word prediction and a Bidirectional LSTM (Bi-LSTM) for masked language modeling.
3. **Noise-Robust Error Correction:** Integrate trained language models with the decryption pipeline to correct corrupted words and improve text reconstruction under different noise levels.

All neural architectures were implemented from scratch using PyTorch rather than built-in RNN/LSTM modules.

## Key Features

- Custom Seq2Seq encoder-decoder architectures with Bahdanau-style and Luong-style attention.
- Manual RNN, LSTM, and Bi-LSTM cell implementations.
- Causal next-word prediction using a custom State Space Model with learnable state matrices.
- Masked language modeling using bidirectional contextual representations.
- Context-aware error correction using language-model candidate generation, word-length constraints, and Levenshtein distance.
- Comparative evaluation across architectures and five noise levels.

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Models:** Seq2Seq, RNN, LSTM, Bi-LSTM, State Space Model
- **Techniques:** Attention, teacher forcing, mixed-precision training, gradient clipping, early stopping, Top-K sampling
- **Evaluation:** Character accuracy, word accuracy, perplexity, Levenshtein distance, BLEU, ROUGE-L

## Methodology

### 1. Neural Cipher Decryption

Built custom encoder-decoder models to learn the mapping from encrypted sequences to English text.

- **RNN:** Manual recurrent cells with additive Bahdanau-style attention.
- **LSTM:** Manual gated recurrent cells with Luong-style scaled dot-product attention.
- Applied teacher forcing, gradient clipping, and validation-based early stopping for training stability.

### 2. Language Modeling

Implemented and compared two approaches:

- **State Space Model (SSM):** Autoregressive next-word prediction using learnable state matrices.
- **Bi-LSTM:** Masked language modeling using bidirectional contextual information to predict masked words.

Both models used a vocabulary of up to 10,000 words and mixed-precision training.

### 3. Noise-Robust Error Correction

Developed a two-stage pipeline:

1. Decrypt noisy cipher text using the trained LSTM Seq2Seq model.
2. Detect invalid words and generate language-model candidates based on surrounding context.
3. Restrict candidates to the original word length and select corrections using minimum Levenshtein distance, with model confidence as a tie-breaker.
4. Evaluate the pipeline across noise levels 0–4 using text reconstruction and language-generation metrics.

## Experimental Results

### Cipher Decryption

| Metric | RNN | LSTM |
|---|---:|---:|
| Character Accuracy | 53.14% | 73.70% |
| Word Accuracy | 37.90% | 57.15% |
| Test Loss | 2.0852 | 1.1609 |
| Average Levenshtein Distance | 45.80 | 26.96 |

The LSTM outperformed the standard RNN across all reported metrics, demonstrating better sequence modeling for longer inputs.

### Language Modeling

| Model | Task | Test Perplexity |
|---|---|---:|
| Custom SSM | Next-Word Prediction | 226.16 |
| Custom Bi-LSTM | Masked Language Modeling | 127.79 |

The Bi-LSTM achieved lower perplexity on its masked language modeling task. Since the two models perform different prediction tasks, their perplexity values should not be interpreted as a direct like-for-like comparison.

### Noise-Robust Error Correction

At noise level 1:

- Baseline decryption achieved a BLEU score of **5.95**.
- Decryption with SSM-based correction achieved a BLEU score of **17.56**, nearly a **3× improvement**.
- Word accuracy increased from **29% to 39%** with SSM-based correction.

At noise level 4, the baseline BLEU score was 0.72, while SSM- and Bi-LSTM-based correction achieved 7.99 and 6.76, respectively.

These results demonstrate the value of language-model-based post-processing for recovering meaningful text from noisy decryption outputs.

## Training Configuration

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Seq2Seq Embedding Dimension | 128 |
| Seq2Seq Hidden Dimension | 256 |
| Language Model Embedding Dimension | 256 |
| Language Model Hidden/State Dimension | 256 |
| Language Model Dropout | 0.3 |
| Seq2Seq Training | Up to 20 epochs |
| Language Model Training | 10 epochs |
| Hardware | CUDA-enabled GPU |

## Repository Structure

```text
.
├── src/                  # Model implementations and pipeline code
├── outputs/
│   ├── logs/             # Training and validation logs
│   ├── plots/            # Evaluation plots
│   └── results/          # Generated predictions and results
├── README.md
└── Report.pdf
```

The exact directory structure may vary depending on the implementation. Model checkpoints and additional experiment outputs can be included where applicable.

## Setup

### Prerequisites

- Python 3.10 or compatible Python environment
- PyTorch
- CUDA-enabled GPU recommended for training

### Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install torch numpy
```

Install any additional packages required by the implementation.

### Running the Project

Run the corresponding training and evaluation scripts from the repository's source directory.

The workflow consists of:

1. Training and evaluating the RNN and LSTM decryption models.
2. Training and evaluating the SSM and Bi-LSTM language models.
3. Running the noise-robust correction pipeline.
4. Comparing the generated results across models and noise levels.

**Note:** Replace this section with the actual script names and commands from your repository before publishing the README.

## Evaluation Metrics

- **Character Accuracy:** Measures character-level reconstruction correctness.
- **Word Accuracy:** Measures word-level reconstruction correctness.
- **Perplexity:** Evaluates language-model prediction performance.
- **Levenshtein Distance:** Measures the edit distance between predicted and reference text.
- **BLEU:** Measures n-gram overlap with reference text.
- **ROUGE-L:** Measures longest-common-subsequence overlap.

## Key Takeaways

- LSTM gating and attention improved long-sequence decryption compared with a standard RNN.
- Bidirectional context improved masked-word prediction on clean text.
- Autoregressive SSM-based correction proved more robust than Bi-LSTM correction in the evaluated noisy-decryption pipeline.
- Language-model-based post-processing substantially improved text reconstruction under moderate and high noise.



Developed as part of the Introduction to NLP coursework, Spring 2026.
