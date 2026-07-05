# 06 — IMDB Sentiment Analysis — BERT Fine-tuning

Fine-tuning a pre-trained BERT model for binary sentiment classification — the final project in the `deep-learning-foundations` series. This completes the full progression from building attention from scratch (`05`) to leveraging weights pre-trained on 3.3 billion words.

## Results

| Metric | Value |
|---|---|
| BERT Test Accuracy | **86.90%** |
| BERT Test Loss | 0.4301 |
| Training samples | 5,000 (CPU subset) |
| Transformer Baseline (`05`) | 87.57% |
| LSTM Baseline (`04`) | 85.03% |

> BERT was trained on only 1K samples due to CPU constraints. Pre-trained weights give BERT a strong starting point, but fine-tuning still needs sufficient task-specific data to adapt. Full dataset training (25K samples) is expected to push accuracy above 92%.

## Full Series Comparison

| # | Model | Accuracy | Parameters | Key Concept |
|---|---|---|---|---|
| 01 | ANN | 97.86% | ~109K | Fully connected, no spatial awareness |
| 02 | CNN | 99.14% | ~93K | Spatial feature detection via filters |
| 03 | RNN | — | — | Sequential hidden state |
| 04 | LSTM | 85.03% | ~1.4M | Gated memory for long sequences |
| 05 | Transformer | 87.57% | ~300K | Parallel self-attention from scratch |
| 06 | BERT | **86.90%\*** | 110M | Transfer learning from pre-trained weights |

*\*CPU subset — full dataset expected 92%+*

## What Fine-tuning Means

BERT was pre-trained on BookCorpus + Wikipedia using masked language modeling — it already understands English grammar, semantics, and context. Fine-tuning adds a classification head on top and trains for 2-3 epochs with a very small learning rate (`2e-5`), adapting the weights to the specific task without forgetting what was learned during pre-training.

## How BERT Differs from the Transformer in `05`

| `05` Transformer (from scratch) | `06` BERT (fine-tuned) |
|---|---|
| Random weight initialization | Pre-trained on 3.3B words |
| Trained on IMDB only | Already knows English |
| ~300K parameters | 110M parameters |
| 15 epochs needed | 3 epochs sufficient |
| 87.57% on full 25K | 86.90% on 5K subset |

## Key Implementation Details

- **`bert-base-uncased`** — 12 transformer layers, 12 attention heads, 768 hidden dimensions
- **WordPiece tokenization** — unknown words split into subword units, no true OOV words
- **`MAX_LEN=128`** — shorter than BERT's max 512 for CPU speed
- **`[CLS]` token** — BERT adds a special classification token at position 0; its final hidden state is used for classification
- **Learning rate `2e-5`** — fine-tuning requires a much smaller LR than training from scratch to avoid catastrophic forgetting

## Files

- `bert_sentiment.ipynb` — full notebook: tokenization, BERT loading, fine-tuning, evaluation, custom predictions, full series comparison
- `bert_sentiment_model/` — saved model + tokenizer (HuggingFace format)
- `training_curves.png` — loss and accuracy per epoch
- `confusion_matrix.png` — per-class prediction breakdown

## Stack

TensorFlow, Keras, HuggingFace Transformers (4.40.0), tf-keras, NumPy, Matplotlib, Seaborn, Scikit-learn

## The Complete Progression

ANN → ignores spatial structure → CNN fixes it with convolutions → RNN handles sequences but forgets → LSTM adds gated memory → Transformer replaces recurrence with parallel attention → BERT scales attention with pre-training on billions of words.

Each architecture addressed a specific limitation of the previous one.
