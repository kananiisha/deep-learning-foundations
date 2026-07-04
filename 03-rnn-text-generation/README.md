# 03 — Shakespeare Text Generation — RNN

A character-level Recurrent Neural Network trained on the complete works of William Shakespeare. The model learns to predict the next character given a sequence of previous characters — the third project in the `deep-learning-foundations` series, introducing sequence modeling after ANN and CNN.

## Results

| Metric | Value |
|---|---|
| Final Training Loss | 3.1263 (5 epochs) |
| Final Training Accuracy | 23.01% |
| Vocabulary size | 79 unique characters |
| Sequence length | 100 characters |
| Architecture | Embedding(256) → LSTM(256, return_seq) → LSTM(128, return_seq) → Dense(vocab) |

> **Note on accuracy:** Random guessing over 79 characters = 1.3%. 23% means the model genuinely learned character-level patterns — letter clusters, word boundaries, punctuation placement. Shakespeare-quality output requires 30-50 epochs on the full dataset with GPU. The architecture and pipeline are production-correct; output quality scales with compute.

## How It Works

This is fundamentally different from ANN and CNN:

| Model | Input type | Memory |
|---|---|---|
| ANN | Fixed-size flat vector | None — each input independent |
| CNN | 2D spatial grid | Local only — filter window |
| RNN (LSTM) | Sequence of any length | Hidden state carries context across all steps |

At each timestep, the LSTM updates its hidden state based on the current character and the previous hidden state — giving it memory across the entire sequence.

## Files

- `rnn_text_generation.ipynb` — full notebook: data loading, vocabulary mapping, sequence creation, model training, text generation at multiple temperatures, training curves
- `rnn_shakespeare_ckpt.keras` — best checkpoint saved during training
- `rnn_shakespeare.keras` — final trained model
- `training_curves.png` — loss and accuracy per epoch
- `temperature_comparison.png` — generated text at temperatures 0.2 → 1.2

## Key Implementation Details

- **Character-level tokenization** — 79 unique characters as vocabulary, no word-level tokenization needed
- **Sliding window sequences** — input is 100 chars, target is the same 100 chars shifted by 1
- **`return_sequences=True`** on both LSTMs — outputs a prediction at every timestep, not just the final one
- **Temperature sampling** — instead of always picking the most likely character (`argmax`), samples from the probability distribution scaled by temperature. Low temperature = predictable, high temperature = creative
- **Model checkpointing** — saves best model during training based on loss

## Why LSTM over Simple RNN

Simple RNNs suffer from vanishing gradients — they forget context from many steps ago. LSTM adds three gates (input, forget, output) that explicitly control what to remember and what to discard, making them far better at capturing long-range dependencies.

## Stack

TensorFlow / Keras, NumPy, Matplotlib

## Next

`04-lstm-sentiment-analysis` — using LSTM for classification rather than generation, on real movie review sentiment data.
