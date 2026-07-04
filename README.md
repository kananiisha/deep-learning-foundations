# Deep Learning Foundations

A progressive deep learning learning path — built from scratch, one architecture at a time. Each project is self-contained with a full notebook, trained model, and README explaining the what, why, and results.

## Roadmap

| # | Project | Architecture | Status |
|---|---|---|---|
| 01 | [ANN — MNIST Digit Classifier](./01-ann-mnist-digit-classifier/) | Fully Connected ANN | ✅ 97.86% |
| 02 | [CNN — MNIST Digit Classifier](./02-cnn-image-classifier/) | Convolutional Neural Network | ✅ 99.14% |
| 03 | [RNN — Shakespeare Text Generation](./03-rnn-text-generation/) | LSTM | ✅ Complete |
| 04 | [LSTM — Sentiment Analysis](./04-lstm-sentiment-analysis/) | LSTM | ✅ 85.03% |
| 05 | [Transformer — Sentiment Analysis](./05-transformer-sentiment-analysis/) | Self-Attention | ✅ 87.57% |
| 06 | BERT — Text Classification | Fine-tuned Transformer | 📅 Planned |

## Why this structure?

Each architecture builds on the previous one — ANN establishes the baseline, CNN adds spatial awareness, RNN adds sequence modeling, LSTM fixes the vanishing gradient problem, and Transformers replace recurrence entirely with attention. Following this progression makes the "why" behind each architecture obvious, not just the "how."

## Stack

TensorFlow / Keras, NumPy, Matplotlib, Seaborn, Scikit-learn

## Setup

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```
