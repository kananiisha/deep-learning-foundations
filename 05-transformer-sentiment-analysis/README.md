# 05 — Sentiment Analysis — Transformer from Scratch

A Transformer encoder built entirely from scratch using TensorFlow/Keras — no `transformers` library, no pre-trained weights. Every component is implemented manually: positional encoding, multi-head self-attention, feedforward sublayer, residual connections, and layer normalization.

Trained on IMDB movie reviews for direct comparison with the LSTM in `04-lstm-sentiment-analysis`.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **87.57%** |
| Test Loss | 0.3080 |
| LSTM Baseline (`04`) | 85.03% |
| Improvement over LSTM | +2.54% |
| Parameters | ~300K (vs ~1.4M for LSTM) |
| Architecture | Embedding + PE → TransformerEncoder × 2 → GlobalAvgPool → Dense(sigmoid) |

## LSTM vs Transformer

| Model | Test Accuracy | Parameters | Processing |
|---|---|---|---|
| LSTM (`04`) | 85.03% | ~1.4M | Sequential |
| Transformer (`05`) | **87.57%** | ~300K | Parallel |

Transformer achieves higher accuracy with 4× fewer parameters — because attention directly connects any two tokens regardless of distance, while LSTM hidden state degrades over long sequences.

## Components Built from Scratch

### Positional Encoding
Transformers process all tokens in parallel — no inherent sense of order. Positional encoding adds position information using sine and cosine functions at different frequencies, giving each position a unique fingerprint.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Multi-Head Self-Attention
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
```
- **Q (Query):** what this token is looking for
- **K (Key):** what each token offers  
- **V (Value):** what each token contributes
- **4 heads** — each head attends to different aspects (syntax, semantics, sentiment markers)

### Transformer Encoder Block
Multi-Head Attention + Feedforward Network, each with residual connection + layer normalization — allows gradients to flow directly through deep networks.

## Files

- `transformer_sentiment.ipynb` — full notebook: data loading, positional encoding visualization, attention implementation, model training, evaluation, attention weight visualization, comparison with LSTM
- `transformer_sentiment.keras` — trained model
- `positional_encoding.png` — heatmap of the 200×64 positional encoding matrix
- `training_curves.png` — train vs validation accuracy and loss per epoch
- `confusion_matrix.png` — per-class prediction breakdown

## Key Implementation Details

- **`d_model=64`, `num_heads=4`, `dff=128`, `num_blocks=2`** — lightweight config for CPU training
- **`GlobalAveragePooling1D`** — aggregates all 200 token representations into one vector for classification (alternative to using only the `[CLS]` token)
- **`EarlyStopping(patience=3)`** — stops when val_loss plateaus
- **Custom layers with `get_config`** — enables proper model serialization and saving

## Stack

TensorFlow / Keras, NumPy, Matplotlib, Seaborn, Scikit-learn

## Next

`06-bert-text-classification` — fine-tuning a pre-trained BERT model. Instead of building attention from scratch, we leverage weights already trained on billions of words — showing the full progression from scratch implementation to transfer learning.
