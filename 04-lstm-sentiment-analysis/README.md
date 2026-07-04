# 04 — IMDB Sentiment Analysis — LSTM

A word-level LSTM trained on the IMDB movie reviews dataset to classify sentiment as positive or negative. The fourth project in the `deep-learning-foundations` series — moving from character-level sequence generation (RNN) to word-level sequence classification.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **85.03%** |
| Test Loss | 0.3879 |
| Dataset | IMDB 50K reviews (25K train + 25K test) |
| Vocabulary size | 10,000 words |
| Max sequence length | 200 words |
| Architecture | Embedding(128) → SpatialDropout1D(0.2) → LSTM(128) → Dense(sigmoid) |

## Custom Review Predictions

| Review | Sentiment | Confidence |
|---|---|---|
| "This movie was absolutely brilliant and I loved every minute of it" | POSITIVE | 93.93% |
| "Terrible film waste of time boring and predictable" | NEGATIVE | 98.09% |
| "The acting was great but the story was weak and disappointing" | NEGATIVE | 82.60% |
| "One of the best movies I have ever seen in my life" | POSITIVE | see output |

## RNN vs LSTM — Generation vs Classification

| | `03-rnn-text-generation` | `04-lstm-sentiment-analysis` |
|---|---|---|
| Task | Generate next character | Classify entire review |
| Input | Character sequences | Word sequences |
| Output | Character at every timestep | Single probability |
| `return_sequences` | True on all layers | False — final state only |
| Loss | Sparse categorical crossentropy | Binary crossentropy |

## Files

- `lstm_sentiment_analysis.ipynb` — full notebook: data loading, padding, model training, evaluation, confusion matrix, custom review predictor, saved model
- `lstm_sentiment.keras` — trained model
- `review_length_distribution.png` — distribution of review lengths with MAX_LEN cutoff
- `training_curves.png` — train vs validation accuracy and loss per epoch
- `confusion_matrix.png` — per-class prediction breakdown

## Key Implementation Details

- **`pad_sequences(maxlen=200)`** — pads short reviews with zeros, truncates long ones. Reviews longer than 200 words: ~20% of training data
- **`SpatialDropout1D(0.2)`** — drops entire word embedding vectors randomly, more effective than standard dropout for sequence data
- **`recurrent_dropout=0.2`** — dropout applied to recurrent connections inside the LSTM, reduces overfitting on the hidden state
- **`EarlyStopping(patience=3)`** — stops training when val_loss stops improving, restores best weights automatically
- **Custom predictor** — takes raw text string, tokenizes, pads, and returns sentiment + confidence

## Why LSTM for Sentiment

Movie reviews have long-range dependencies — a negative phrase at the start of a 200-word review affects the overall sentiment. Simple RNNs lose that context due to vanishing gradients. LSTM gates retain relevant information across the full sequence length.

## Stack

TensorFlow / Keras, NumPy, Matplotlib, Seaborn, Scikit-learn

## Next

`05-transformer-from-scratch` — replacing recurrence entirely with self-attention. Transformers process all tokens in parallel (no sequential hidden state), making them faster and better at capturing long-range dependencies.

