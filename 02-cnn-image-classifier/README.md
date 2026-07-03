# 02 — CNN MNIST Digit Classifier

A Convolutional Neural Network trained on MNIST — built to directly compare against the ANN baseline in `01-ann-mnist-digit-classifier`. The core question: how much does exploiting 2D spatial structure actually matter for image classification?

**Answer: ANN hit 97.86%. CNN hits 99.14% — with fewer errors across every digit class.**

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **99.14%** |
| Test Loss | 0.0277 |
| Architecture | Conv2D(32) → Pool → Conv2D(64) → Pool → Flatten → Dense(64) → Dropout(0.5) → Dense(10) |
| Epochs | 10 |
| Optimizer | Adam |

## ANN vs CNN Comparison

| Model | Architecture | Test Accuracy | Parameters |
|---|---|---|---|
| ANN (`01`) | Flatten → Dense(128) → Dropout → Dense(64) → Dense(10) | 97.86% | ~109K |
| CNN (`02`) | Conv2D(32) → Pool → Conv2D(64) → Pool → Dense(64) → Dense(10) | **99.14%** | ~93K |

CNN achieves higher accuracy with *fewer* parameters — because convolutional weight sharing is more efficient than fully-connected layers for image data.

## Files

- `cnn_digit_classifier.ipynb` — full notebook: data loading, model architecture, training, evaluation, filter visualization, feature maps, inference demo, saved model
- `cnn_digit_classifier.keras` — trained model, ready to reload without retraining
- `training_curves.png` — train vs validation accuracy and loss per epoch
- `confusion_matrix.png` — per-class prediction breakdown
- `conv_filters.png` — 32 learned 3×3 filters from Conv2D layer 1
- `feature_maps.png` — what each filter activates on for a real digit
- `inference_demo.png` — single image prediction with confidence score

## Key Implementation Details

- **Input shape `(28, 28, 1)`** — CNN needs 4D input (height, width, channels), unlike ANN's flat vector
- **Two Conv2D + MaxPooling blocks** — first layer learns low-level features (edges, curves), second layer combines them into higher-level patterns
- **`Dropout(0.5)`** — stronger regularization than ANN since CNN has more expressive power
- **Filter visualization** — 32 learned 3×3 filters shown after training
- **Feature map visualization** — activations of Conv2D layer 1 on a real test image

## Why CNN Wins

The ANN flattens each 28×28 image into 784 independent pixels. A CNN scans with small 3×3 filters that look at local patches — learning that an edge in the top-left of a '7' is the same pattern as an edge in the top-right of a '1'. This spatial weight sharing is why CNNs need fewer parameters to achieve higher accuracy on image data.

## Stack

TensorFlow / Keras, NumPy, Matplotlib, Seaborn, Scikit-learn

## Next

`03-rnn-text-generation` — sequence modeling, where the *order* of inputs matters, not their spatial arrangement.
