# 01 — ANN MNIST Digit Classifier

A fully-connected Artificial Neural Network trained on MNIST — the baseline model in the `deep-learning-foundations` series. Built to establish a measurable accuracy floor before introducing convolutional layers in `02-cnn-image-classifier`.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **97.86%** |
| Test Loss | 0.0869 |
| Architecture | Flatten → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dense(10, softmax) |
| Epochs | 15 |
| Optimizer | Adam |

## Files

- `mnist_digit_classifier.ipynb` — full notebook: data loading, preprocessing, model training, training curves, confusion matrix, inference demo, saved model
- `ann_digit_classifier.keras` — trained model, ready to reload without retraining
- `sample_images.png` — sample MNIST digits
- `training_curves.png` — train vs validation accuracy and loss per epoch
- `confusion_matrix.png` — per-class prediction breakdown
- `inference_demo.png` — single image prediction with confidence score

## Key Implementation Details

- **`Flatten` layer** inside the model — no manual reshaping needed in preprocessing
- **`Dropout(0.2)`** — regularization to reduce overfitting
- **`softmax`** on output layer — correct for multiclass classification, forces 10 outputs to sum to 1
- **`validation_split=0.1`** — tracks train vs val accuracy every epoch
- **Confidence score** on inference demo — not just predicted label but probability

## Core Limitation of ANN on Images

`Flatten` destroys all spatial structure. A pixel in the top-left and a pixel in the bottom-right are treated as completely independent features. The network learns which pixel values correlate with which labels — but has no concept of edges, curves, or neighboring pixels.

## Stack

TensorFlow / Keras, NumPy, Matplotlib, Seaborn, Scikit-learn

## Next

`02-cnn-image-classifier` — same dataset, same task. Convolutional layers preserve spatial structure and scan the image with learned filters, hitting **99.14%** vs this ANN's **97.86%**.
