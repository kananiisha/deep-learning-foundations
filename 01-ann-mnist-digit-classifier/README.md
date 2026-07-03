# 01 — ANN MNIST Digit Classifier

A from-scratch artificial neural network trained on MNIST, built in three progressive stages to show — with measured results — how architecture choices affect performance.

## Results

| Stage | Architecture | Test Accuracy |
|---|---|---|
| 1 | No hidden layer (input → softmax output) | 92.45% |
| 2 | + Hidden layer (100 units, ReLU) | 97.74% |
| 3 | + `Flatten` layer, trained 10 epochs | 97.66% |

Adding a hidden layer gave the single biggest accuracy gain — it lets the network learn non-linear feature combinations instead of mapping pixels to classes in one linear step. Stage 3 swaps manual reshaping for a built-in `Flatten` layer and trains longer; training accuracy (99.36%) pulls ahead of validation accuracy (97.97%) — early signs of overfitting from the extra epochs.

## Files

- `mnist_digit_classifier.ipynb` — full notebook: data loading, preprocessing, all three model stages, confusion matrices, saved model, and single-image inference demo
- `digit_classifier.keras` — final trained model (stage 3), ready to reload without retraining

## Key Implementation Details

- **`softmax`** on the output layer — correct for multiclass classification, forces 10 outputs to sum to 1
- **`validation_split=0.1`** on every training run to track train vs val accuracy per epoch
- **Model persistence** via `model.save()` + `keras.models.load_model()` with single-image prediction demo

## Stack

TensorFlow / Keras, NumPy, Matplotlib, Seaborn

## Next

`02-cnn-image-classifier` — convolutional layers exploit the 2D spatial structure of images that this fully-connected network ignores entirely.
