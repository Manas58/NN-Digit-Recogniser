import os

import numpy as np
from data_utils import load_mnist_csv, normalize, one_hot_encode, train_val_split, accuracy, predict_label
from model import (
    init_parameters,
    forward_pass,
    backward_pass,
    update_parameters,
    save_model
)
from loss import cross_entropy_loss
from visualize import show_single_image, show_wrong_predictions


# ---------------- CONFIG ----------------
INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10

LEARNING_RATE = 0.1
EPOCHS = 10
BATCH_SIZE = 128
# ----------------------------------------


# ---------------- LOAD DATA ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "mnist_train.csv")

X, y = load_mnist_csv(DATA_PATH)
X = normalize(X)
Y = one_hot_encode(y)

X_train, X_val, Y_train, Y_val = train_val_split(X, Y)

print("Data loaded")
print("Train samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
# ------------------------------------------


# ---------------- INIT MODEL ----------------
params = init_parameters(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
# --------------------------------------------


# ---------------- TRAINING ----------------
num_samples = X_train.shape[0]

for epoch in range(EPOCHS):
    permutation = np.random.permutation(num_samples)
    X_shuffled = X_train[permutation]
    Y_shuffled = Y_train[permutation]

    epoch_loss = 0

    for i in range(0, num_samples, BATCH_SIZE):
        X_batch = X_shuffled[i:i + BATCH_SIZE]
        Y_batch = Y_shuffled[i:i + BATCH_SIZE]

        Y_hat, cache = forward_pass(X_batch, params)
        loss = cross_entropy_loss(Y_batch, Y_hat)

        grads = backward_pass(Y_batch, params, cache)
        params = update_parameters(params, grads, LEARNING_RATE)

        epoch_loss += loss

    epoch_loss /= (num_samples // BATCH_SIZE)

    # validation
    Y_val_pred, _ = forward_pass(X_val, params)
    val_acc = accuracy(Y_val, Y_val_pred)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Val Acc: {val_acc*100:.2f}%"
    )
# ------------------------------------------


# ---------------- SAVE MODEL ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(BASE_DIR, "models", "mnist_nn.npz")
save_model(params, SAVE_PATH)
print("Model saved to models/mnist_nn.npz")
# -------------------------------------------


# ---------------- FINAL EVALUATION ----------------
Y_val_pred, _ = forward_pass(X_val, params)
val_acc = accuracy(Y_val, Y_val_pred)

print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
# -------------------------------------------------


# ---------------- SINGLE PREDICTION ----------------
idx = np.random.randint(0, X_val.shape[0])

X_sample = X_val[idx:idx+1]
Y_sample = Y_val[idx]

Y_pred, _ = forward_pass(X_sample, params)

pred_digit = predict_label(Y_pred)[0]
true_digit = np.argmax(Y_sample)

print("\nSingle sample prediction:")
print("Predicted:", pred_digit)
print("Actual   :", true_digit)

show_single_image(
    X_val[idx],
    true_label=true_digit,
    pred_label=pred_digit,
    filename="correct_prediction.png"
)

print("Saved correct_prediction.png")
# --------------------------------------------------


# ---------------- WRONG PREDICTIONS ----------------
show_wrong_predictions(
    X_val,
    Y_val,
    Y_val_pred,
    max_images=1
)
# --------------------------------------------------
