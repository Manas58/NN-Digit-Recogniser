import numpy as np
import pandas as pd


def load_mnist_csv(path):
    """
    Load MNIST CSV file.
    First column: label
    Remaining 784 columns: pixel values
    """
    df = pd.read_csv(path)

    X = df.iloc[:, 1:].values   # pixels
    y = df.iloc[:, 0].values   # labels

    return X, y


def normalize(X):
    """
    Normalize pixel values to [0, 1]
    """
    return X.astype(np.float32) / 255.0


def one_hot_encode(y, num_classes=10):
    """
    Convert labels to one-hot encoding
    """
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y] = 1
    return Y


def train_val_split(X, Y, val_ratio=0.1):
    """
    Split dataset into training and validation sets
    """
    indices = np.random.permutation(X.shape[0])

    val_size = int(X.shape[0] * val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (
        X[train_idx],
        X[val_idx],
        Y[train_idx],
        Y[val_idx]
    )

def accuracy(Y_true, Y_pred):
    """
    Y_true: one-hot labels
    Y_pred: predicted probabilities
    """
    true_labels = np.argmax(Y_true, axis=1)
    pred_labels = np.argmax(Y_pred, axis=1)
    return np.mean(true_labels == pred_labels)

def predict_label(Y_pred):
    """
    Y_pred: output probabilities from model
    Returns predicted class labels
    """
    return np.argmax(Y_pred, axis=1)
