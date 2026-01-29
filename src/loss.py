import numpy as np

def cross_entropy_loss(Y, Y_hat):
    """
    Categorical cross-entropy loss
    Y: true labels (one-hot), shape (n_samples, n_classes)
    Y_hat: predicted probabilities, same shape
    """
    eps = 1e-9
    Y_hat = np.clip(Y_hat, eps, 1 - eps)

    loss = -np.sum(Y * np.log(Y_hat)) / Y.shape[0]
    return loss
