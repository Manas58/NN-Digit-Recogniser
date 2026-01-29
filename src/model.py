import numpy as np
from activations import relu, relu_backward, softmax
from loss import cross_entropy_loss

def init_parameters(input_dim, hidden_dim, output_dim, seed=42):
    """
    He initialization for ReLU network
    """
    np.random.seed(seed)

    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros((1, hidden_dim))

    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros((1, output_dim))

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return params

def forward_pass(X, params):
    """
    Forward propagation
    X: (n_samples, 784)
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = X @ W1 + b1          # (n, 128)
    A1 = relu(Z1)             # (n, 128)

    Z2 = A1 @ W2 + b2         # (n, 10)
    A2 = softmax(Z2)          # (n, 10)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache

def backward_pass(Y, params, cache):
    """
    Backward propagation
    Y: true labels (one-hot)
    """
    m = Y.shape[0]

    W2 = params["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    X = cache["X"]

    # ---- Output layer ----
    dZ2 = A2 - Y                      # (n, 10)
    dW2 = (A1.T @ dZ2) / m            # (128, 10)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # ---- Hidden layer ----
    dA1 = dZ2 @ W2.T                  # (n, 128)
    dZ1 = relu_backward(dA1, Z1)      # (n, 128)
    dW1 = (X.T @ dZ1) / m             # (784, 128)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads

def update_parameters(params, grads, learning_rate):
    """
    Update parameters using SGD
    """
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]

    return params

def save_model(params, path):
    np.savez(
        path,
        W1=params["W1"],
        b1=params["b1"],
        W2=params["W2"],
        b2=params["b2"]
    )


def load_model(path):
    data = np.load(path)
    return {
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"]
    }
