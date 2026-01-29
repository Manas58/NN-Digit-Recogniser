import numpy as np

# ---------------- ReLU ----------------

def relu(Z):
    """
    ReLU activation (forward pass)
    Z: pre-activation values
    """
    return np.maximum(0, Z)


def relu_backward(dA, Z):
    """
    ReLU backward pass
    dA: gradient from next layer
    Z: pre-activation values from forward pass
    """
    dZ = dA * (Z > 0)
    return dZ


# ---------------- Softmax ----------------
def softmax(Z):
    """
    Softmax activation (forward pass)
    Z: (n_samples, n_classes)
    """
    # numerical stability trick
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)