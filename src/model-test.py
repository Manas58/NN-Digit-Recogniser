import os
import numpy as np

from data_utils import load_mnist_csv, normalize, one_hot_encode,accuracy
from model import forward_pass, load_model


# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "mnist_test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_nn.npz")
# -------------------------------------------

# ---------------- LOAD TEST DATA ----------------
X_test, y_test = load_mnist_csv(DATA_PATH)
X_test = normalize(X_test)
Y_test = one_hot_encode(y_test)

print("Test data loaded")
print("Test samples:", X_test.shape[0])
# -----------------------------------------------

# ---------------- LOAD TRAINED MODEL ----------------
params = load_model(MODEL_PATH)
print("Model loaded from mnist_nn.npz")
# ---------------------------------------------------

# ---------------- RUN TESTING ----------------
Y_test_pred, _ = forward_pass(X_test, params)
test_acc = accuracy(Y_test, Y_test_pred)

print(f"\n✅ Test Accuracy on MNIST test set: {test_acc * 100:.2f}%")
# ---------------------------------------------