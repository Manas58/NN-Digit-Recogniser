import matplotlib.pyplot as plt
import numpy as np


def show_single_image(X, true_label, pred_label, filename):
    image = X.reshape(28, 28)

    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="gray")
    plt.title(f"True: {true_label} | Predicted: {pred_label}")
    plt.axis("off")

    plt.savefig(filename)
    plt.close()


def show_wrong_predictions(X, Y_true, Y_pred, max_images=5):
    """
    Save images where model predicted incorrectly
    """
    true_labels = np.argmax(Y_true, axis=1)
    pred_labels = np.argmax(Y_pred, axis=1)

    wrong_idxs = np.where(true_labels != pred_labels)[0]

    print(f"Total wrong predictions: {len(wrong_idxs)}")

    for i, idx in enumerate(wrong_idxs[:max_images]):
        if i > 1:
            show_single_image(
                X[idx],
                true_label=true_labels[idx],
                pred_label=pred_labels[idx],
                filename=f"wrong_prediction_{i+1}.png"
            )
        else:
            show_single_image(
                X[idx],
                true_label=true_labels[idx],
                pred_label=pred_labels[idx],
                filename=f"wrong_prediction.png"
            )

    print(f"Saved {min(max_images, len(wrong_idxs))} wrong prediction images")
