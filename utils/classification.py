from argparse import ArgumentParser
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def get_args():
    parser = ArgumentParser()

    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--path_data", type=str, default="data/tt100k")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trained", type=str, default="trained")
    parser.add_argument("--logging", type=str, default="tensorboard")
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--deep", type=int, default=4)
    parser.add_argument("--size", nargs=2, type=int, metavar=("H, W"), default=(48, 48))

    args = parser.parse_args()
    if args.size:
        args.size = tuple(map(int, args.size))

    return args

def plot_confusion_matrix(writer, cm, class_names, epoch, fold=None, train=True):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    cm = cm.astype(float)

    recall_denom = cm.sum(axis=1, keepdims=True)
    recall_denom[recall_denom == 0] = 1
    recall = cm / recall_denom

    precision_denom = cm.sum(axis=0, keepdims=True)
    precision_denom[precision_denom == 0] = 1
    precision = cm / precision_denom

    f1_diagonal = np.zeros(len(class_names))
    for i in range(len(class_names)):
        p = precision[i, i]
        r = recall[i, i]
        if (p + r) > 0:
            f1_diagonal[i] = 2 * (p * r) / (p + r)

    f1_matrix = np.diag(f1_diagonal)

    figure = plt.figure(figsize=(20, 20))
    plt.imshow(f1_matrix, interpolation='nearest', cmap="ocean")
    plt.title("Per-Class F1-Score Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = 0.5

    for i in range(len(class_names)):
        val = np.around(f1_diagonal[i], decimals=2)
        color = "black" if val > threshold else "white"

        plt.text(i, i, val,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color=color,
                 fontweight='bold')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    s1 = f"Fold {fold}/" if fold is not None else ""
    s2 = "Validation" if train else "Test"
    tag = f"{s1}F1 Score {s2}"

    writer.add_figure(tag, figure, epoch)
    plt.close(figure)


def get_mean_and_std(dataset, workers=4):
    data_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=workers
    )

    n_images = 0
    mean_x = torch.zeros(3)
    mean_x_squared = torch.zeros(3)

    for images, _ in data_loader:
        b = images.size(0)

        batch_mean = torch.mean(images, dim=[0, 2, 3])
        batch_mean_sq = torch.mean(images ** 2, dim=[0, 2, 3])

        mean_x = (mean_x * n_images + batch_mean * b) / (n_images + b)
        mean_x_squared = (mean_x_squared * n_images + batch_mean_sq * b) / (n_images + b)

        n_images += b

    # std = sqrt(E[X^2] - (E[X])^2)
    std = torch.sqrt(mean_x_squared - mean_x ** 2)

    return tuple(mean_x.tolist()), tuple(std.tolist())

if __name__ == '__main__':
    args = get_args()
    print(args.size)
