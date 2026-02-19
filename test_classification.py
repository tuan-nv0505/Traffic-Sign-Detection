import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from datasets.gtsrb import GTSRBDataset
from models.classification.mamba_classifier import MambaClassifier
from utils.classification import get_args, get_mean_and_std

args = get_args()

DEVICE = torch.device(args.device)
TRAINED = args.trained
SIZE = args.size
PATH_DATA = args.path_data

def plot_confusion_matrix(list_label, list_prediction, num_classes=43):
    cm = confusion_matrix(list_label, list_prediction)
    plt.figure(figsize=(20, 15))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(num_classes),
        yticklabels=range(num_classes)
    )

    plt.xlabel('Predicted Label', fontsize=15)
    plt.ylabel('True Label', fontsize=15)
    plt.title('Confusion Matrix - GTSRB Traffic Signs', fontsize=20)

    plt.savefig('confusion_matrix.png', dpi=300)


def test():
    temp_ds = GTSRBDataset(root=PATH_DATA, split='train', transforms=transforms.Compose([
        transforms.Resize(SIZE), transforms.ToTensor()
    ]))

    mean, std = get_mean_and_std(temp_ds, workers=4)
    transforms_test = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_dataset = GTSRBDataset(root=PATH_DATA, split='test', transforms=transforms_test)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    checkpoint = torch.load(os.path.join(TRAINED, 'best_checkpoint.pth'), map_location=DEVICE, weights_only=True)
    model = MambaClassifier(dims=3, depth=3).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

    list_prediction, list_label = [], []
    progress_bar = tqdm(test_dataloader)
    with torch.no_grad():
        for img, label in progress_bar:
            img, label = img.to(DEVICE), label.cpu().numpy()
            list_label.extend(label)
            prediction = torch.argmax(model(img), dim=1).cpu().numpy()
            list_prediction.extend(prediction)

    plot_confusion_matrix(list_label, list_prediction, num_classes=43)

if __name__ == '__main__':
    test()