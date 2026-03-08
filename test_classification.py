import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import cv2
import torch.nn.functional as F

from datasets.gtsrb import GTSRBDataset
from models.classification.mamba_classifier import MambaClassifier
from utils import get_args

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


def generate_confusion_matrix():
    # temp_ds = GTSRBDataset(root=PATH_DATA, split='train', transforms=transforms.Compose([
    #     transforms.Resize(SIZE), transforms.ToTensor()
    # ]))

    # mean, std = get_mean_and_std(temp_ds, workers=4)
    transforms_test = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.3417820930480957, 0.3126334846019745, 0.3216340243816376),
            std=(0.27580520510673523, 0.2633080780506134, 0.26914146542549133)
        )
    ])

    test_dataset = GTSRBDataset(root=PATH_DATA, split='test', transforms=transforms_test)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    checkpoint = torch.load(os.path.join(TRAINED, 'classification', 'best_checkpoint.pth'), map_location=DEVICE, weights_only=True)
    print(f'Accuracy: {checkpoint["best_accuracy"]}')
    model = MambaClassifier(dims=3, depth=4, ssm_d_state=16).to(DEVICE)
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

def inference():
    transforms_test = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.3417820930480957, 0.3126334846019745, 0.3216340243816376),
            std=(0.27580520510673523, 0.2633080780506134, 0.26914146542549133)
        )
    ])

    test_dataset_non_transform = GTSRBDataset(root=PATH_DATA, split='test')
    test_dataset = GTSRBDataset(root=PATH_DATA, split='test', transforms=transforms_test)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    checkpoint = torch.load(os.path.join(TRAINED, 'classification', 'best_checkpoint.pth'), map_location=DEVICE, weights_only=True)
    model = MambaClassifier(dims=3, depth=4, ssm_d_state=16).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for i, (img, label) in enumerate(test_dataloader):
            img, label = img.to(DEVICE), label.cpu().numpy()
            confidence, _ = torch.max(F.softmax(model(img), dim=1), dim=1)

            img_show, label_show = test_dataset_non_transform[i]

            img_show = to_tensor(img_show).numpy()
            img_show = img_show.transpose(1, 2, 0)
            img_show = (img_show * 255).astype(np.uint8)
            img_show = img_show[:, :, ::-1]
            cv2.imshow(f'{confidence.item() * 100:.2f}%: {label.item()} - label: {label_show}', img_show)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    generate_confusion_matrix()
    # inference()