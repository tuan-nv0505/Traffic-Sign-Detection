import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score
from tqdm.autonotebook import tqdm
import torch.nn as nn

from datasets.gtsrb import GTSRBDataset
from models.classification.mamba_classifier import MambaClassifier
from models.loss import FocalLoss
from utils.classification import get_args, get_mean_and_std

args = get_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch
LR = args.lr
DEVICE = torch.device(args.device)
PATH_DATA = args.path_data
WORKERS = args.workers
TRAINED = args.trained
LOGGING = args.logging
LOAD_CHECKPOINT = args.load_checkpoint
DEEP = args.deep
SIZE = args.size

Dataset = GTSRBDataset


def get_alpha(stats, num_classes=151, beta=0.999):
    counts = np.array([stats.get(i, 0) for i in range(num_classes)], dtype=np.float64)

    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)

    weights = np.zeros_like(effective_num)
    mask = effective_num > 0
    weights[mask] = 1.0 / effective_num[mask]

    if weights.sum() > 0:
        weights = weights / np.sum(weights) * num_classes

    return torch.FloatTensor(weights)

def train():
    os.makedirs(TRAINED, exist_ok=True)
    os.makedirs(LOGGING, exist_ok=True)

    temp_ds = Dataset(root=PATH_DATA, split='train', transforms=transforms.Compose([
        transforms.Resize(SIZE), transforms.ToTensor()
    ]))
    mean, std = get_mean_and_std(temp_ds, workers=WORKERS)

    transforms_train = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        ),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = Dataset(root=PATH_DATA, transforms=transforms_train, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    test_dataset = Dataset(root=PATH_DATA, transforms=transforms_test, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    model = MambaClassifier(dims=3, depth=DEEP, num_classes=43).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # criterion = FocalLoss(alpha=get_alpha(train_dataset.stats, num_classes=43, beta=0.999).to(DEVICE), gamma=2.0)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(LOGGING)

    start_epoch = 0
    best_accuracy = 0

    checkpoint_path = os.path.join(TRAINED, 'checkpoint.pth')
    if LOAD_CHECKPOINT and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            best_accuracy = checkpoint.get("best_accuracy", 0)
            print(f"Resuming from epoch {start_epoch + 1}")
        except Exception as ex:
            print(f"Load checkpoint failed: {ex}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss_train = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for i, (images, labels_batch) in enumerate(progress_bar):
            images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_dataloader) + i)

        print(f"--> Average Train Loss: {(total_loss_train / len(train_dataloader)):.4f}")

        model.eval()
        list_prediction, list_label = [], []
        total_loss_val = 0.0

        with torch.no_grad():
            for images, labels_val in test_dataloader:
                images, labels_val = images.to(DEVICE), labels_val.to(DEVICE)
                outputs = model(images)
                total_loss_val += criterion(outputs, labels_val).item()

                list_prediction.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                list_label.extend(labels_val.cpu().numpy())

        accuracy = accuracy_score(list_label, list_prediction)
        avg_val_loss = total_loss_val / len(test_dataloader)

        print(f"Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/Accuracy", accuracy, epoch)

        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy

        checkpoint_data = {
            "state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
        }

        torch.save(checkpoint_data, checkpoint_path)
        if is_best:
            torch.save(checkpoint_data, os.path.join(TRAINED, 'best_checkpoint.pth'))
            print(f"--> [NEW BEST] Accuracy score improved to {best_accuracy:.4f}\n")
        else:
            print()

    writer.close()
    print('Training finished!')

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    train()