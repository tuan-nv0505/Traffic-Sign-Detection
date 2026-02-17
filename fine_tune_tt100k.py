import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from typing import Type

from datasets.tt100k import TT100KClassificationDataset
from datasets.gtsrb import GTSRBDataset
from models.classification.mamba_classifier import MambaClassifier
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

Dataset = TT100KClassificationDataset

def train():
    os.makedirs(TRAINED, exist_ok=True)
    os.makedirs(LOGGING, exist_ok=True)

    temp_ds = Dataset(root=PATH_DATA, split='train', transforms=transforms.Compose([
        transforms.Resize(SIZE), transforms.ToTensor()
    ]))
    mean, std = get_mean_and_std(temp_ds, workers=WORKERS)

    data_transforms = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = Dataset(root=PATH_DATA, transforms=data_transforms, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    test_dataset = Dataset(root=PATH_DATA, transforms=data_transforms, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    checkpoint = torch.load('best_checkpoint.pth', map_location=DEVICE, weights_only=True)
    state_dict_raw = checkpoint['state_dict']

    keys_to_exclude = ['classifier.classifier.head.weight', 'classifier.classifier.head.bias']
    state_dict = {k: v for k, v in state_dict_raw.items() if k not in keys_to_exclude}

    model = MambaClassifier(dims=3, depth=DEEP, num_classes=151).to(DEVICE)
    model_dict = model.state_dict()

    matched_keys = []
    mismatched_keys = []
    missing_keys = []

    for k, v in model_dict.items():
        if k in state_dict:
            if v.shape == state_dict[k].shape:
                matched_keys.append(k)
            else:
                mismatched_keys.append(f"{k} (Size: {state_dict[k].shape} -> {v.shape})")
        else:
            missing_keys.append(k)

    model.load_state_dict(state_dict, strict=False)

    print(f"\n" + "=" * 40)
    print(f"Success: {len(matched_keys)} layers")
    print(f"Failed: {len(mismatched_keys)} layers")
    print(f"Blank: {len(missing_keys)} layers")
    print("=" * 40)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(LOGGING)

    start_epoch = 0
    best_f1_score = 0

    checkpoint_path = os.path.join(TRAINED, 'checkpoint.pth')
    if LOAD_CHECKPOINT and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            best_f1_score = checkpoint.get("best_f1_score", 0)
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

        list_prediction, list_label = [], []
        total_loss_val = 0.0

        with torch.no_grad():
            for images, labels_val in test_dataloader:
                images = images.to(DEVICE).float()
                labels_val = labels_val.to(DEVICE).long()

                mask = (labels_val >= 0) & (labels_val < 151)
                if not mask.all():
                    bad_labels = labels_val[~mask].tolist()
                    images = images[mask]
                    labels_val = labels_val[mask]

                    if len(labels_val) == 0:
                        continue

                outputs = model(images)
                loss = criterion(outputs.float(), labels_val)
                total_loss_val += loss.item()

                preds = torch.argmax(outputs, dim=1)
                list_prediction.extend(preds.cpu().numpy())
                list_label.extend(labels_val.cpu().numpy())

        f1score = f1_score(list_label, list_prediction, average="macro")
        avg_val_loss = total_loss_val / len(test_dataloader)

        print(f"Val Loss: {avg_val_loss:.4f} | F1 score: {f1score:.4f}")
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/F1", f1score, epoch)

        is_best = f1score > best_f1_score
        if is_best:
            best_f1_score = f1score

        checkpoint_data = {
            "state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "best_f1_score": best_f1_score,
        }

        torch.save(checkpoint_data, checkpoint_path)
        if is_best:
            torch.save(checkpoint_data, os.path.join(TRAINED, 'best_checkpoint.pth'))
            print(f"--> [NEW BEST] F1 score improved to {best_f1_score:.4f}\n")
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