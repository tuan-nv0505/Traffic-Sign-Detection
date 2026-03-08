import os
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from datasets.gtsdb import GTSDBDataset
from models.detection.faster_rcnn import FasterRCNN
from utils import get_args

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

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    os.makedirs(TRAINED, exist_ok=True)
    os.makedirs(LOGGING, exist_ok=True)

    train_dataset = GTSDBDataset(root=PATH_DATA, split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        collate_fn=collate_fn
    )

    test_dataset = GTSDBDataset(root=PATH_DATA, split='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        collate_fn=collate_fn
    )

    checkpoint = torch.load('best_checkpoint.pth', weights_only=True, map_location=DEVICE)
    new_checkpoint = {}
    for k, v in checkpoint['state_dict'].items():
        if k.startswith("backbone."):
            new_k = k[len("backbone."):]
        else:
            new_k = k
        new_checkpoint[new_k] = v
    model = FasterRCNN(num_classes=44, weight=new_checkpoint).to(DEVICE)
    for name, param in model.named_parameters():
        if 'extractor' in name and 'fpn' not in name:
            param.requires_grad_(False)
    optimizer = Adam(model.parameters(), lr=LR)

    metric = MeanAveragePrecision(box_format='xyxy').to(DEVICE)

    writer = SummaryWriter(os.path.join(LOGGING, 'detection'))
    start_epoch = 0
    best_map = 0.0

    checkpoint_path = os.path.join(TRAINED, 'faster_rcnn_checkpoint.pth')

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss_train = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

        for images, targets in progress_bar:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            losses = model(images, targets)
            final_loss = sum(loss for loss in losses.values())

            final_loss.backward()
            optimizer.step()

            total_loss_train += losses.item()
            progress_bar.set_postfix({"loss": f"{losses.item():.4f}"})

        avg_train_loss = total_loss_train / len(train_dataloader)
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)
        print(f"--> Average Train Loss: {avg_train_loss:.4f}")

        model.eval()
        metric.reset()
        progress_bar_val = tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")

        with torch.no_grad():
            for images, targets in progress_bar_val:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                outputs = model(images, targets)

                metric.update(outputs, targets)

        results = metric.compute()
        current_map = results['map_50'].item()

        print(f"Val mAP@0.5: {current_map:.4f} | mAP@0.5:0.95: {results['map'].item():.4f}")
        writer.add_scalar("Val/mAP_50", current_map, epoch)
        writer.add_scalar("Val/mAP_0.5_0.95", results['map'].item(), epoch)

        is_best = current_map > best_map
        if is_best:
            best_map = current_map

        checkpoint_data = {
            "state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "best_map": best_map,
        }

        torch.save(checkpoint_data, checkpoint_path)
        if is_best:
            torch.save(checkpoint_data, os.path.join(TRAINED, 'faster_rcnn_best_checkpoint.pth'))
            print(f"--> [NEW BEST] mAP score improved to {best_map:.4f}\n")

    writer.close()
    print('Training finished!')

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    train()