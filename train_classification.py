import os
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
import torch
from typing import Type
from torchvision import transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets.tt100k import TT100KClassificationDataset
from models.classification.mamba_classifier import MambaClassifier
from utils.classification import get_args, get_mean_and_std, plot_confusion_matrix

args = get_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch
LR = args.lr
DEVICE = torch.device(args.device)
PATH_DATA = args.path_data
FOLDS = args.folds
WORKERS = args.workers
TRAINED = args.trained
LOGGING = args.logging
LOAD_CHECKPOINT = args.load_checkpoint
DEEP = args.deep
SIZE = args.size

def train(Dataset: Type[TT100KClassificationDataset]):
    mean, std = get_mean_and_std(
        TT100KClassificationDataset(
            root=PATH_DATA,
            transforms=transforms.Compose([
                transforms.Resize(SIZE),
                transforms.ToTensor()
            ]),
            split='train'
        ),
        workers=WORKERS
    )

    train_transforms = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = Dataset(root=PATH_DATA, transforms=train_transforms, split='train')
    indices = np.arange(len(train_dataset))
    labels = np.array(train_dataset.labels)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    folds = [(train_idx, validation_idx) for train_idx, validation_idx in skf.split(indices, labels)]

    start_fold = 0
    if os.path.exists(TRAINED):
        trained_folds = [
            int(d.replace("fold_", ""))
            for d in os.listdir(TRAINED)
            if d.startswith("fold_") and os.path.isdir(os.path.join(TRAINED, d))
        ]
        if len(trained_folds) > 0:
            start_fold = max(trained_folds) - 1

    test_dataset = Dataset(root=PATH_DATA, transforms=test_transforms, split='test')
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
    )

    final_best_checkpoint = os.path.join(TRAINED, "final_best_checkpoint.pth")
    final_best_f1_score = 0
    for fold in range(start_fold, FOLDS):
        print(f"\n" + "=" * 20 + f" TRAIN FOLD {fold + 1}/{FOLDS} " + "=" * 20)

        train_idx, validation_idx = folds[fold]
        fold_trained_path = os.path.join(TRAINED, f"fold_{fold + 1}")
        fold_logging_path = os.path.join(LOGGING, f"fold_{fold + 1}")
        os.makedirs(fold_trained_path, exist_ok=True)
        os.makedirs(fold_logging_path, exist_ok=True)

        train_dataloader = DataLoader(
            dataset=Subset(train_dataset, train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            drop_last=False
        )

        validation_dataloader = DataLoader(
            dataset=Subset(train_dataset, validation_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
            drop_last=False
        )

        model = MambaClassifier(dims=3, depth=DEEP, num_classes=153).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss()

        checkpoint_path_fold = os.path.join(fold_trained_path, "checkpoint.pth")
        best_checkpoint_path_fold = os.path.join(fold_trained_path, "best_checkpoint.pth")

        start_epoch = 0
        best_f1_score = 0

        if os.path.exists(checkpoint_path_fold) and LOAD_CHECKPOINT:
            if os.path.exists(checkpoint_path_fold):
                try:
                    checkpoint = torch.load(checkpoint_path_fold, map_location=DEVICE, weights_only=True)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    start_epoch = checkpoint["epoch"]
                    best_f1_score = checkpoint.get("best_f1_score", 0)
                    print(f"Resume Fold {fold + 1} from epoch {start_epoch + 1}")
                except Exception as ex:
                    print(f"Load checkpoint failed for fold {fold + 1}: {ex}")
        else:
            print(f"Start training for Fold {fold + 1}")

        num_iterations = len(train_dataloader)
        writer = SummaryWriter(fold_logging_path)

        for epoch in range(start_epoch, EPOCHS):
            progress_bar = tqdm(train_dataloader)
            model.train()
            total_loss_train = 0.0
            for i, (images, labels_batch) in enumerate(progress_bar):
                images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                progress_bar.set_description(
                    f"Fold [{fold + 1}/{FOLDS}] - "
                    f"Epoch [{epoch + 1}/{EPOCHS}] - "
                    f"Loss: {loss.item():.4f}"
                )
                total_loss_train += loss.item()
                writer.add_scalar(f"Fold {fold + 1}/Train/Loss", loss.item(), epoch * num_iterations + i)
            print(f"--> Loss for epoch {epoch + 1} : {(total_loss_train / num_iterations):.4f}")

            print("=" * 5 + f" VALIDATION EPOCH {epoch + 1}/{EPOCHS} " + "=" * 5)

            model.eval()
            list_prediction = []
            list_label = []
            total_loss_validation = 0.0

            with torch.no_grad():
                for images, labels_val in validation_dataloader:
                    images = images.to(DEVICE)
                    labels_val = labels_val.to(DEVICE)
                    outputs = model(images)
                    total_loss_validation += criterion(outputs, labels_val).item()

                    list_prediction.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    list_label.extend(labels_val.cpu().numpy())

            f1_score_epoch = f1_score(list_label, list_prediction, average="macro")
            print(f"Epoch {epoch + 1} Loss: {(total_loss_validation / len(validation_dataloader)):.4f}   F1 score: {f1_score_epoch:.4f}")

            is_best = f1_score_epoch > best_f1_score
            if is_best:
                best_f1_score = f1_score_epoch

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1_score": best_f1_score,
            }

            torch.save(checkpoint, checkpoint_path_fold)
            if is_best:
                torch.save(checkpoint, best_checkpoint_path_fold)
                print(f"-->[EPOCH BEST] New Best F1 score for Epoch {epoch + 1}\n")
            else:
                print()

            writer.add_scalar(f"Fold {fold + 1}/Validation/F1 score", f1_score_epoch, epoch + 1)
            plot_confusion_matrix(
                writer=writer,
                cm=confusion_matrix(list_label, list_prediction),
                class_names=train_dataset.labels,
                epoch=epoch + 1,
                fold=fold + 1
            )

        print(f"\n" + "=" * 20 + f" TEST FOLD {fold + 1}/{FOLDS} " + "=" * 20)

        best_checkpoint = torch.load(best_checkpoint_path_fold, map_location=DEVICE)
        model.load_state_dict(best_checkpoint["model_state_dict"])

        model.eval()
        list_prediction = []
        list_label = []
        total_loss_test = 0.0

        with torch.no_grad():
            for images, labels_val in test_dataloader:
                images = images.to(DEVICE)
                labels_val = labels_val.to(DEVICE)
                outputs = model(images)
                total_loss_test += criterion(outputs, labels_val).item()

                list_prediction.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                list_label.extend(labels_val.cpu().numpy())

        f1_score_fold = f1_score(list_label, list_prediction, average="macro")
        if f1_score_fold > final_best_f1_score:
            final_best_f1_score = f1_score_fold
            torch.save({
                "model_state_dict": model.state_dict(),
                "f1_score": f1_score_fold,
                "fold": fold + 1
            }, final_best_checkpoint)
            print(f"--> [GLOBAL BEST] New Best checkpoint found in Fold {fold + 1}!")

        print(
            f"Fold {fold + 1} "
            f"Loss: {(total_loss_test / len(test_dataloader)):.4f}  "
            f"F1 score: {f1_score_fold:.4f}"
        )

        plot_confusion_matrix(
            writer=writer,
            cm=confusion_matrix(list_label, list_prediction),
            class_names=train_dataset.labels,
            epoch=EPOCHS,
            train=False,
            fold=fold + 1
        )

        writer.close()
        del model
        del optimizer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    train(TT100KClassificationDataset)