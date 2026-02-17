import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class GTSRBDataset(Dataset):
    def __init__(self, root, transforms: Compose = None, split='train'):
        super().__init__()
        root = os.path.abspath(root)
        self.transforms = transforms

        if split == 'train':
            df = pd.read_csv(os.path.join(root, "GT-training.csv"), sep=";")
            df["ClassId"] = df["ClassId"].apply(lambda x: f"{x:05d}")
            self.path_images = df.apply(lambda row: os.path.join(root, "Training", row["ClassId"], row["Filename"]), axis=1).values
        else:
            df = pd.read_csv(os.path.join(root, "GT-final_test.csv"), sep=";")
            self.path_images = df.apply(lambda row: os.path.join(root, "Final_Test", "Images", row["Filename"]), axis=1).values

        self.labels = df["ClassId"].astype(int).values
        self.stats = df["ClassId"].astype(int).value_counts().sort_index().to_dict()
        print(self.stats)
        self.categories = list(self.stats.keys())

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        img_path = self.path_images[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

if __name__ == '__main__':
    GTSRBDataset(root='../data/GTSRB', split='train')