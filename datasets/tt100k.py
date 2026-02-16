import json

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
import pandas as pd
import os

from tqdm import tqdm


class TT100KClassificationDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        super().__init__()
        self.transforms = transforms

        root_dir_save = os.path.abspath('data/tt100k')
        os.makedirs(root_dir_save, exist_ok=True)
        save_dir = os.path.join(root_dir_save, f'{split}_classification')
        os.makedirs(save_dir, exist_ok=True)

        df = create_or_get_csv(root, root_dir_save, split).reset_index(drop=True)
        categories = create_or_get_categories(root_dir_save)
        self.labels_dict = {v: int(k) for k, v in categories.items()}
        self.labels = df['category'].map(self.labels_dict).values

        self.cropped_paths = [
            os.path.join(save_dir, f'{row["image_id"]}_{i}.jpg')
            for i, row in df.iterrows()
        ]

        progress_bar = tqdm(df.iterrows(), total=len(df), desc=f"Preparing {split} data")
        for i, row in progress_bar:
            save_path = self.cropped_paths[i]
            if os.path.exists(save_path):
                continue

            img_path = os.path.join(root, row['path'])
            img = cv2.imread(img_path)
            if img is None: continue

            h_orig, w_orig = img.shape[:2]
            ymin, ymax, xmin, xmax = row[['ymin', 'ymax', 'xmin', 'xmax']].values

            crop_img = img[
                max(0, int(ymin)):min(h_orig, int(ymax)),
                max(0, int(xmin)):min(w_orig, int(xmax))
            ]

            cv2.imwrite(save_path, crop_img)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.cropped_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label



def create_or_get_csv(root, root_data, split='train'):
    path_csv = os.path.join(root_data, f'{split}_classification.csv')
    path_json = os.path.join(root, f'{split}.json')
    if os.path.exists(path_csv):
        return pd.read_csv(path_csv)

    with open(path_json, 'r') as f:
        data = json.load(f)

    rows = []
    for image_id, image_data in data.items():
        path = image_data['path']

        for obj in image_data["objects"]:
            bbox = obj['bbox']
            rows.append({
                'image_id': int(image_id),
                'path': path,
                'category': obj['category'],
                'xmin': bbox['xmin'],
                'ymin': bbox['ymin'],
                'xmax': bbox['xmax'],
                'ymax': bbox['ymax'],
            })

    df = pd.DataFrame(rows).sort_values(by=['category'])
    df.to_csv(path_csv, index=False)
    return df

def create_or_get_categories(root):
    if os.path.exists(os.path.join(root, 'categories.json')):
        with open(os.path.join(root, 'categories.json'), 'r') as f:
            return json.load(f)

    df = pd.read_csv(os.path.join(root, 'train_classification.csv'))
    categories = {i: category for i, category in enumerate(df['category'].unique())}
    with open(os.path.join(root, 'categories.json'), 'w') as f:
        json.dump(categories, f, indent=4)

    return categories

if __name__ == '__main__':
    TT100KClassificationDataset(root='data/tt100k', split='test')