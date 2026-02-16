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

        df = create_or_get_csv(root, split).reset_index(drop=True)

        categories = create_or_get_categories(root)
        self.categories = {int(k): v for k, v in categories.items()}
        self.labels_dict = {v: k for k, v in categories.items()}

        self.labels = df['category'].map(self.labels_dict).values

        self.cropped_paths = []
        save_dir = os.path.abspath(os.path.join(root, f'{split}_classification'))
        os.makedirs(save_dir, exist_ok=True)

        progress_bar = tqdm(df.iterrows(), total=len(df))
        for i, row in progress_bar:
            progress_bar.set_description(f'Processing {row["image_id"]}_{i}.jpg')

            save_path = os.path.join(save_dir, f'{row["image_id"]}_{i}.jpg')
            if os.path.exists(save_path):
                continue
            img = cv2.imread(os.path.join(root, row['path']))
            h, w = img.shape[:2]
            ymin, ymax, xmin, xmax = row[['ymin', 'ymax', 'xmin', 'xmax']].values
            crop_img = img[max(0, int(ymin)):min(h, int(ymax)), max(0, int(xmin)):min(w, int(xmax)), :]

            self.cropped_paths.append(save_path)
            cv2.imwrite(save_path, crop_img)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.cropped_paths[idx])
        label = int(self.labels[idx])

        if self.transforms:
            image = self.transforms(image)

        return image, label



def create_or_get_csv(root, split='train'):
    path_csv = os.path.join(root, f'{split}_objects.csv')
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

    df = pd.read_csv(os.path.join(root, 'train_objects.csv'))
    categories = {i: category for i, category in enumerate(df['category'].unique())}
    with open(os.path.join(root, 'categories.json'), 'w') as f:
        json.dump(categories, f, indent=4)

    return categories

if __name__ == '__main__':
    TT100KClassificationDataset(root='data/tt100k', split='test')