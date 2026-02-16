import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
import pandas as pd
import os


class TT100KClassificationDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        super().__init__()
        self.transforms = transforms

        df = create_or_get_csv(root, split).reset_index(drop=True)

        categories = create_or_get_categories(root)
        self.categories = {int(k): v for k, v in categories.items()}
        self.labels_dict = {v: k for k, v in categories.items()}

        self.labels = df['category'].map(self.labels_dict).values
        self.paths = [os.path.join(root, p) for p in df['path']]
        self.bbox = df[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        xmin, ymin, xmax, ymax = self.bbox[idx]
        image = image.crop((xmin, ymin, xmax, ymax))
        label = int(self.labels[idx])

        if self.transforms:
            image = self.transforms(image)

        return image, label



def create_or_get_csv(root, split='train'):
    path_csv = os.path.join(f'{split}_objects.csv')
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
    if os.path.exists(os.path.join('.', 'categories.json')):
        with open(os.path.join('.', 'categories.json'), 'r') as f:
            return json.load(f)

    df = pd.read_csv(os.path.join('.', 'train_objects.csv'))
    categories = {i: category for i, category in enumerate(df['category'].unique())}
    with open(os.path.join('.', 'categories.json'), 'w') as f:
        json.dump(categories, f, indent=4)

    return categories


















