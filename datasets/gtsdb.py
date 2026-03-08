from PIL import Image
from timm.data import ToTensor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict
from pprint import pprint
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
import cv2
import torch


class GTSDBDataset(Dataset):
    def __init__(self, root, split='train', test_size: float = 0.1):
        with open(os.path.join(root, 'gt.txt')) as f:
            lines = f.readlines()
        lines = list(map(lambda x: x.strip().split(';'), lines))

        annotation = defaultdict(dict)
        for obj in lines:
            annotation[obj[0]]['path'] = os.path.join(root, obj[0])

            if 'boxes' not in annotation[obj[0]]:
                annotation[obj[0]]['boxes'] = []
            annotation[obj[0]]['boxes'].append(list(map(int, obj[1:-1])))

            if 'labels' not in annotation[obj[0]]:
                annotation[obj[0]]['labels'] = []
            annotation[obj[0]]['labels'].append(int(obj[-1]))

        annotation = [v for k, v in annotation.items()]

        Y = []

        for item in annotation:
            vec = [0] * 43
            for l in item["labels"]:
                vec[l] = 1
            Y.append(vec)
        Y = np.array(Y)

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=42
        )

        train_idx, test_idx = next(msss.split(annotation, Y))
        train_data = [annotation[i] for i in train_idx]
        test_data = [annotation[i] for i in test_idx]

        if split == 'train':
            self.data = train_data
        elif split == 'test':
            self.data = test_data
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]['path']
        labels = torch.as_tensor(self.data[idx]['labels'], dtype=torch.int64)
        target = {
            'boxes': torch.as_tensor(self.data[idx]['boxes'], dtype=torch.float32),
            'labels': labels + 1,
        }
        image = Image.open(path)
        image = image.convert('RGB')
        image = ToTensor()(image)

        return image, target

if __name__ == '__main__':
    root = '/Users/tuan-nv0505/Projects/Personal/AI/CV/Traffic-Sign-Detection/data/FullIJCNN2013'
    dataset = GTSDBDataset(root)
    for i in range(len(dataset)):
        path, target = dataset[i]
        img = cv2.imread(path)
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = list(map(int, box.numpy().tolist()))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(img, str(label.item()), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        cv2.imshow("image", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()