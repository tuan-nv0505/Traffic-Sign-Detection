import torch
import torch.nn as nn
from torchinfo import summary

from models.backbone.backbone import BackBone
from models.classification.classifier import Classifier

class MambaClassifier(nn.Module):
    def __init__(self, dims=3, depth=4, num_classes=43):
        super().__init__()
        self.backbone = BackBone(dims=3, depth=depth)
        self.num_features = self.backbone.num_features
        self.classifier = Classifier(num_features=self.num_features, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 48, 48)
    model = MambaClassifier(dims=3, depth=4, num_classes=43)

    # 2. Load checkpoint
    checkpoint = torch.device('cpu')
    checkpoint = torch.load('../../final_best_checkpoint.pth', map_location=checkpoint)

    # Lấy state_dict từ checkpoint
    old_state_dict = checkpoint['model_state_dict']
    new_state_dict = {}

    # 3. Ánh xạ lại tên các layer
    for k, v in old_state_dict.items():
        # Nếu key bắt đầu bằng classifier, đổi classifier. -> classifier.classifier.
        if k.startswith('classifier.'):
            new_key = k.replace('classifier.', 'classifier.classifier.')
        # Nếu không, thêm tiền tố backbone. vào trước các phần khác (pre_embd, layers, out_norm...)
        else:
            new_key = f'backbone.{k}'

        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)