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
    checkpoint = torch.load('../../best_checkpoint.pth', map_location=torch.device('cpu'))
    keys_to_exclude = ['classifier.classifier.head.weight', 'classifier.classifier.head.bias']
    state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k not in keys_to_exclude}
    model = MambaClassifier(dims=3, depth=4, num_classes=151)
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        print(name, param.requires_grad)