import torch
import torch.nn as nn
from torchinfo import summary

from models.backbone.backbone import BackBone
from models.classification.classifier import Classifier

class MambaClassifier(nn.Module):
    def __init__(self, dims=3, depth=3, num_classes=43, ssm_d_state=16):
        super().__init__()
        self.backbone = BackBone(dims=3, depth=depth, ssm_d_state=ssm_d_state)
        self.num_features = self.backbone.num_features
        self.classifier = Classifier(num_features=self.num_features, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 608, 1024)
    model = MambaClassifier(dims=3, depth=4, ssm_d_state=16, num_classes=43)
    summary(model, input_size=(1, 3, 608, 1024))