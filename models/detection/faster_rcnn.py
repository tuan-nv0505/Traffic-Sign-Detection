import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from pprint import pprint

from models.detection.extractor import FeatureMapExtractor
from models.detection.rpn import RegionProposalNetwork, RPNHead


class FasterRCNN(nn.Module):
    def __init__(
            self,
            num_classes=None,
            weight=None,
            # transform parameters
            min_size=600,
            max_size=1333,
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            # RPN parameters
            rpn_anchor_generator=None,
            rpn_head=None,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            rpn_score_thresh=0.0,
            # Box parameters
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,
            bbox_reg_weights=None,
    ):
        super().__init__()

        self.extractor = FeatureMapExtractor(in_channels=3, features='last', out_channels=64, depth=3)

        if rpn_anchor_generator is None:
            anchor_sizes = ((32, 64, 128),)
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(
            in_channels=self.extractor.out_channels,
            anchors_generator=rpn_anchor_generator,
            fg_iou_thresh=rpn_fg_iou_thresh,
            bg_iou_thresh=rpn_bg_iou_thresh,
            batch_size_per_image=rpn_batch_size_per_image,
            positive_fraction=rpn_positive_fraction,
            pre_nms_top_n=rpn_pre_nms_top_n,
            post_nms_top_n=rpn_post_nms_top_n,
            nms_thresh=rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        box_roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        box_head = TwoMLPHead(self.extractor.out_channels * 7 ** 2, 1024)
        box_predictor = FastRCNNPredictor(1024, num_classes)

        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=box_fg_iou_thresh,
            bg_iou_thresh=box_bg_iou_thresh,
            batch_size_per_image=box_batch_size_per_image,
            positive_fraction=box_positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            score_thresh=box_score_thresh,
            nms_thresh=box_nms_thresh,
            detections_per_img=box_detections_per_img
        )

        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    def forward(self, images, targets):
        original_image_sizes = [img.shape[-2:] for img in images]

        images, targets = self.transform(images, targets)

        features = self.extractor(images.tensors)

        proposals, rpn_losses = self.rpn(images, features, targets)

        if isinstance(features, torch.Tensor):
            features = {"0": features}
        elif isinstance(features, list):
            features = {str(i): f for i, f in enumerate(features)}

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(rpn_losses)

        if self.training:
            return losses
        return detections


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4: x = x.flatten(start_dim=1)
        return self.cls_score(x), self.bbox_pred(x)

if __name__ == '__main__':
    # img1 = torch.rand(3, 800, 1360)
    # img2 = torch.rand(3, 800, 1360)
    #
    # images = [img1, img2]
    #
    # targets = [
    #     {
    #         'boxes': torch.tensor([[100, 100, 200, 200], [1000, 1200, 1100, 1300]], dtype=torch.float32),
    #         'labels': torch.tensor([0, 1], dtype=torch.int64),
    #     },
    #     {
    #         'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
    #         'labels': torch.tensor([1], dtype=torch.int64),
    #     }
    # ]
    net = FasterRCNN(num_classes=44)
    # out = net(images, targets)
    # pprint(out)