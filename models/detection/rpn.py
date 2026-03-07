import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection._utils import BoxCoder, Matcher, BalancedPositiveNegativeSampler
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes

from models.detection.extractor import FeatureMapExtractor, FeaturePyramidNetwork

def permute_and_flatten(layer: torch.Tensor, N: int, A: int, C: int, H: int, W: int) -> torch.Tensor:
    layer = layer.view(N, A, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

def concat_box_prediction_layers(
        box_cls: list[torch.Tensor],
        box_regression: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

class RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_prediction = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, features: list[torch.Tensor]):
        logits = []
        bbox_deltas = []
        for feature in features:
            feature = F.relu(self.conv(feature))
            logits.append(self.cls_logits(feature))
            bbox_deltas.append(self.bbox_prediction(feature))
        return logits, bbox_deltas


class RegionProposalNetwork(nn.Module):
    def __init__(
            self,
            in_channels: int,
            anchors_generator: AnchorGenerator,
            # # Faster R-CNN Training
            fg_iou_thresh: float,
            bg_iou_thresh: float,
            batch_size_per_image: int,
            positive_fraction: float,
            # # Faster R-CNN Inference
            pre_nms_top_n: dict[str, int],
            post_nms_top_n: dict[str, int],
            nms_thresh: float,
            score_thresh: float = 0.0,
    ):
        super().__init__()
        self.anchor_generator = anchors_generator
        self.head = RPNHead(in_channels=in_channels, num_anchors=self.anchor_generator.num_anchors_per_location()[0])
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def get_pre_nms_top_n(self):
        if self.training:
            return self.pre_nms_top_n['training']
        return self.pre_nms_top_n['testing']

    def get_post_nms_top_n(self):
        if self.training:
            return self.post_nms_top_n['training']
        return self.post_nms_top_n['testing']

    def assign_targets_to_anchors(
            self,
            anchors: list[torch.Tensor],
            targets: list[dict[str, torch.Tensor]]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        labels = []
        matched_gt_boxes = []
        for a, t in zip(anchors, targets):
            gt_boxes = t['boxes']

            match_quality_matrix = boxes.box_iou(a, gt_boxes)

            matched_vals_max, matched_indices = torch.max(match_quality_matrix, dim=1)

            labels_per_image = torch.full_like(matched_vals_max, -1, dtype=torch.float32)

            labels_per_image[matched_vals_max >= self.fg_iou_thresh] = 1.0
            labels_per_image[matched_vals_max < self.bg_iou_thresh] = 0.0

            highest_gt, _ = torch.max(match_quality_matrix, dim=0)
            matched_gt_indices = torch.where(match_quality_matrix == highest_gt)[0]
            labels_per_image[matched_gt_indices] = 1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(gt_boxes[matched_indices])

        return labels, matched_gt_boxes

    def filter_proposals(
            self,
            anchors: list[torch.Tensor], # [torch.Size([num_anchors_per_image, 4]), ...] len=num_images
            logits: torch.Tensor, # torch.Size([num_anchors_all_images, 1])
            bbox_deltas: torch.Tensor, # torch.Size([num_anchors_all_images, 4])
            image_sizes: list[tuple[int, int]] # [(pre_pad_h_image, pre_pad_w_image), ...] len=num_images
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        scores = torch.sigmoid(logits).detach()
        bbox_deltas = bbox_deltas.detach()
        proposals = self.box_coder.decode(bbox_deltas, anchors).squeeze(1)

        num_anchors_per_image = anchors[0].shape[0]
        scores = torch.split(scores, num_anchors_per_image, dim=0)
        proposals = torch.split(proposals, num_anchors_per_image, dim=0)

        final_scores, final_proposals = [], []
        for s, p, img_size in zip(scores, proposals, image_sizes):
            p = boxes.clip_boxes_to_image(p, img_size)

            keep = boxes.remove_small_boxes(p, min_size=self.min_size)
            s, p = s[keep], p[keep]

            pre_n = min(self.get_pre_nms_top_n(), p.shape[0])
            s, indices = s.squeeze(-1).topk(pre_n, dim=0, largest=True, sorted=True)
            p = p[indices]

            keep = boxes.nms(p, s, self.nms_thresh)
            s, p = s[keep], p[keep]
            post_n = min(self.get_post_nms_top_n(), p.shape[0])
            s, p = s[:post_n], p[:post_n]

            final_scores.append(s)
            final_proposals.append(p)

        return final_scores, final_proposals

    def compute_loss(
        self,
            logits: torch.Tensor,
            pred_bbox_deltas: torch.Tensor,
            labels: list[torch.Tensor],
            regression_targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        sampled_pos_indices, sampled_neg_indices = self.fg_bg_sampler(labels)
        sampled_pos_indices = torch.where(torch.cat(sampled_pos_indices, dim=0))[0]
        sampled_neg_indices = torch.where(torch.cat(sampled_neg_indices, dim=0))[0]

        sampled_indices = torch.cat([sampled_pos_indices, sampled_neg_indices], dim=0)

        logits = logits.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_indices],
            regression_targets[sampled_pos_indices],
            beta=1 / 9,
            reduction='sum',
        ) / (sampled_indices.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(logits[sampled_indices], labels[sampled_indices])

        return objectness_loss, box_loss

    def forward(self, image_list: ImageList, features: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        logits, bbox_deltas = self.head(features)
        logits, bbox_deltas = concat_box_prediction_layers(logits, bbox_deltas)

        anchors = self.anchor_generator(image_list, features)

        scores, proposals = self.filter_proposals(anchors, logits, bbox_deltas, image_list.image_sizes)

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                logits, bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }

        return proposals, losses

if __name__ == '__main__':
    extractor = FeatureMapExtractor(
        in_channels=3,
        out_channels=64,
        depth=3,
        features='last',
        weight='../../best_checkpoint.pth'
    )

    anchor_sizes = ((32, 64, 128),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    net = RegionProposalNetwork(
        in_channels=64,
        anchors_generator=rpn_anchor_generator,
        pre_nms_top_n={
            'training': 2000,
            'testing': 1000,
        },
        post_nms_top_n={
            'training': 2000,
            'testing': 1000,
        },
        nms_thresh=0.7,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
    )

    transform = GeneralizedRCNNTransform(
        min_size=600,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )


    img1 = torch.rand(3, 800, 1360)
    img2 = torch.rand(3, 800, 1360)

    images = [img1, img2]

    targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200], [1000, 1200, 1100, 1300]], dtype=torch.float32),
            'labels': torch.tensor([0, 1], dtype=torch.int64),
        },
        {
            'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
        }
    ]

    image_list, transformed_targets = transform(images, targets)
    image_list: ImageList
    features = extractor(image_list.tensors)
    net(image_list, features, targets)


