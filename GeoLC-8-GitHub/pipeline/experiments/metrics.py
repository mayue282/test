from __future__ import annotations

import torch
import torch.nn.functional as F


class SegmentationMetrics:
    def __init__(self, classes=8, device="cpu"):
        self.classes = classes
        self.confusion = torch.zeros(classes, classes, dtype=torch.int64, device=device)
        # Apple MPS does not implement float64 tensors.
        self.boundary_tp = torch.zeros((), dtype=torch.float32, device=device)
        self.boundary_fp = torch.zeros((), dtype=torch.float32, device=device)
        self.boundary_fn = torch.zeros((), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(1)
        valid = (target >= 0) & (target < self.classes)
        ids = self.classes * target[valid] + pred[valid]
        self.confusion += torch.bincount(ids, minlength=self.classes ** 2).reshape(self.classes, self.classes)

        # Class-agnostic boundary F1 with a two-pixel tolerance.
        p_boundary = self._boundary(pred)
        t_boundary = self._boundary(target)
        p_dilate = F.max_pool2d(p_boundary.float()[:, None], 5, 1, 2)[:, 0].bool()
        t_dilate = F.max_pool2d(t_boundary.float()[:, None], 5, 1, 2)[:, 0].bool()
        self.boundary_tp += (p_boundary & t_dilate).sum()
        self.boundary_fp += (p_boundary & ~t_dilate).sum()
        self.boundary_fn += (t_boundary & ~p_dilate).sum()

    @staticmethod
    def _boundary(mask):
        result = torch.zeros_like(mask, dtype=torch.bool)
        result[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
        result[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
        return result

    def compute(self):
        cm = self.confusion.float()
        tp = cm.diag()
        union = cm.sum(0) + cm.sum(1) - tp
        iou = tp / union.clamp_min(1)
        f1 = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp_min(1)
        oa = tp.sum() / cm.sum().clamp_min(1)
        bf1 = 2 * self.boundary_tp / (2 * self.boundary_tp + self.boundary_fp + self.boundary_fn).clamp_min(1)
        return {
            "per_class_iou": [round(v.item(), 6) for v in iou],
            "miou": round(iou.mean().item(), 6),
            "mean_f1": round(f1.mean().item(), 6),
            "overall_accuracy": round(oa.item(), 6),
            "boundary_f1": round(bf1.item(), 6),
        }
