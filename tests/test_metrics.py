# tests/test_metrics.py
import torch
import pytest
from utils.metrics import accuracy, compute_iou, AverageMeter

def test_accuracy():
    pred = torch.tensor([0, 1, 2, 3, 0])
    target = torch.tensor([0, 1, 2, 2, 1])
    acc = accuracy(pred, target)
    assert abs(acc - 60.0) < 1e-5

def test_compute_iou():
    pred = torch.tensor([0, 0, 1, 1, 2, 2])
    target = torch.tensor([0, 1, 1, 1, 2, 0])
    iou = compute_iou(pred, target, num_classes=3)
    assert len(iou) == 3
    assert abs(iou[0] - 1 / 3) < 1e-5
    assert abs(iou[1] - 2 / 3) < 1e-5
    assert abs(iou[2] - 1 / 2) < 1e-5

def test_average_meter():
    meter = AverageMeter()
    meter.update(1.0)
    meter.update(3.0)
    assert meter.avg == 2.0
    assert meter.count == 2
    meter.reset()
    assert meter.avg == 0.0
