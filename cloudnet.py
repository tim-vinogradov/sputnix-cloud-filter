"""CloudNet: нейросеть для попиксельной оценки облачности на обзорных снимках СПУТНИКС.

Архитектура: энкодер ResNet-18 (предобучен на ImageNet) + лёгкий декодер U-Net.
Выход: карта вероятности облака для каждого пикселя (0..1).
Процент облачности снимка = доля пикселей Земли, отнесённых к облакам
(вероятность облака >= 0.5). Пиксели космоса у горизонта не учитываются.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

INPUT_W, INPUT_H = 384, 288          # размер, к которому приводится снимок 4:3
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class CloudNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # 1/2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.up3 = UpBlock(512, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 48)
        self.up0 = UpBlock(48, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)              # 1/2
        s1 = self.l1(self.pool(s0))    # 1/4
        s2 = self.l2(s1)               # 1/8
        s3 = self.l3(s2)               # 1/16
        s4 = self.l4(s3)               # 1/32
        d = self.up3(s4, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        d = self.up0(d, s0)
        logits = self.head(d)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def load_image(path):
    """Читает снимок, возвращает RGB-массив uint8 размера INPUT_H x INPUT_W."""
    im = Image.open(path).convert("RGB").resize((INPUT_W, INPUT_H), Image.BILINEAR)
    return np.asarray(im)


def to_tensor(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr).copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return (t - MEAN) / STD


def space_mask(arr):
    """Пиксели космоса (чёрное небо у горизонта) не участвуют в расчёте облачности."""
    return arr.max(-1) < 30


class CloudEstimator:
    """Обёртка для применения обученной модели к снимкам."""

    def __init__(self, weights_path, device="cpu", threshold=0.5):
        self.model = CloudNet(pretrained=False)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        state = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval().to(device)
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def prob_map(self, arr):
        x = to_tensor(arr).to(self.device)
        # усреднение по отражениям (test-time augmentation)
        p = torch.sigmoid(self.model(x))
        p = p + torch.flip(torch.sigmoid(self.model(torch.flip(x, [3]))), [3])
        p = p + torch.flip(torch.sigmoid(self.model(torch.flip(x, [2]))), [2])
        return (p / 3)[0, 0].cpu().numpy()

    def estimate(self, path):
        arr = load_image(path)
        prob = self.prob_map(arr)
        valid = ~space_mask(arr)
        mask = (prob >= self.threshold) & valid
        percent = 100.0 * mask.sum() / max(1, valid.sum())
        return percent, mask, prob, arr
