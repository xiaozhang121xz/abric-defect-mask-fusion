#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多标签图像分类 Dataset 与数据增强。

数据格式(data/ai-zzdn-class/):
    images/        图像(原始文件名)
    labels.json    {图片stem: [类别idx, ...]}，空列表 = 非瑕疵
    train.txt / val.txt   图像列表(每行 ./images/<stem>.jpg)

数据增强（训练）：
    RandomResizedCrop(608x512, scale=(0.8,1.0)) -> RandomHorizontalFlip -> ColorJitter
    -> [目标检测增强桥] -> ToTensor -> Normalize(ImageNet)
    其中「目标检测增强桥」复刻 detect_yolo/train.py 的 8 种 albumentations
    + YOLO 内置 RandomPerspective / VerticalFlip（RandomHSV 已停用）；
    Mosaic(4 图拼接) 在 Dataset 取数层实现，默认关闭（train.py --mosaic 开启，close_mosaic 见 train.py）。
"""

import os
import random

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
try:
    import albumentations as A
except ModuleNotFoundError as exc:
    if exc.name != 'albumentations':
        raise
    A = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 目标检测自定义 albumentations（与 detect_yolo/train.py 的 augmentations 逐条一致）
DETECT_ALBUMENTATIONS = [
    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.1),
    A.GaussNoise(std_range=(0.01, 0.03), p=0.1),
    A.GaussianBlur(blur_limit=(3, 3), p=0.1),
    A.Sharpen(alpha=(0.05, 0.15), lightness=(0.9, 1.1), p=0.05),
    A.ImageCompression(quality_range=(85, 100), p=0.10),
    A.RGBShift(r_shift_limit=0.01, g_shift_limit=0.01, b_shift_limit=0.01, p=0.01),
    A.MotionBlur(blur_limit=3, p=0.1),
    A.RandomGamma(gamma_limit=(90, 110), p=0.1),
] if A is not None else []

# YOLO 内置 RandomHSV 参数（与 detect args.yaml 一致：h=0.005, s=0.7, v=0.4）
YOLO_HSV = {"h": 0.005, "s": 0.7, "v": 0.4}

# 模型输入尺寸（宽 608 x 高 512 -> 记作 (H, W)）。RandomResizedCrop 直接裁到该尺寸。
INPUT_SIZE = (512, 608)        # (H, W) 模型输入尺寸

# 验证/单图推理预处理：先把 2448x2048 中心裁剪成 2432x2048（去掉左右各 8px 边缘），
# 再 resize 到 INPUT_SIZE；2432/2048 == 608/512 == 19/16，中心裁剪后宽高比与输入一致。
VAL_CROP_SIZE = (2048, 2432)   # (H, W) 验证中心裁剪尺寸（宽 2432 x 高 2048）


def _yolo_random_hsv(image, **kwargs):
    """复刻 ultralytics RandomHSV.apply_image（BGR 上的 LUT：H 加性、S/V 乘性）。"""
    h, s, v = YOLO_HSV["h"], YOLO_HSV["s"], YOLO_HSV["v"]
    r = np.random.uniform(-1, 1, 3) * [h, s, v]
    x = np.arange(0, 256, dtype=np.float32)
    lut_hue = ((x + r[0] * 180) % 180).astype(np.uint8)
    lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(np.uint8)
    lut_sat[0] = 0  # 防止纯白改变颜色（ultralytics 8.3.79+ 引入）
    lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hue, sat, val = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    bgr = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class _AlbumentationsAug:
    """PIL -> numpy -> albumentations -> uint8 numpy -> PIL，把检测增强桥插入 torchvision 流程。"""

    def __init__(self, compose):
        self.compose = compose

    def __call__(self, img):
        arr = np.array(img)  # PIL RGB -> HxWx3 uint8
        out = self.compose(image=arr)["image"]
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)


def _build_detect_aug(include_affine=True):
    """目标检测增强桥（与 detect_yolo/train.py 的 albumentations + YOLO flipud 逐条一致）。

    include_affine=False 时去掉 A.Affine（用于 resize_pad：ResizeRatioPad 已做几何归一化，
    再做 scale/translate 平移意义有限，且在 1024 画布上效果重复）。
    """
    if A is None:
        raise RuntimeError('旧版检测增强需要 albumentations')
    augs = []
    if include_affine:
        # YOLO RandomPerspective：degrees/shear/perspective=0，等价缩放+平移，填充(114,114,114)
        # 默认值已调：scale 0.5（±50%），0.1 概率触发
        augs.append(A.Affine(scale=(0.5, 1.5), translate_percent=(-0.1, 0.1), rotate=0, shear=0,
                             border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=0.1))
    augs.extend(DETECT_ALBUMENTATIONS)
    augs.append(A.VerticalFlip(p=0.1))  # YOLO flipud；fliplr 由上面的 RandomHorizontalFlip 覆盖
    return _AlbumentationsAug(A.Compose(augs))


class ResizeRatioPad:
    """按比例缩放到 size 框内（保持长宽比），贴到全黑画布左上角。

    与 preview_segres_resize_pad.py 的 resize_ratio_pad 一致：作为「几何预处理」步骤
    替换 RandomResizedCrop（训练）/ CenterCrop+Resize（验证），消除固定裁剪对非标准长宽比的失真，
    同时让训练与部署共用同一确定性的几何映射。
    """

    def __init__(self, size):
        # size: (H, W)，与 INPUT_SIZE 约定一致
        self.out_h, self.out_w = size

    def __call__(self, img):
        w, h = img.size
        scale = min(self.out_w / w, self.out_h / h)
        new_w = max(int(round(w * scale)), 1)
        new_h = max(int(round(h * scale)), 1)
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.out_w, self.out_h), (0, 0, 0))  # 全黑画布
        canvas.paste(resized, (0, 0))                                 # 左上角对齐
        return canvas


def build_transforms(train=True, method="crop"):
    """训练/验证图像预处理（ImageNet 归一化，与预训练一致）。

    method:
        - "crop"（默认，原方法，保留）：训练 RandomResizedCrop(608x512, 0.8~1.0)；
          验证 CenterCrop(2432x2048) -> Resize(608x512)。
        - "resize_pad"（新方法）：训练/验证统一 ResizeRatioPad(608x512)（按比例缩放+黑边贴左上角，
          与 preview_segres_resize_pad.py 一致）；训练额外保留 HFlip/ColorJitter/检测增强桥。
    实际尺寸由 INPUT_SIZE / VAL_CROP_SIZE 常量决定。
    """
    if method not in ("crop", "resize_pad"):
        raise ValueError(f"未知预处理方法 {method!r}，可选: crop / resize_pad")

    # ---- 新方法：按比例缩放 + 黑边 ----
    if method == "resize_pad":
        if train:
            # 顺序：随机增强（在原图上）→ ResizeRatioPad（几何归一化）→ ToTensor → Normalize
            # 不含 A.Affine（include_affine=False），因 ResizeRatioPad 已做几何归一化
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                _build_detect_aug(include_affine=False),
                ResizeRatioPad(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        return transforms.Compose([
            ResizeRatioPad(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # ---- 原方法（保留，后续仍会使用）----
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            _build_detect_aug(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.CenterCrop(VAL_CROP_SIZE),
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_stems(txt_path):
    """读取 train.txt/val.txt，返回 stem 列表（去除 './images/' 前缀与扩展名）。"""
    stems = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.append(os.path.splitext(os.path.basename(line))[0])
    return stems


class MultiLabelDataset(Dataset):
    """多标签图像分类数据集。label 为 num_classes 维 multi-hot 向量。

    mosaic=True 时，以 mosaic_prob 概率将 4 张图拼成 2×2 大图（标签取并集），
    复刻检测的 Mosaic 增强；训练末期由 train.py 关闭（close_mosaic）。
    """

    def __init__(self, image_dir, stems, labels, num_classes, transform=None,
                 imgsz=INPUT_SIZE, mosaic=False, mosaic_prob=0.001, image_suffix=".jpg"):
        self.image_dir = image_dir
        self.stems = stems
        self.labels = labels          # dict: stem -> [idx, ...]
        self.num_classes = num_classes
        self.transform = transform
        self.imgsz = imgsz
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.image_suffix = image_suffix  # 默认 ".jpg"；segres 图源用 "_segres.jpg"

    def __len__(self):
        return len(self.stems)

    def _img_path(self, stem):
        return os.path.join(self.image_dir, stem + self.image_suffix)

    def _load_image_safe(self, stem):
        """读图，损坏(broken stream/截断)则换随机 stem 重试(最多 10 次)。
        返回 (img, stem)：坏图的 stem 被替换为有效 stem，target 按有效 stem 取。
        兜底仍失败时返回纯色图（极罕见，避免训练崩）。"""
        for _ in range(10):
            try:
                return Image.open(self._img_path(stem)).convert("RGB"), stem
            except (OSError, Image.UnidentifiedImageError):
                stem = self.stems[random.randrange(len(self.stems))]
        return Image.new("RGB", tuple(self.imgsz[::-1])), stem

    def _load(self, index):
        stem = self.stems[index]
        img, stem = self._load_image_safe(stem)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for idx in self.labels.get(stem, []):
            target[idx] = 1.0
        return img, target

    def _mosaic(self, index):
        h, w = self.imgsz  # (H, W) 画布尺寸，默认 INPUT_SIZE
        sh, sw = h // 2, w // 2
        indices = [index] + [random.randrange(len(self.stems)) for _ in range(3)]
        canvas = Image.new("RGB", (w, h))
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for k, i in enumerate(indices):
            stem = self.stems[i]
            img, stem = self._load_image_safe(stem)
            canvas.paste(img.resize((sw, sh)), ((k % 2) * sw, (k // 2) * sh))
            for idx in self.labels.get(stem, []):
                target[idx] = 1.0
        return canvas, target

    def __getitem__(self, index):
        if self.mosaic and random.random() < self.mosaic_prob:
            img, target = self._mosaic(index)
        else:
            img, target = self._load(index)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# ---- 单标签口径（正常显式 + 互斥 softmax）----
# 优先级：经向(0) > 纬向(1) > 面积(2) > 正常(num_defect_classes)
NORMAL_IDX = -1  # 占位，由 SingleLabelDataset 在构造时定为 num_defect_classes（最后一个类）


def relabel_single(label_list, normal_idx):
    """多标签 idx 列表 -> 单标签 int（优先级 经向>纬向>面积>正常）。

    normal_idx = 瑕疵类数（即正常类的 idx，约定为最后一个类）。
    """
    s = set(label_list or [])
    if 0 in s:
        return 0
    if 1 in s:
        return 1
    if 2 in s:
        return 2
    return normal_idx


class SingleLabelDataset(Dataset):
    """单标签互斥分类数据集。

    label 为单个 int（正常 = normal_idx，约定为最后一个类）。从原多标签 labels.json
    按优先级 经向>纬向>面积>正常 重标。Mosaic 标签 = 拼图各自优先级后再取优先级。

    epoch_sample={类idx: 每轮上限} 时启用每轮重抽样（如拍摄二分类的正常类 1:3000）：
    __init__ 立即按其抽一次，train.py 每轮 epoch 开始再调 resample() 换一批；
    未配置时 stems 即固定全量，行为与原来完全一致。
    """

    def __init__(self, image_dir, stems, labels, num_defect_classes, normal_idx,
                 transform=None, imgsz=INPUT_SIZE, mosaic=False, mosaic_prob=0.001,
                 image_suffix=".jpg", epoch_sample=None):
        self.image_dir = image_dir
        self.stems = stems
        self.labels = labels                 # dict: stem -> [idx, ...]（原多标签）
        self.num_classes = num_defect_classes + 1  # 瑕疵类 + 正常
        self.normal_idx = normal_idx
        self.transform = transform
        self.imgsz = imgsz
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.image_suffix = image_suffix
        self.epoch_sample = epoch_sample      # {类idx: 每轮最多取多少张}；None=固定全量
        # 预算每张图的单标签，避免重复计算（对全量池预算，重抽样后仍覆盖）
        self._pool = list(stems)              # 全量池（resample 的抽样来源）
        self._single = {s: relabel_single(labels.get(s, []), normal_idx) for s in stems}
        if self.epoch_sample:
            self.resample()

    def resample(self):
        """按 epoch_sample 从全量池重新抽各类样本（train.py 每轮调用一次）。

        某类样本数不足上限时取全部；DataLoader 默认 persistent_workers=False，
        每轮 iter() 重新 fork worker，更新后的 self.stems 对新 epoch 生效。
        """
        if not self.epoch_sample:
            self.stems = list(self._pool)
            return
        by_cls = {}
        for s in self._pool:
            by_cls.setdefault(self._single[s], []).append(s)
        picked = []
        for cls, lst in sorted(by_cls.items()):
            k = self.epoch_sample.get(cls)
            picked.extend(random.sample(lst, k) if k is not None and k < len(lst) else lst)
        self.stems = picked

    def __len__(self):
        return len(self.stems)

    def _img_path(self, stem):
        return os.path.join(self.image_dir, stem + self.image_suffix)

    def _load(self, index):
        stem = self.stems[index]
        img = Image.open(self._img_path(stem)).convert("RGB")
        target = self._single[stem]
        return img, target

    def _mosaic(self, index):
        h, w = self.imgsz
        sh, sw = h // 2, w // 2
        indices = [index] + [random.randrange(len(self.stems)) for _ in range(3)]
        canvas = Image.new("RGB", (w, h))
        labs = []
        for k, i in enumerate(indices):
            stem = self.stems[i]
            img = Image.open(self._img_path(stem)).convert("RGB")
            canvas.paste(img.resize((sw, sh)), ((k % 2) * sw, (k // 2) * sh))
            labs.append(self._single[stem])
        # 拼图整体标签：取所含子图优先级最高的瑕疵类
        s = set(labs) - {self.normal_idx}
        target = min(s) if s else self.normal_idx
        return canvas, target

    def __getitem__(self, index):
        if self.mosaic and random.random() < self.mosaic_prob:
            img, target = self._mosaic(index)
        else:
            img, target = self._load(index)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
