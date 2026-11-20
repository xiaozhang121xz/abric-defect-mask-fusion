"""Paired inputs for the fabric MobileNetV3 experiment."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

ROOT = Path(os.environ.get("FABRIC_WORK_ROOT", "data")).resolve()
DATA3 = Path(os.environ.get("FABRIC_ARTIFACT_ROOT", "artifacts")).resolve()
HERE = Path(__file__).resolve().parent
MODES = {"raw": 3, "enhanced": 1, "raw_enhanced": 4}
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
GRAY_MEAN = (0.5,)
GRAY_STD = (0.25,)

def inside(path, strict=True, allow_data3=False):
    # Reject symlinks before resolution, including links that point outside zx.
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    roots = (ROOT, DATA3) if allow_data3 else (ROOT,)
    base = next((r for r in roots if p.is_relative_to(r)), None)
    if base is None or ".." in p.parts:
        raise ValueError(f"路径不在允许的项目目录内: {p}")
    if base.is_symlink():
        raise ValueError(f"项目根目录不能是软链接: {base}")
    cursor = base
    for part in p.relative_to(base).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"不使用软链接: {cursor}；请指定 zx 内的实际路径")
    if strict and not p.exists():
        raise FileNotFoundError(p)
    return p

def read_inputs(data_dir):
    data = inside(data_dir, allow_data3=True)
    labels = json.loads(inside(data / "labels.json", allow_data3=True).read_text(encoding="utf-8"))
    meta = json.loads(inside(data / "classes.json", allow_data3=True).read_text(encoding="utf-8"))
    if meta["nc"] != 3 or len(meta["names"]) != 3:
        raise ValueError("预期 3 个缺陷标签")
    splits = {}
    for tag in ("train", "val"):
        lines = inside(data / (tag + ".txt"), allow_data3=True).read_text(encoding="utf-8").splitlines()
        stems = [Path(row.strip()).stem for row in lines if row.strip()]
        if not stems or len(stems) != len(set(stems)):
            raise ValueError(f"{tag} 清单为空或有重复 ID")
        for stem in stems:
            if not stem or stem in (".", "..") or stem not in labels:
                raise ValueError(f"{tag} 标签缺失或 ID 不合法: {stem}")
            ids = labels[stem]
            if not isinstance(ids, list) or any(type(k) is not int or k not in range(3) for k in ids):
                raise ValueError(f"{tag} 标签非法: {stem}")
        splits[tag] = stems
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("train 和 val 有重复 ID")
    return labels, meta["names"], splits

def rgb_decode(raw):
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")

def require_guided():
    if not hasattr(getattr(cv2, "ximgproc", None), "guidedFilter"):
        raise RuntimeError("当前 OpenCV 缺少 cv2.ximgproc.guidedFilter，请使用带 contrib 的环境")

def make_enhanced(raw, max_side):
    require_guided()
    from fabric_enhance import best
    cv2.setNumThreads(1)
    gray = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("OpenCV 灰度解码失败")
    h, w = gray.shape
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    out = best(gray)
    if out.ndim != 2 or out.dtype != np.uint8:
        raise ValueError(f"best() 输出不符合单通道 uint8: {out.shape}, {out.dtype}")
    return out, (w, h)

class PairedFabric(Dataset):
    def __init__(self, image_dir, stems, labels, mode, train=False, size=(512, 608),
                 max_side=2048, cache=None):
        self.image_dir = inside(image_dir)
        self.stems, self.labels = stems, labels
        self.mode, self.train, self.size = mode, train, tuple(size)
        self.max_side = max_side
        self.cache = inside(cache, strict=False) if cache else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
            self.cache_tag = hashlib.sha256(
                inside(HERE / "fabric_enhance.py").read_bytes()
                + f"|{cv2.__version__}|{max_side}".encode()).digest()

    def __len__(self):
        return len(self.stems)

    def enhancement(self, raw, rgb_size):
        path = None
        if self.cache:
            key = hashlib.sha256(self.cache_tag + raw).hexdigest()
            path = inside(self.cache / (key + ".png"), strict=False)
            if path.is_file():
                with Image.open(path) as image:
                    if image.mode != "L":
                        raise ValueError(f"增强缓存不是灰度: {path}")
                    return image.copy()
        out, original_size = make_enhanced(raw, self.max_side)
        if original_size != rgb_size:
            raise ValueError(f"RGB/灰度解码尺寸不同: {rgb_size} vs {original_size}")
        # Map the enhancement back onto the source grid before applying the same crop.
        image = Image.fromarray(out)
        if image.size != rgb_size:
            image = F.resize(image, [rgb_size[1], rgb_size[0]], antialias=True)
        if path:
            with tempfile.NamedTemporaryFile(dir=self.cache, suffix=".png", delete=False) as fp:
                tmp = Path(fp.name)
            try:
                image.save(tmp, format="PNG")
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return image

    def pair(self, stem):
        raw = inside(self.image_dir / (stem + ".jpg")).read_bytes()
        rgb = rgb_decode(raw)
        enhanced = self.enhancement(raw, rgb.size) if self.mode != "raw" else None
        if enhanced is not None and enhanced.size != rgb.size:
            raise ValueError(f"{stem} RGB/增强图尺寸不一致")
        return rgb, enhanced

    def __getitem__(self, index):
        stem = self.stems[index]
        rgb, enhanced = self.pair(stem)
        if self.train:
            crop = transforms.RandomResizedCrop.get_params(rgb if rgb is not None else enhanced, (0.8, 1.0), (3/4, 4/3))
            hflip = bool(torch.rand(()) < 0.5)
            vflip = bool(torch.rand(()) < 0.1)
            def geometry(image):
                image = F.resized_crop(image, *crop, self.size, antialias=True)
                if hflip:
                    image = F.hflip(image)
                if vflip:
                    image = F.vflip(image)
                return image
        else:
            def geometry(image):
                return F.resize(F.center_crop(image, [2048, 2432]), self.size, antialias=True)
        parts = []
        if self.mode != "enhanced":
            parts.append(F.normalize(F.to_tensor(geometry(rgb)), RGB_MEAN, RGB_STD))
        if enhanced is not None:
            parts.append(F.normalize(F.to_tensor(geometry(enhanced)), GRAY_MEAN, GRAY_STD))
        x = torch.cat(parts, dim=0)
        y = torch.zeros(3, dtype=torch.float32)
        y[self.labels[stem]] = 1
        return x, y

class PrecomputedFabric(PairedFabric):
    """Read the saved enhancement by stem. Never fall back to online processing."""
    def __init__(self, image_dir, stems, labels, mode, train=False, size=(512, 608),
                 enhanced_dir=None):
        if mode not in MODES:
            raise ValueError(f"未知输入模式: {mode}")
        self.stems, self.labels = stems, labels
        self.mode, self.train, self.size = mode, train, tuple(size)
        self.image_dir = inside(image_dir) if mode in ("raw", "raw_enhanced") else None
        self.enhanced_dir = (inside(enhanced_dir, allow_data3=True)
                             if mode in ("enhanced", "raw_enhanced") else None)
        for folder in (self.image_dir, self.enhanced_dir):
            if folder is not None and not folder.is_dir():
                raise NotADirectoryError(folder)

    def pair(self, stem):
        if not stem or Path(stem).name != stem or stem in (".", ".."):
            raise ValueError(f"图片 ID 不合法: {stem!r}")
        rgb = None
        enhanced = None
        if self.image_dir is not None:
            rgb = rgb_decode(inside(self.image_dir / (stem + ".jpg")).read_bytes())
        if self.enhanced_dir is not None:
            path = inside(self.enhanced_dir / (stem + ".png"), strict=False, allow_data3=True)
            if not path.is_file():
                raise FileNotFoundError(f"缺少预生成增强图，未在线补算: {path}")
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "L":
                    raise ValueError(f"预生成增强图必须是单通道 PNG: {path}")
                enhanced = image.copy()
            if rgb is not None and enhanced.size != rgb.size:
                raise ValueError(f"{stem} RGB/增强图尺寸不一致: {rgb.size} vs {enhanced.size}")
        return rgb, enhanced


def adapt_stem(model, mode):
    channels = MODES[mode]
    if channels == 3:
        return model
    old = model.features[0][0]
    new = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride, old.padding,
                    old.dilation, old.groups, bias=old.bias is not None,
                    padding_mode=old.padding_mode, device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        if channels == 1:
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        else:
            new.weight.zero_()
            new.weight[:, :3].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.features[0][0] = new
    return model
