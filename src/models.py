#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工厂：mobilenet_v3_large / repvit_m1_0（多标签分类头 = num_classes 维 logits）。

两者统一为「单头输出 num_classes 维 logits」的格式：
    * mobilenet_v3_large : torchvision 官方实现，替换最后一层 Linear 为 num_classes
    * repvit_m1_0        : 复用 RepViT-main/model/repvit.py（distillation=False 单头）

预训练加载统一为 strict=False：仅载入 backbone 特征层，分类头随机初始化；
原预训练文件不被修改，训练结果另存到 run 目录。
"""

import os
import sys

import torch
from torch import nn

MODEL_NAMES = ["mobilenet_v3_large", "mobilenet_v3_small", "repvit_m1_0", "eva_giant",
               "eva02_large", "shufflenet_v2_x1_0", "efficientnet_b3", "mobilenetv4_conv_large"]

# RepViT 源码目录（复用仓库自带实现，避免改动）
_HERE = os.path.dirname(os.path.abspath(__file__))
REPVIT_REPO = os.path.join(_HERE, "RepViT-main")

# EVA 系列（timm）模型规格：timm 模型名 + 默认输入尺寸 (H, W)。
# 选 14 的倍数（518=14*37, 616=14*44）让 patch_embed Conv(kernel=14, stride=14) 不丢像素，
# pos_embed grid 整齐为 37x44=1628。原 (512,608) 会让 floor(512/14)=36, floor(608/14)=43，
# 丢掉右下 8x6 像素且 grid 不优。
EVA_SPECS = {
    "eva_giant": {
        "timm_name": "eva_giant_patch14_336.m30m_ft_in22k_in1k",  # ImageNet top1=89.57
        "img_size": (518, 616),  # 14 倍数：37x44 patch grid
    },
    "eva02_large": {
        "timm_name": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",  # ImageNet top1=90.06
        "img_size": (518, 616),
    },
}


def _extract_state_dict(ckpt):
    """兼容两种格式：纯 state_dict，或 {'model': state_dict, ...} 检查点。"""
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        keys = list(ckpt["model"].keys())
        if keys and keys[0].startswith(("features.", "conv", "classifier.")):
            return ckpt["model"]
    return ckpt


def _load_pretrained(model, pretrained_path):
    """按「键名 + 形状」匹配载入预训练权重，返回 (loaded, skipped_keys)。

    只载入与模型同名且同形状的键（即 backbone 特征层），分类头因类别数不同自动跳过，
    从而避免 torch 对形状不匹配键直接抛错。原预训练文件不会被改动。
    """
    if pretrained_path is None:
        return 0, []
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练权重不存在: {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    model_sd = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    model.load_state_dict(filtered, strict=False)
    return len(filtered), skipped


def _build_mobilenet_v3_large(num_classes):
    from torchvision.models import mobilenet_v3_large
    model = mobilenet_v3_large(num_classes=num_classes)
    return model


def _build_mobilenet_v3_small(num_classes):
    from torchvision.models import mobilenet_v3_small
    model = mobilenet_v3_small(num_classes=num_classes)
    return model


def _build_repvit_m1_0(num_classes):
    # 复用 RepViT-main 仓库自带实现（依赖 timm 已安装）
    if REPVIT_REPO not in sys.path:
        sys.path.insert(0, REPVIT_REPO)
    from model.repvit import repvit_m1_0
    return repvit_m1_0(num_classes=num_classes, distillation=False)


def _build_shufflenet_v2_x1_0(num_classes):
    """torchvision ShuffleNetV2 x1.0（2.3M, ImageNet 69.4%），ImageNet 预训练内部加载。"""
    from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替换分类头
    return model


def _build_efficientnet_b3(num_classes):
    """torchvision EfficientNet-B3（12M, ImageNet 81.6%），ImageNet 预训练内部加载。"""
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _build_mobilenetv4_conv_large(num_classes):
    """timm MobileNetV4-Conv-Large（31M），ImageNet 预训练内部加载。"""
    import timm
    return timm.create_model("mobilenetv4_conv_large", pretrained=True, num_classes=num_classes)


def _build_eva(model_name, num_classes, img_size=None):
    """用 timm 构建 EVA-Giant / EVA-02-Large，img_size 与 dataset.INPUT_SIZE 一致，
    预训练 pos_embed 会由 timm 在 create_model 时自动插值。"""
    if model_name not in EVA_SPECS:
        raise ValueError(f"未知 EVA 模型 {model_name!r}，可选: {list(EVA_SPECS)}")
    import timm
    spec = EVA_SPECS[model_name]
    if img_size is None:
        img_size = spec["img_size"]
    model = timm.create_model(
        spec["timm_name"],
        pretrained=False,
        num_classes=num_classes,
        img_size=img_size,
    )
    return model


def build_model(model_name, num_classes, pretrained_path=None, img_size=None):
    """构建模型并（可选）从 ImageNet 预训练权重迁移，返回 (model, pretrained_path)。

    img_size 仅对 EVA 系列有效（控制 timm 构建 ViT 的 pos_embed grid 形状）；
    其他模型忽略。不传时用 EVA_SPECS 默认 (518, 616)。
    """
    if model_name == "mobilenet_v3_large":
        model = _build_mobilenet_v3_large(num_classes)
    elif model_name == "mobilenet_v3_small":
        model = _build_mobilenet_v3_small(num_classes)
    elif model_name == "repvit_m1_0":
        model = _build_repvit_m1_0(num_classes)
    elif model_name in EVA_SPECS:
        model = _build_eva(model_name, num_classes, img_size=img_size)
    elif model_name == "shufflenet_v2_x1_0":
        model = _build_shufflenet_v2_x1_0(num_classes)
    elif model_name == "efficientnet_b3":
        model = _build_efficientnet_b3(num_classes)
    elif model_name == "mobilenetv4_conv_large":
        model = _build_mobilenetv4_conv_large(num_classes)
    else:
        raise ValueError(f"未知模型 {model_name!r}，可选: {MODEL_NAMES}")

    if pretrained_path is not None:
        loaded, skipped = _load_pretrained(model, pretrained_path)
        # EVA/timm 的分类头叫 head.*；mobilenet 叫 classifier.*。两者合并统计用于日志。
        head_keys = [k for k in skipped if k.startswith(("classifier", "head."))]
        print(f"[预训练] 已从 {pretrained_path} 迁移，载入 {loaded} 个键（backbone）")
        if skipped:
            print(f"        跳过 {len(skipped)} 个键（分类头/形状不一致，需随机初始化）"
                  f"；其中分类头键 {len(head_keys)} 个")
    return model


def default_pretrained_path(model_name):
    """返回各模型默认的 ImageNet 预训练权重路径（pretrained/ 下）。"""
    if model_name == "mobilenet_v3_large":
        return os.path.join(_HERE, "Mobile-v3", "pretrained", "mobilenet_v3_large.pth")
    if model_name == "mobilenet_v3_small":
        return os.path.join(_HERE, "Mobile-v3-small", "pretrained", "mobilenet_v3_small.pth")
    if model_name == "repvit_m1_0":
        return os.path.join(REPVIT_REPO, "pretrained", "repvit_m1_0_distill_300e.pth")
    if model_name == "eva_giant":
        return os.path.join(_HERE, "EVA-Giant", "pretrained", "eva_giant.pth")
    if model_name == "eva02_large":
        return os.path.join(_HERE, "EVA-02-Large", "pretrained", "eva02_large.pth")
    # 以下模型 ImageNet 预训练在 _build 内部用 torchvision weights= / timm pretrained=True
    # 直接加载（已缓存到 ~/.cache/torch），无需 pretrained/ 文件，故返回 None。
    if model_name in ("shufflenet_v2_x1_0", "efficientnet_b3", "mobilenetv4_conv_large"):
        return None
    raise ValueError(f"未知模型 {model_name!r}")
