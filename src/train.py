#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
布面瑕疵图像分类(多标签)训练脚本。

- 从 ImageNet 预训练模型(各自 pretrained/ 下)迁移训练，不覆盖原预训练权重。
- 多标签 Sigmoid：模型输出 num_classes 维 logits，BCEWithLogitsLoss 训练；
  非瑕疵 = 全零标签（无需 45 类 Softmax）。
- 训练过程可视化：每轮写 results.csv + TensorBoard(若可用) + 结束时生成 curves.png。

用法:
    python train.py --model repvit_m1_0 --device 0
    python train.py --model mobilenet_v3_large --device 1 --epochs 30
    bash train_both.sh                                        # 两模型同时训练

结果输出到 <model>/<name>/ 目录（默认 repvit_m1_0/ai-zzdn-<北京时间> 等），
包含 best.pth / last.pth / config.json / results.csv / curves.png。
"""

import argparse
import time
from fusion_report import write_epoch, write_best
import json
import os
os.environ.setdefault("TMPDIR", os.environ.get("FABRIC_TMP_DIR", os.path.join(os.path.dirname(__file__), ".runtime")))
import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import paths
from class_table import build_class_table
from dataset import MultiLabelDataset, SingleLabelDataset, build_transforms, load_stems
from metrics import multilabel_metrics, predict_scores, predict_softmax, singlelabel_metrics
import models

CST = timezone(timedelta(hours=8))  # 中国标准时间 = 北京时间/上海时间

CLOSE_MOSAIC = 10  # 最后 10 轮关闭 Mosaic（与检测 close_mosaic 一致）


def beijing_now_str(fmt="%Y%m%d_%H%M%S"):
    return datetime.now(CST).strftime(fmt)


def parse_device(s):
    """返回 (torch.device, gpu_ids)；gpu_ids 非 None 表示用 DataParallel。"""
    s = str(s).strip()
    if s.lower() == "cpu":
        return torch.device("cpu"), None
    ids = [int(x) for x in s.split(",") if x.strip() != ""]
    if len(ids) == 1:
        return torch.device(f"cuda:{ids[0]}"), None
    return torch.device(f"cuda:{ids[0]}"), ids


def load_class_info(data_dir):
    with open(os.path.join(data_dir, "classes.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return meta["nc"], meta["names"], meta.get("classes", [])


# ---- 温和均衡油污（移植自 ConvNeXt-V2/train.py）----
OIL_STAIN_CODE = "oil_stain"
SZBM_LABEL_DIR = os.environ.get("FABRIC_LABEL_JSON_DIR", "data/labels")


def load_oil_stain_stems(train_stems):
    """读取 szbm/label/<stem>.json，返回 primaryLabelCode == 'oil_stain' 的 stem 集合。
    缺失 json 的 stem 视为非油污。仅用于温和均衡：大油污相机降权封顶，其余自然频率。"""
    oil = set()
    for s in train_stems:
        p = os.path.join(SZBM_LABEL_DIR, f"{s}.json")
        try:
            with open(p, encoding="utf-8") as f:
                if json.load(f).get("primaryLabelCode") == OIL_STAIN_CODE:
                    oil.add(s)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return oil


def build_camera_balance_sampler(train_ds, oil_stems, cap=400):
    """温和均衡油污采样器：
    - 油污图按相机(ZH\\d+-[LR]|HL\\d+-[LR])分组，每组配额=min(组内油污数, cap)；
      组内每张权重 = 配额/组内油污数 → 大相机封顶，小相机自然。
    - 非油污图均分剩余权重（不按频率补偿），避免少数大相机主导。
    返回 (sampler, n_oil, n_cameras, n_non_oil)；无油污或相机≤1 时返回 (None, ...)。"""
    stems = train_ds.stems
    N = len(stems)
    cam_count = Counter()
    is_oil = []
    cams = []
    for s in stems:
        m = re.match(r"([A-Z]+\d+-[LR])", s)
        c = m.group(1) if m else "other"
        cams.append(c)
        oil = s in oil_stems
        is_oil.append(oil)
        if oil:
            cam_count[c] += 1
    n_oil = sum(is_oil)
    num_cameras = len([c for c in cam_count if cam_count[c] > 0])
    num_non_oil = N - n_oil
    if n_oil == 0 or num_cameras <= 1:
        return None, 0, 0, 0, 0
    quota = {c: min(n, cap) for c, n in cam_count.items()}
    E = sum(quota.values())
    w_non = max(N - E, 1) / max(num_non_oil, 1)
    weights = []
    for i in range(N):
        if is_oil[i]:
            w = quota[cams[i]] / cam_count[cams[i]]
        else:
            w = w_non
        weights.append(w)
    sampler = WeightedRandomSampler(weights, num_samples=N, replacement=True)
    return sampler, n_oil, num_cameras, num_non_oil


def build_datasets(data_dir, batch, workers, mosaic=False, preprocess="crop",
                   seg_dir=None, single_label=False, epoch_sample=None, image_dir=None,
                   balance_camera=False, balance_cap=400):
    """构建训练/验证 DataLoader。

    image_dir 非 None 时，图像直接从该目录取 `<stem>.jpg`（如指向 szbm/origin 的绝对图源，
    适用于 train.txt/val.txt 含绝对路径、coarse 目录无 images/ 的数据集）。
    seg_dir 非 None 时，图像改从 seg_dir 下取 `<stem>_segres.jpg`（布面 segres 裁剪图），
    并过滤掉 seg_dir 中无对应 segres 图的 stem；否则沿用 data_dir/images/<stem>.jpg。
    优先级: image_dir > seg_dir > data_dir/images。
    preprocess 决定几何预处理方法（crop=原方法，resize_pad=按比例缩放+黑边）。
    single_label=True 时改用 SingleLabelDataset（正常显式 + 互斥 softmax，4 类）。
    epoch_sample={类idx: 每轮上限}（仅单标签）时训练集每轮重抽样（如 1:3000），
    验证集不受影响；详见 SingleLabelDataset.resample。
    balance_camera=True（温和均衡油污）时改用 WeightedRandomSampler：油污图按相机封顶
    到 balance_cap/相机，非油污图均分剩余权重；仅多标签口径生效（single_label 时忽略）。
    """
    if image_dir is not None:
        image_dir = image_dir
        image_suffix = ".jpg"
        have = None
    elif seg_dir is not None:
        image_dir = seg_dir
        image_suffix = "_segres.jpg"
        import os as _os
        have = {f[:-len(image_suffix)] for f in _os.listdir(seg_dir) if f.endswith(image_suffix)}
    else:
        image_dir = os.path.join(data_dir, "images")
        image_suffix = ".jpg"
        have = None

    with open(os.path.join(data_dir, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    n_defect = len(load_class_info(data_dir)[1])

    def _stems(txt):
        s = load_stems(txt)
        if have is not None:
            miss = [x for x in s if x not in have]
            if miss:
                print(f"[数据] 过滤掉 {len(miss)} 个无 segres 图的 stem（示例: {miss[:2]}）")
            s = [x for x in s if x in have]
        return s

    train_stems = _stems(os.path.join(data_dir, "train.txt"))
    val_stems = _stems(os.path.join(data_dir, "val.txt"))

    if single_label:
        if epoch_sample is not None:
            bad = [k for k in epoch_sample if not (0 <= k < n_defect + 1)]
            assert not bad, f"--epoch-subsample 类别 idx 越界: {bad}（应为 0~{n_defect}）"
        normal_idx = n_defect  # 正常 = 最后一个类
        nc = n_defect + 1
        train_ds = SingleLabelDataset(image_dir, train_stems, labels, n_defect, normal_idx,
                                      build_transforms(train=True, method=preprocess),
                                      mosaic=mosaic, image_suffix=image_suffix,
                                      epoch_sample=epoch_sample)
        val_ds = SingleLabelDataset(image_dir, val_stems, labels, n_defect, normal_idx,
                                    build_transforms(train=False, method=preprocess),
                                    image_suffix=image_suffix)
    elif epoch_sample is not None:
        raise SystemExit("--epoch-subsample 仅支持 --single-label 口径")
    else:
        nc = n_defect
        train_ds = MultiLabelDataset(image_dir, train_stems, labels, nc,
                                     build_transforms(train=True, method=preprocess),
                                     mosaic=mosaic, image_suffix=image_suffix)
        val_ds = MultiLabelDataset(image_dir, val_stems, labels, nc,
                                   build_transforms(train=False, method=preprocess),
                                   image_suffix=image_suffix)

    # ---- 温和均衡油污采样器（仅多标签、且 --balance-camera 时生效）----
    sampler = None
    if balance_camera and not single_label:
        oil_stems = load_oil_stain_stems(train_stems)
        sampler, n_oil, n_cam, n_non = build_camera_balance_sampler(train_ds, oil_stems, cap=balance_cap)
        if sampler is not None:
            print(f"[均衡] 温和均衡油污 cap={balance_cap}: 油污 {n_oil} | 相机 {n_cam} | 非油污 {n_non}")
        else:
            print(f"[均衡] 未启用（油污 0 或相机≤1）")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=(sampler is None),
                              sampler=sampler, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=workers, pin_memory=True)
    return train_loader, val_loader, train_ds, val_ds, nc


def compute_pos_weight(train_ds, nc, max_w=50.0):
    """多标签正负不平衡加权：pos_weight[c] = 负样本数/正样本数，裁剪到 [1, max_w]。

    只读标签（train_ds.labels / train_ds.stems），不加载图像、不跑变换——
    旧实现对整训练集每张图跑完整 train 变换只为读标签，单线程耗时数分钟，此版本瞬间完成且结果一致。
    """
    y = np.zeros((len(train_ds), nc), dtype=np.float32)
    for i in range(len(train_ds)):
        for idx in train_ds.labels.get(train_ds.stems[i], []):
            y[i, idx] = 1.0
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = np.clip(neg / (pos + 1e-6), 1.0, max_w)
    return torch.tensor(w, dtype=torch.float32)


def compute_class_weights(train_ds, nc, max_w=5.0):
    """单标签互斥分类的类别权重：w_c = N / (C * n_c)，裁剪到 [1/max_w, max_w]。

    正常类(最后一类)占比 ~55% 会被显著降权，瑕疵类(尤其经向 ~9%)会被升权，
    以提升瑕疵召回。裁剪防止极端：normal 不会过弱(伤 precision)，经向不会过强(伤 precision)。
    只读 SingleLabelDataset._single（已预算的 int 标签），瞬间完成。
    """
    counts = np.zeros(nc, dtype=np.float64)
    for s in train_ds.stems:
        counts[train_ds._single[s]] += 1.0
    n = counts.sum()
    w = n / (nc * np.maximum(counts, 1.0))
    w = np.clip(w, 1.0 / max_w, max_w)
    print(f"[类权重] 各类样本数 {counts.astype(int).tolist()} -> 权重 {np.round(w,3).tolist()}")
    return torch.tensor(w, dtype=torch.float32)


def evaluate(model, loader, criterion, device, single_label=False, amp=False):
    if single_label:
        y_true, y_prob = predict_softmax(model, loader, device, amp=amp)
        with torch.no_grad():
            val_loss = 0.0
            n = 0
            for imgs, targets in loader:
                imgs = imgs.to(device)
                targets = targets.to(device).long()
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(imgs)
                    val_loss += criterion(logits.float(), targets).item() * imgs.size(0)
                n += imgs.size(0)
        val_loss /= max(n, 1)
        m = singlelabel_metrics(y_true, y_prob, threshold=None)  # argmax 预测
        m["val_loss"] = val_loss
        return m
    y_true, y_score = predict_scores(model, loader, device, amp=amp)
    with torch.no_grad():
        # val loss 复算一次（logits）
        val_loss = 0.0
        n = 0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(imgs)
                val_loss += criterion(logits.float(), targets).item() * imgs.size(0)
            n += imgs.size(0)
    val_loss /= max(n, 1)
    metrics = multilabel_metrics(y_true, y_score)
    metrics["val_loss"] = val_loss
    metrics["_fusion_preds"]=(y_true,y_score)
    return metrics


def save_checkpoint(path, model, epoch, best_metric, optimizer, scheduler, metrics):
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
    }, path + ".tmp")
    os.replace(path + ".tmp",path)


def _build_optimizer(model, lr, head_lr, weight_decay):
    """AdamW 优化器。head_lr 非 None 时给分类头单独的更高 LR（ViT 迁移学习标准做法）。

    分类头键名：timm ViT 是 'head.*'，torchvision mobilenet 是 'classifier.{0,3}.*'。
    DataParallel 下参数名前缀 'module.'，先剥掉再判断 'head.' 或 'classifier.' 开头。
    """
    if head_lr is None or abs(head_lr - lr) < 1e-12:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[优化器] AdamW  lr={lr}  weight_decay={weight_decay}  (单一 lr)")
        return opt

    def _is_head(name):
        n = name[len("module."):] if name.startswith("module.") else name
        return n.startswith(("head.", "classifier.", "fc."))

    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if _is_head(n) else backbone_params).append(p)
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=weight_decay)
    print(f"[优化器] AdamW  backbone lr={lr} ({len(backbone_params)} 张量) | "
          f"head lr={head_lr} ({len(head_params)} 张量) | weight_decay={weight_decay}")
    return opt


def _build_scheduler(optimizer, epochs, warmup_epochs):
    """LinearLR warmup(0.01*lr → lr) → CosineAnnealing。warmup_epochs=0 时仅 Cosine。

    用 SequentialLR 组合：前 warmup_epochs 轮线性升温，剩余 epochs-warmup_epochs 轮余弦退火。
    """
    if warmup_epochs <= 0:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        print(f"[调度] CosineAnnealing  epochs={epochs}  (无 warmup)")
        return sch
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs)
    sch = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    print(f"[调度] LinearLR warmup({warmup_epochs}ep, 0.1→1.0) → CosineAnnealing({epochs - warmup_epochs}ep)")
    return sch


def main():
    ap = argparse.ArgumentParser(description="布面瑕疵图像分类(多标签)训练")
    ap.add_argument('--fabric-mode', choices=['A', 'B', 'C'])
    ap.add_argument('--fabric-enhanced-dir', help='预生成的单通道 PNG 目录，B/C 共用')
    ap.add_argument("--model", required=True, choices=models.MODEL_NAMES, help="模型名")
    ap.add_argument("--data", default=paths.CLASS_DIR, help="分类数据目录(默认 data/ai-zzdn-class)")
    ap.add_argument("--pretrained", default=None, help="预训练权重路径(默认各模型 pretrained/ 下)")
    ap.add_argument("--device", default="0", help="GPU，如 0 或 0,1 或 cpu")
    ap.add_argument("--epochs", type=int, default=30, help="训练轮数")
    ap.add_argument("--patience", type=int, default=0,
                    help="早停耐心值: 连续 N 轮 val mAP 无提升则停止训练(0=关闭早停, 训满全部轮数)")
    ap.add_argument("--preprocess", choices=["crop", "resize_pad"], default="crop",
                    help="图像预处理方法: crop=原 CenterCrop/RandomResizedCrop; "
                         "resize_pad=按比例缩放+黑边贴左上(与 preview_segres_resize_pad.py 一致)")
    ap.add_argument("--seg-dir", default=None,
                    help="非空时改用该目录下 <stem>_segres.jpg 作为训练/验证图像(布面 segres 裁剪图)，"
                         "并过滤无对应 segres 的 stem；默认 None=用 <data>/images/<stem>.jpg")
    ap.add_argument("--image-dir", default=None,
                    help="非空时直接用该目录下 <stem>.jpg 作为图像源(如 szbm/origin)；"
                         "优先级高于 --seg-dir 与 <data>/images。适用于 coarse 目录无 images/ 的数据集")
    ap.add_argument("--imgsz", default=None,
                    help="模型输入尺寸 (H,W)，如 1024,1024；默认 None=用 dataset.INPUT_SIZE (512,608)。"
                         "设置后覆写 dataset.INPUT_SIZE，影响 ResizeRatioPad 画布与 Mosaic 画布")
    ap.add_argument("--mosaic", action="store_true", help="开启 Mosaic 增强（默认关闭）")
    ap.add_argument("--batch", type=int, default=32, help="batch size")
    ap.add_argument("--workers", type=int, default=8, help="dataloader 线程数")
    ap.add_argument("--lr", type=float, default=1e-3, help="学习率(AdamW, backbone)")
    ap.add_argument("--head-lr", type=float, default=None,
                    help="分类头单独学习率(默认 None=与 --lr 相同)。ViT 迁移学习通常设 5e-3 即 10x backbone")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="线性 warmup 轮数(默认 0=无 warmup)。1B+ 大模型迁移学习建议 3-5")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    ap.add_argument("--no-pos-weight", action="store_true", help="关闭多标签正负不平衡加权")
    ap.add_argument("--single-label", action="store_true",
                    help="单标签互斥口径：正常显式为最后一类，优先级 经向>纬向>面积>正常；"
                         "CrossEntropyLoss + 类权重；按 macro_f1 选 best")
    ap.add_argument("--epoch-subsample", default=None,
                    help="每轮 epoch 从训练池按类重抽样，逗号分隔 '类idx:数量'，如 '1:3000' "
                         "（仅 --single-label；验证集固定不受影响；某类不足数量时取全部）")
    ap.add_argument("--class-weights", default=None,
                    help="单标签各类 CE 权重手动覆写，逗号分隔，如 '3.0,1.3,1.0,0.45' "
                         "(经向,纬向,面积,正常)；默认 None=逆频自动权重")
    ap.add_argument("--amp", action="store_true",
                    help="混合精度训练(AMP, fp16 autocast + GradScaler)，加速 ~2x、显存减半")
    ap.add_argument("--balance-camera", action="store_true",
                    help="温和均衡油污：油污图按相机封顶到 --balance-cap/相机，其余自然频率；"
                         "移植自 ConvNeXt-V2；仅多标签口径生效（默认关闭）")
    ap.add_argument("--balance-cap", type=int, default=400,
                    help="温和均衡油污每相机油污配额上限（默认 400；与 ConvNeXt-V2 一致）")
    ap.add_argument("--project", default=None, help="输出项目目录(默认=模型名)")
    ap.add_argument("--name", default=None, help="输出实验名(默认 ai-zzdn-<北京时间>)")
    ap.add_argument("--exist-ok", action="store_true", help="允许覆盖已存在的输出目录")
    ap.add_argument("--resume", action="store_true", help="从 last.pth 继续训练")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()
    if args.device != "0" or torch.cuda.device_count()!=1:
        raise RuntimeError("exactly one visible CUDA device required")
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("FUSION_GPU_MEMORY_FRACTION",".12")), 0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- 覆写模型输入尺寸 ----
    imgsz = None
    if args.imgsz:
        parts = [int(x.strip()) for x in args.imgsz.split(",")]
        if len(parts) == 1:
            imgsz = (parts[0], parts[0])
        elif len(parts) == 2:
            imgsz = (parts[0], parts[1])  # (H, W)
        else:
            raise ValueError(f"--imgsz 格式错误: {args.imgsz!r}，应为 '1024,1024' 或 '1024'")
        import dataset as _ds
        _ds.INPUT_SIZE = imgsz
        print(f"[imgsz] 覆写 dataset.INPUT_SIZE = {imgsz} (H,W)")

    device, gpu_ids = parse_device(args.device)

    # ---- 数据 ----
    epoch_sample = None
    if args.epoch_subsample:
        epoch_sample = {}
        for part in args.epoch_subsample.split(","):
            cls_s, n_s = part.strip().split(":")
            epoch_sample[int(cls_s)] = int(n_s)
        print(f"[重抽样] 每轮按类抽样: {epoch_sample}（验证集固定不受影响）")
    if args.fabric_mode:
        from pathlib import Path
        from fabric_inputs import PrecomputedFabric, inside, read_inputs
        if (args.model != "mobilenet_v3_large" or args.single_label or args.mosaic
                or args.balance_camera or epoch_sample is not None or args.seg_dir
                or args.preprocess != "crop" or args.resume != args.exist_ok):
            ap.error("织布对照只支持 MobileNetV3-Large 多标签、共享 crop、从预训练启动及新输出目录")
        if not all((args.pretrained, args.project, args.name)):
            ap.error("请明确指定 zx 内的 --pretrained、--project 和 --name")
        if args.fabric_mode in ("raw", "raw_enhanced") and not args.image_dir:
            ap.error("A、C 需要 --image-dir 指向原图 JPG 目录")
        if args.fabric_mode in ("enhanced", "raw_enhanced") and not args.fabric_enhanced_dir:
            ap.error("B、C 需要 --fabric-enhanced-dir 指向预生成的灰度 PNG 目录")
        if Path(args.name).name != args.name or args.name in (".", ".."):
            ap.error("非法实验名")
        inside(args.pretrained)
        inside(Path(args.project) / args.name, strict=False)
        labels, _, splits = read_inputs(args.data)
        datasets = [PrecomputedFabric(args.image_dir, splits[tag], labels, args.fabric_mode,
                                      train=(tag == "train"), size=imgsz or (512, 608),
                                      enhanced_dir=args.fabric_enhanced_dir)
                    for tag in ("train", "val")]
        train_ds, val_ds = datasets
        train_loader, val_loader = [
            DataLoader(ds, batch_size=args.batch, shuffle=(i == 0),
                       num_workers=args.workers, pin_memory=device.type == "cuda")
            for i, ds in enumerate(datasets)]
        nc = 3
    else:
        train_loader, val_loader, train_ds, val_ds, nc = build_datasets(
            args.data, args.batch, args.workers, mosaic=args.mosaic,
            preprocess=args.preprocess, seg_dir=args.seg_dir, single_label=args.single_label,
            epoch_sample=epoch_sample, image_dir=args.image_dir,
            balance_camera=args.balance_camera, balance_cap=args.balance_cap)
    names = load_class_info(args.data)[1]
    if args.single_label:
        names = list(names) + ["正常"]
    print(f"[数据] 训练 {len(train_ds)} | 验证 {len(val_ds)} | 类别 {nc}"
          f"{' (单标签: ' + '/'.join(names) + ')' if args.single_label else ' | 类别 ' + str(nc)}"
          f" | 预处理={args.preprocess}"
          f"{' | 图源=image_dir(' + args.image_dir + ')' if args.image_dir else (' | 图源=segres(' + args.seg_dir + ')' if args.seg_dir else ' | 图源=images')}")

    # ---- 模型 ----
    pretrained = args.pretrained or models.default_pretrained_path(args.model)
    model = models.build_model(args.model, nc, pretrained_path=pretrained, img_size=imgsz)
    if args.fabric_mode:
        from fabric_inputs import adapt_stem
        model = adapt_stem(model, args.fabric_mode)
    model = model.to(device)
    if gpu_ids and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"[设备] DataParallel GPU {gpu_ids}")
    else:
        print(f"[设备] {device}")

    # ---- 损失 & 优化器 ----
    if args.single_label:
        if args.class_weights:
            w_list = [float(x) for x in args.class_weights.split(",")]
            assert len(w_list) == nc, f"--class-weights 需 {nc} 个值，得到 {len(w_list)}"
            class_w = torch.tensor(w_list, dtype=torch.float32)
            print(f"[类权重] 手动覆写 {np.round(w_list,3).tolist()} (经向,纬向,面积,正常)")
        else:
            class_w = compute_class_weights(train_ds, nc).to(device)
            class_w = class_w.to("cpu")  # 下面统一 .to(device)
        print(f"[损失] CrossEntropyLoss + 类权重(裁剪[0.2,5]) + label_smoothing 0.1")
        criterion = nn.CrossEntropyLoss(weight=class_w.to(device), label_smoothing=0.1)
    else:
        pos_weight = None if args.no_pos_weight else compute_pos_weight(train_ds, nc).to(device)
        if pos_weight is not None:
            print(f"[损失] BCEWithLogitsLoss + pos_weight(各类正负比, 裁剪到[1,50])")
        else:
            print(f"[损失] BCEWithLogitsLoss(无加权)")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = _build_optimizer(model, args.lr, args.head_lr, args.weight_decay)
    scheduler = _build_scheduler(optimizer, args.epochs, args.warmup_epochs)

    # ---- 输出目录 ----
    name = args.name or f"ai-zzdn-{beijing_now_str()}"
    out_dir = os.path.join(args.project or args.model, name)
    os.makedirs(out_dir, exist_ok=args.exist_ok)

    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        ckpt_path = os.path.join(out_dir, "last.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"resume 找不到 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -1.0)
        if os.path.isfile(os.path.join(out_dir,"best.pth")):
            best_metric=max(best_metric,torch.load(os.path.join(out_dir,"best.pth"),map_location="cpu",weights_only=False)["best_metric"])
        print(f"[resume] 从 epoch {start_epoch} 继续")

    # ---- 保存 config.json（供 test/infer 复用）----
    _, _, classes = load_class_info(args.data)
    if args.single_label:
        classes = list(classes) + [{"code": "normal", "name": "正常", "group": "正常",
                                     "category": "正常", "idx": nc - 1}]
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "num_classes": nc,
                   "names": names, "classes": classes,
                   "preprocess": args.preprocess,
                   "seg_dir": args.seg_dir,
                   "image_dir": args.image_dir,
                   "imgsz": list(imgsz) if imgsz else None,
                   "single_label": args.single_label,
                   "fabric_mode": args.fabric_mode,
                   "fabric_enhanced_dir": args.fabric_enhanced_dir}, f, ensure_ascii=False, indent=2)

    # ---- 可视化记录 ----
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=out_dir)
    except Exception as e:
        print(f"[提示] TensorBoard 不可用({e})，仅写 CSV/曲线图。")

    csv_path = os.path.join(out_dir, "results.csv")
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "mAP": [],
            "exact_acc": [], "micro_f1": [], "macro_f1": []}

    def write_csv_header():
        if args.single_label:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss,val_loss,macro_f1,macro_p,macro_r,accuracy,"
                        "def_recall,def_p,def_img_recall,def_img_precision\n")
        else:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss,val_loss,mAP,exact_acc,hamming_acc,micro_f1,macro_f1,micro_p,micro_r\n")

    if not args.resume:
        write_csv_header()

    print(f"[训练] 模型={args.model}  epochs={args.epochs}  batch={args.batch}  "
          f"lr={args.lr}  输出={out_dir}")
    if args.patience and args.patience > 0:
        print(f"[早停] 开启: 连续 {args.patience} 轮 val mAP 无提升即停止")

    # ---- 早停状态 ----
    patience = args.patience if args.patience and args.patience > 0 else 0
    no_improve = 0  # 连续未提升轮数

    # ---- AMP GradScaler ----
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    # ---- 训练循环 ----
    for epoch in range(start_epoch, args.epochs):
        t_epoch=time.monotonic()
        train_ds.mosaic = args.mosaic and (epoch < args.epochs - CLOSE_MOSAIC)  # 最后 CLOSE_MOSAIC 轮关闭 mosaic
        if epoch_sample:
            train_ds.resample()  # 每轮换一批抽样图（worker 每轮重新 fork，随即生效）
            comp = Counter(train_ds._single[s] for s in train_ds.stems)
            print(f"[重抽样] epoch {epoch + 1}: " + "/".join(
                f"{names[i]}={comp.get(i, 0)}" for i in range(nc)))
        model.train()
        train_loss = 0.0
        for step,(imgs, targets) in enumerate(train_loader,1):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(imgs)
                loss = criterion(logits.float(), targets)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lv=loss.item()
            if not np.isfinite(lv): raise RuntimeError(f"nonfinite loss: epoch={epoch+1}, step={step}")
            train_loss += lv * imgs.size(0)
            if step==1 or step%100==0: print(f"[step] epoch={epoch+1} step={step}/{len(train_loader)} loss={lv:.5f} elapsed={time.monotonic()-t_epoch:.1f}s",flush=True)
        train_loss /= max(len(train_ds), 1)

        m = evaluate(model, val_loader, criterion, device,
                     single_label=args.single_label, amp=args.amp)
        y,scores=m.pop("_fusion_preds")
        scheduler.step()

        # 当前轮实际学习率（优化器所有参数组的 LR，便于诊断 warmup/cosine 是否生效）
        cur_lrs = [g["lr"] for g in optimizer.param_groups]
        lr_str = "/".join(f"{x:.2e}" for x in cur_lrs)

        # 记录（单标签/多标签分别记不同指标）
        hist["epoch"].append(epoch + 1)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(m["val_loss"])
        if args.single_label:
            hist["macro_f1"].append(m["macro_f1"])
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch + 1},{train_loss:.6f},{m['val_loss']:.6f},{m['macro_f1']:.4f},"
                        f"{m['macro_precision']:.4f},{m['macro_recall']:.4f},{m['accuracy']:.4f},"
                        f"{m['defect_recall']:.4f},{m['def_img_precision']:.4f},"
                        f"{m['def_img_recall']:.4f},{m['def_img_precision']:.4f}\n")
        else:
            hist["mAP"].append(m["mAP"])
            hist["exact_acc"].append(m["exact_acc"])
            hist["micro_f1"].append(m["micro_f1"])
            hist["macro_f1"].append(m["macro_f1"])
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch + 1},{train_loss:.6f},{m['val_loss']:.6f},{m['mAP']:.4f},"
                        f"{m['exact_acc']:.4f},{m['hamming_acc']:.4f},{m['micro_f1']:.4f},"
                        f"{m['macro_f1']:.4f},{m['micro_p']:.4f},{m['micro_r']:.4f}\n")

        if args.single_label:
            print(f"[epoch {epoch + 1:>3}/{args.epochs}] train_loss={train_loss:.4f}  "
                  f"val_loss={m['val_loss']:.4f}  macro_f1={m['macro_f1']:.4f}  "
                  f"def_recall={m['defect_recall']:.4f}  acc={m['accuracy']:.4f}  lr={lr_str}")
        else:
            print(f"[epoch {epoch + 1:>3}/{args.epochs}] train_loss={train_loss:.4f}  "
                  f"val_loss={m['val_loss']:.4f}  mAP={m['mAP']:.4f}  "
                  f"exact_acc={m['exact_acc']:.4f}  macro_f1={m['macro_f1']:.4f}  "
                  f"lr={lr_str}")

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", m["val_loss"], epoch)
            if args.single_label:
                writer.add_scalar("metrics/macro_f1", m["macro_f1"], epoch)
                writer.add_scalar("metrics/def_recall", m["defect_recall"], epoch)
            else:
                writer.add_scalar("metrics/mAP", m["mAP"], epoch)
                writer.add_scalar("metrics/exact_acc", m["exact_acc"], epoch)
                writer.add_scalar("metrics/macro_f1", m["macro_f1"], epoch)
                writer.add_scalar("metrics/micro_f1", m["micro_f1"], epoch)

        # 保存 last + 最优 best（单标签按 macro_f1，多标签按 val mAP）
        best_key = "macro_f1" if args.single_label else "mAP"
        if m[best_key] > best_metric:
            best_metric = m[best_key]
            no_improve = 0  # 有提升，重置早停计数
            save_checkpoint(os.path.join(out_dir, "best.pth"), model, epoch,
                            best_metric, optimizer, scheduler, m)
            if args.fabric_mode:
                write_best(out_dir,val_ds.stems,names,y,scores,epoch+1,args.fabric_mode,m)
        else:
            no_improve += 1  # 未提升
            if patience > 0:
                print(f"[早停] 已连续 {no_improve}/{patience} 轮无提升")

        save_checkpoint(os.path.join(out_dir,"last.pth"),model,epoch,best_metric,optimizer,scheduler,m)
        write_epoch(out_dir,names,y,scores,epoch+1,args.fabric_mode,m)
        print(f"[epoch timing] {epoch+1}/{args.epochs} {time.monotonic()-t_epoch:.1f}s",flush=True)

        # 早停判断: 达到耐心值即停止
        if patience > 0 and no_improve >= patience:
            print(f"[早停] 连续 {patience} 轮 {best_key} 无提升，提前停止训练 "
                  f"(当前 epoch {epoch + 1}, best {best_key}={best_metric:.4f})")
            break

    # ---- 曲线图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
        axes[0].plot(hist["epoch"], hist["val_loss"], label="val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
        if args.single_label:
            axes[1].plot(hist["epoch"], hist["macro_f1"], label="macro_f1")
            axes[1].set_title("Val macro_f1"); axes[1].set_xlabel("epoch")
            axes[2].plot(hist["epoch"], [h for h in hist["macro_f1"]], label="macro_f1")
            axes[2].set_title("Val macro_f1"); axes[2].set_xlabel("epoch")
        else:
            axes[1].plot(hist["epoch"], hist["mAP"], label="mAP")
            axes[1].set_title("Val mAP"); axes[1].set_xlabel("epoch")
            axes[2].plot(hist["epoch"], hist["micro_f1"], label="micro_f1")
            axes[2].plot(hist["epoch"], hist["macro_f1"], label="macro_f1")
            axes[2].set_title("Val F1"); axes[2].set_xlabel("epoch"); axes[2].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "curves.png"), dpi=120)
        plt.close(fig)
        print(f"[可视化] 曲线已保存: {os.path.join(out_dir, 'curves.png')}")
    except Exception as e:
        print(f"[提示] 生成曲线图失败: {e}")

    if writer is not None:
        writer.close()

    best_key = "macro_f1" if args.single_label else "mAP"
    print(f"[完成] 最优 {best_key}={best_metric:.4f}  输出目录: {out_dir}")


if __name__ == "__main__":
    main()
