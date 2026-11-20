#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多标签分类评估指标（不依赖 sklearn，纯 numpy 实现）。

输入约定:
    y_true  : [N, C] 0/1 ground-truth multi-hot
    y_score : [N, C] 概率（sigmoid 后，取值 [0,1]）
"""

import numpy as np
import torch


def predict_scores(model, loader, device, amp=False):
    """在给定 DataLoader 上做推理，返回 (y_true, y_score) 两个 [N,C] numpy 数组（概率）。"""
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(imgs)
            ss.append(torch.sigmoid(logits.float()).cpu().numpy())
            ys.append(targets.cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(ss, axis=0)


def predict_softmax(model, loader, device, amp=False):
    """单标签互斥分类推理：返回 (y_true[N] int, y_prob[N,C] softmax 概率)。"""
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(imgs)
            ss.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            ys.append(targets.cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(ss, axis=0)


def singlelabel_metrics(y_true, y_prob, threshold=None):
    """单标签互斥分类指标。

    y_true : [N] int  真实类（正常约定为最后一个类 normal_idx = C-1）
    y_prob : [N, C] softmax 概率
    threshold : None=argmax 预测；或长度 C 的每类门限——
        预测规则：在 prob>=thr 的类中取概率最大者；全不达标则判为 normal_idx。
    返回 dict：每类 {tp,fp,fn,precision,recall,f1,support} + macro P/R/F1 + accuracy
        + defect_recall（真实为瑕疵类的图被正确判该瑕疵类的比例）+ defect_n。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float)
    n, c = y_prob.shape
    normal_idx = c - 1
    if threshold is None:
        y_pred = y_prob.argmax(axis=1)
    else:
        thr = np.asarray(threshold, dtype=float).reshape(-1)
        mask = y_prob >= thr[None, :]  # [N, C]
        y_pred = np.full(n, normal_idx, dtype=int)
        for i in range(n):
            cand = np.where(mask[i])[0]
            if cand.size == 0:
                continue
            y_pred[i] = cand[int(np.argmax(y_prob[i, cand]))]

    out = {"per_class": {}}
    ps, rs, fs = [], [], []
    for k in range(c):
        tp = int(np.sum((y_pred == k) & (y_true == k)))
        fp = int(np.sum((y_pred == k) & (y_true != k)))
        fn = int(np.sum((y_pred != k) & (y_true == k)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        sup = int(np.sum(y_true == k))
        out["per_class"][k] = {"tp": tp, "fp": fp, "fn": fn,
                               "precision": p, "recall": r, "f1": f, "support": sup}
        if sup > 0:
            ps.append(p); rs.append(r); fs.append(f)
    out["macro_precision"] = float(np.mean(ps)) if ps else 0.0
    out["macro_recall"] = float(np.mean(rs)) if rs else 0.0
    out["macro_f1"] = float(np.mean(fs)) if fs else 0.0
    out["accuracy"] = float(np.mean(y_pred == y_true))
    dmask = y_true != normal_idx
    out["defect_n"] = int(dmask.sum())
    out["defect_recall"] = float(np.mean(y_pred[dmask] == y_true[dmask])) if dmask.sum() > 0 else 0.0
    # 图像级「有瑕疵 vs 无瑕疵」二分类口径
    true_has = (y_true != normal_idx).astype(float)
    pred_has = (y_pred != normal_idx).astype(float)
    tp_img = float((pred_has * true_has).sum())
    fp_img = float((pred_has * (1.0 - true_has)).sum())
    fn_img = float(((1.0 - pred_has) * true_has).sum())
    out["def_img_recall"] = tp_img / (tp_img + fn_img + 1e-12) if (tp_img + fn_img) > 0 else 0.0
    out["def_img_precision"] = tp_img / (tp_img + fp_img + 1e-12) if (tp_img + fp_img) > 0 else 0.0
    return out



def average_precision(y_true, y_score):
    """单类 Average Precision（VOC/sklearn 口径）。无正样本返回 nan。"""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = float(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1.0 - y_true)
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / n_pos
    rec = np.concatenate([[0.0], rec])
    prec = np.concatenate([[1.0], prec])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def multilabel_metrics(y_true, y_score, threshold=0.5):
    """计算多标签评估指标，返回 dict。"""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n, c = y_true.shape
    y_pred = (y_score >= threshold).astype(float)

    # 各类 AP 与 mAP（仅对存在正样本的类求均值，避免无正样本类拉低指标）
    aps = []
    for j in range(c):
        aps.append(average_precision(y_true[:, j], y_score[:, j]))
    valid_aps = [a for a in aps if not np.isnan(a)]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    # 完全匹配（子集）准确率 & 汉明（逐标签）准确率
    exact_acc = float(np.mean(np.all(y_pred == y_true, axis=1)))
    hamming_acc = float(np.mean(y_pred == y_true))

    # micro P/R/F1
    tp = float(np.sum(y_pred * y_true))
    fp = float(np.sum(y_pred * (1.0 - y_true)))
    fn = float(np.sum((1.0 - y_pred) * y_true))
    micro_p = tp / (tp + fp + 1e-12)
    micro_r = tp / (tp + fn + 1e-12)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-12)

    # macro P/R/F1（对存在正样本的类）
    macro_p, macro_r, macro_f1 = [], [], []
    for j in range(c):
        if y_true[:, j].sum() == 0:
            continue
        tpj = float(np.sum(y_pred[:, j] * y_true[:, j]))
        fpj = float(np.sum(y_pred[:, j] * (1.0 - y_true[:, j])))
        fnj = float(np.sum((1.0 - y_pred[:, j]) * y_true[:, j]))
        pj = tpj / (tpj + fpj + 1e-12)
        rj = tpj / (tpj + fnj + 1e-12)
        macro_p.append(pj)
        macro_r.append(rj)
        macro_f1.append(2 * pj * rj / (pj + rj + 1e-12))
    macro_p = float(np.mean(macro_p)) if macro_p else 0.0
    macro_r = float(np.mean(macro_r)) if macro_r else 0.0
    macro_f1 = float(np.mean(macro_f1)) if macro_f1 else 0.0

    return {
        "mAP": mAP,
        "exact_acc": exact_acc,
        "hamming_acc": hamming_acc,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "per_class_ap": aps,
        "n_positive_classes": len(valid_aps),
    }


def defect_level_metrics(y_true, y_score, threshold=0.5):
    """瑕疵「总召回率 / 总 Precision」图像级二分类口径（不把非瑕疵当作独立类别）。

    图像级二分类「有瑕疵 vs 无瑕疵」：
        每张图 y>0（任一类≥1）视为有瑕疵，全零视为无瑕疵；
        召回=有瑕疵图被判有瑕疵的比例，Precision=被判有瑕疵的图中真正有瑕疵的比例
        （无瑕疵图作为负类，其误报计入 FP，即工业检测的检出率/误检率）。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(float)

    true_has = (y_true.sum(axis=1) > 0).astype(float)   # 1=有瑕疵
    pred_has = (y_pred.sum(axis=1) > 0).astype(float)   # 1=预测有瑕疵

    tp_img = float((pred_has * true_has).sum())
    fp_img = float((pred_has * (1.0 - true_has)).sum())
    fn_img = float(((1.0 - pred_has) * true_has).sum())
    def_img_recall = tp_img / (tp_img + fn_img + 1e-12)
    def_img_precision = tp_img / (tp_img + fp_img + 1e-12)
    def_img_f1 = 2 * def_img_recall * def_img_precision / (def_img_recall + def_img_precision + 1e-12)
    # 非瑕疵错检率：无瑕疵图被误判为有瑕疵的比例 = FP / 无瑕疵图数（工业口径误报率）
    nondefect_fpr = fp_img / (((1.0 - true_has).sum()) + 1e-12)

    return {
        "def_img_recall": def_img_recall,
        "def_img_precision": def_img_precision,
        "def_img_f1": def_img_f1,
        "def_img_n_pos": int(true_has.sum()),          # 有瑕疵图数
        "def_img_n_neg": int((1.0 - true_has).sum()),  # 无瑕疵图数
        "def_img_tp": int(tp_img), "def_img_fp": int(fp_img), "def_img_fn": int(fn_img),
        "nondefect_fpr": float(nondefect_fpr),         # 非瑕疵类型数据错检率
    }


def type_level_pr(true_has, pred_has, names):
    """按类型(列)计算图像级 micro 召回/准确率（容忍度三档 + 正常类型 用）。

    入参：
        true_has : [N, K] 0/1，图 i 真值含类型 k
        pred_has : [N, K] 0/1，图 i 预测为类型 k
        names    : list[K]，每列对应的 key 后缀（如 "tier0"/"tier1"/"tier2"/"normal"）
    返回 flattened dict：{<name>_recall, <name>_precision, <name>_tp/fp/fn}。
        无该类正样本时 recall/precision=None。
    口径（图像级 micro，按图池化）：
        recall_T  = TP_T / (TP_T + FN_T)   # 该类型图被判对该类型的比例
        precision_T = TP_T / (TP_T + FP_T)  # 预测为该类型的图中真为该类型的比例
    说明：一张图含多档瑕疵时在多档真值列同时为 1（多标签，非互斥）；正常类型与各档互斥。
    """
    th = np.asarray(true_has, dtype=float)
    ph = np.asarray(pred_has, dtype=float)
    if th.ndim == 1:
        th = th.reshape(-1, 1)
        ph = ph.reshape(-1, 1)
    out = {}
    for k in range(th.shape[1]):
        yt = th[:, k]
        yp = ph[:, k]
        tp = float(np.sum(yp * yt))
        fp = float(np.sum(yp * (1.0 - yt)))
        fn = float(np.sum((1.0 - yp) * yt))
        recall = tp / (tp + fn + 1e-12) if (tp + fn) > 0 else None
        precision = tp / (tp + fp + 1e-12) if (tp + fp) > 0 else None
        nm = names[k]
        out[f"{nm}_tp"] = int(round(tp))
        out[f"{nm}_fp"] = int(round(fp))
        out[f"{nm}_fn"] = int(round(fn))
        out[f"{nm}_recall"] = float(recall) if recall is not None else None
        out[f"{nm}_precision"] = float(precision) if precision is not None else None
    return out
