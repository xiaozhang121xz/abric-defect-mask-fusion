#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分类类别表（与目标检测 detect_yolo/convert_szbm_to_yolo.py 口径完全一致）。

类别口径（已与需求方确定）：
    44 类（Excel）+ hole(破洞) = 瑕疵标签；fold_back(折返) 映射到 Excel 的 repair_return(折返)。
    pending_review(待复核缺陷) 是复核状态而非具体瑕疵，对应框丢弃。

以下 6 类「不参与训练」，不占类别位、对应框/图在转换时排除：
    previous_process_break(前道因倒断头) / hard_object_pressure(刺皮硬物压伤) /
    abrasion(擦伤) / white_abrasion(擦白) / other(其它)（5 类，需求方 2026-09-12 确认）+
    photo_incorrect(拍摄异常)（2026-09-13 确认）。
    说明：szbm 原始归档仍照常下载保存这些数据（pull_data 不变），仅在
    转换为训练数据时整体排除，使其不进入 ai-zzdn / ai-zzdn-class(-binary) / ai-zzdn-single。

非瑕疵（正常布面）不占标签位，在分类里表示为「全零向量」。
"""

import collections
import json
import os
import zipfile
from xml.etree import ElementTree as ET

import paths

EXCEL_PATH = paths.EXCEL_PATH

# 类别口径（与 detect_yolo 一致）：44 类 Excel + hole 兜底类（other 已下线，见 DROP_CODES）
EXTRA_CLASSES = [  # (code, 中文名)，追加在 44 类之后
    ("hole", "破洞"),
]
CODE_ALIAS = {  # 数据里的编码 -> 规范编码（同义但编码不一致）
    "fold_back": "repair_return",  # 折返
    "拆返": "repair_return",        # 折返（中文异写/别名）
    "take_apart": "repair_return",  # 拆返（英文）-> 折返
}
DROP_CODES = {  # 直接丢弃的编码（非具体瑕疵 / 不参与训练，不应成为类别）
    "pending_review",  # 待复核缺陷（复核状态）
    "non_fabric",      # 非布面（非瑕疵，不应作为类别）
    # ---- 以下 6 类不再参与训练（5 类 2026-09-12 + photo_incorrect 2026-09-13 确认）----
    # szbm 归档仍照常下载保存，仅在转换训练数据时排除（框丢弃 + 整图排除）。
    "previous_process_break",  # 前道因倒断头
    "hard_object_pressure",    # 刺皮硬物压伤
    "abrasion",                # 擦伤
    "white_abrasion",          # 擦白
    "other",                   # 其它
    "photo_incorrect",         # 拍摄异常（2026-09-13 确认；无框仅 primaryLabelCode，整图排除，不占类）
}
NAME_OVERRIDE = {  # 动态发现的新编码 -> 规范中文名（数据内无 labelName 时才用）
    "photo_incorrect": "拍摄异常",  # 拍摄异常（无框，仅 primaryLabelCode）
}


# ---------------------------------------------------------------------------
# Excel 读取（仅用标准库，避免依赖 openpyxl）
# ---------------------------------------------------------------------------
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _read_shared_strings(xlsx_path):
    with zipfile.ZipFile(xlsx_path) as z:
        if "xl/sharedStrings.xml" not in z.namelist():
            return []
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        return ["".join(t.text or "" for t in si.iter(_NS + "t")) for si in root.findall(_NS + "si")]


def read_excel_classes(xlsx_path):
    """读取 Excel 里的 44 类，返回按『序号』排序的列表 [{no, group, name, code, category}]。"""
    shared = _read_shared_strings(xlsx_path)
    with zipfile.ZipFile(xlsx_path) as z:
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    classes = []
    for row in sheet.iter(_NS + "row"):
        cells = {}
        for c in row.findall(_NS + "c"):
            ref = c.attrib.get("r", "")
            col = "".join(ch for ch in ref if ch.isalpha())
            t = c.attrib.get("t")
            v = c.find(_NS + "v")
            val = None
            if v is not None:
                val = v.text
                if t == "s":
                    val = shared[int(val)]
            cells[col] = val

        no = cells.get("A")
        code = cells.get("D")
        if no is not None and str(no).strip().isdigit() and code:
            classes.append({
                "no": int(str(no).strip()),
                "group": cells.get("B"),
                "name": (cells.get("C") or "").strip(),
                "code": code.strip(),
                "category": cells.get("E"),
            })
    classes.sort(key=lambda x: x["no"])
    return classes


def discover_extra_classes(labels_dir=None):
    """动态发现 szbm 标注里「固定类表之外」的瑕疵编码，返回 [(code, 中文名), ...]。

    保证 szbm 里只要出现的类型都被收录，而不是写死 46 类；新增类型自动纳入。
    已通过 CODE_ALIAS 归一（拆返/take_apart -> repair_return）的编码不重复收录；
    pending_review 等 DROP_CODES 不收录。名称取数据里出现最多的 labelName（无则退回编码本身）。
    """
    if labels_dir is None:
        labels_dir = os.path.join(paths.SZBM_DIR, "label")
    if not os.path.isdir(labels_dir):
        return []

    fixed = {row["code"] for row in read_excel_classes(EXCEL_PATH)}
    for code, _name in EXTRA_CLASSES:
        fixed.add(code)

    name_counter = collections.defaultdict(collections.Counter)
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(labels_dir, fn), encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue

        for b in d.get("boxes") or []:
            lc = b.get("labelCode")
            if not lc:
                continue
            lc = CODE_ALIAS.get(lc, lc)
            if lc in DROP_CODES or lc in fixed:
                continue
            nm = (b.get("labelName") or "").strip() or lc
            name_counter[lc][nm] += 1

        # 顶层 primaryLabelCode 也可能承载有效瑕疵编码（尤其 boxes 为空时的退化单标签）
        pc = d.get("primaryLabelCode")
        if pc:
            pc = CODE_ALIAS.get(pc, pc)
            if pc not in DROP_CODES and pc not in fixed and pc not in name_counter:
                name_counter[pc][pc] += 1

    extras = []
    for code in sorted(name_counter):
        if code in fixed or code in DROP_CODES:
            continue
        name = sorted(name_counter[code].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        name = NAME_OVERRIDE.get(code, name)
        extras.append((code, name))
    return extras


def build_class_table():
    """返回 (classes, code2idx, names)。classes: [{code,name,group,category,idx}]。

    类别 = Excel(去掉 DROP_CODES 中不参与训练的类) + hole 兜底类 + 动态发现的 szbm 额外类型
    （不写死，新增类型自动纳入）。DROP_CODES（前道因倒断头/刺皮硬物压伤/擦伤/擦白/其它等）
    不占类别位。
    """
    excel = read_excel_classes(EXCEL_PATH)
    if len(excel) != 44:
        raise RuntimeError(f"Excel 应包含 44 个瑕疵类，实际读到 {len(excel)} 个，请检查。")

    classes, code2idx, names = [], {}, []
    for row in excel:
        if row["code"] in DROP_CODES:  # 不参与训练的 Excel 类（如前道因倒断头/擦伤等）
            continue
        idx = len(classes)
        classes.append({"code": row["code"], "name": row["name"],
                        "group": row["group"], "category": row["category"], "idx": idx})
        code2idx[row["code"]] = idx
        names.append(row["name"])
    for code, name in EXTRA_CLASSES:
        idx = len(classes)
        classes.append({"code": code, "name": name, "group": "新增", "category": "新增", "idx": idx})
        code2idx[code] = idx
        names.append(name)
    for code, name in discover_extra_classes():
        idx = len(classes)
        classes.append({"code": code, "name": name, "group": "新增", "category": "新增", "idx": idx})
        code2idx[code] = idx
        names.append(name)
    return classes, code2idx, names


def cross_check_with_detect(names):
    """与目标检测已生成的 classes.json 交叉校验口径一致性（存在才校验，不报错只提示）。"""
    if not os.path.exists(paths.DETECT_CLASSES_JSON):
        return
    try:
        with open(paths.DETECT_CLASSES_JSON, encoding="utf-8") as f:
            det = json.load(f)
        det_names = det.get("names", [])
        if det_names and det_names != names:
            print(f"[口径警告] 分类类别表与目标检测 classes.json 不一致，请检查。")
    except Exception:
        pass
