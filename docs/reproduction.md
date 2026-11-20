# 复现与公开前整理

## 当前状态

`src/` 是 2026-11 实验服务器上的代码快照。原始数据、标签、预训练权重、分割模型权重和部分缓存不在本仓库，所以不能在干净机器上直接复现数值。某些入口仍需把本机数据位置配置为环境变量；先检查 `src/fabric_inputs.py`、`src/legacy_fabric_inputs.py`、`src/train.py` 中路径配置，再运行。

服务器当时使用 Python 3.10、PyTorch、Torchvision、NumPy、OpenCV contrib、Pillow。运行前需依据目标设备安装匹配 CUDA 的 PyTorch/Torchvision 版本；没有锁文件，所以版本并未被严格复现。完整依赖列表见 `requirements.txt`，它只提供参考范围。

## 输入目录约定

训练数据目录需包含 `train.txt`、`val.txt`、`labels.json`、`classes.json`；图像目录由 stem 对应 JPG。分割分数缓存需含 `metadata.json`、`complete.json` 及 train/val 索引与分块。训练初始化还需要 MobileNetV3-Large 预训练权重。请使用私有数据存储，不要把这些文件复制进 Git。

推荐先复制 `configs/local.env.example` 为本地私有配置并设置数据、缓存、预训练权重路径；不要将真实路径或凭证写入公开提交。现有服务器快照中的路径配置在打包时已尽可能外移，但在使用前必须人工核对每个数据加载入口。

## 建议复现步骤

1. 获得数据、标签、预训练权重和分割模型的使用授权，并在私有位置准备 train/val/test 清单。
2. 检查 split 无重复、无 train/val/test 泄漏；记录清单哈希与类别支持数。
3. 固定环境版本、随机种子、输入尺寸、分割阈值、训练参数和 checkpoint 选择规则。
4. 先跑一个小批次检查 RGB/分数图尺寸严格一致，再分别训练 A/B/C。
5. 仅用验证集选 checkpoint；锁定后只在测试集评估一次，公开报告每类及整体 TP/FP/FN/TN、P/R 和支持样本数。
6. 汇总各产线/相机的误报时，先确认授权并只分享匿名聚合统计。

## 上传 GitHub 前清单

- 确认代码、训练数据和标签是否获准公开；暂不添加许可证，直到确认代码归属与授权。
- 运行 `git status --short` 和大文件/敏感信息扫描；确认无图片、权重、缓存、逐图 CSV、日志、凭证、内网路径或个人标识。
- 确认 README 的结果口径与汇总表一致，给模型文件和数据集提供可追溯但不泄密的引用。
- 公开版本将专有数据路径改为环境变量，并在干净环境中按许可可用的数据验证运行说明。
