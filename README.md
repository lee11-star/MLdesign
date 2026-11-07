# 机器学习课程设计

四个任务分别放在一个目录中，每个目录的README说明模型结构、实验方案、训练过程与结果分析。

| 目录 | 项目 | 主要入口 |
| --- | --- | --- |
| [task1](task1) | 植物图像分类 | `train.py`、`train_full.py`、`predict.py` |
| [task2](task2) | 人脸表情识别 | `train.py`、`pipeline.py`、`predict.py` |
| [task3](task3) | 眼底中心凹定位 | `train.py`、`pipeline.py`、`predict_ensemble.py` |
| [task4](task4) | 视网膜血管分割 | `train.py`、`predict.py`、`verify_submission.py` |

各目录保留实现包、必要的实验编排模块、测试和`requirements.txt`。历史报告、重复版本目录、原始数据、模型权重及缓存不随源码上传。脚本中的`outputs/`用于保存运行产物，部分实验编排需要先完成前置训练，具体依赖见各任务说明。

## 运行环境

已有实验使用Python 3.12、PyTorch 2.7.0+cu126及RTX 4060 Laptop GPU。进入对应任务目录后安装依赖，例如：

```powershell
cd task1
python -m pip install -r requirements.txt
python train.py --help
```

课程数据需要自行准备。task1使用DINOv2预训练参数，task2使用ImageNet和DAN公开参数，task3使用ImageNet骨干；task4从零训练。部分历史编排保留了原实验机的解释器或权重路径，新机器上需按任务说明调整。源码整理保留`task1`至`task4`的相对位置，便于共享预训练权重缓存。

## 文件约定

- `README.md`：任务说明、方法、实验及复现步骤。
- `train.py`和`predict.py`：单模型训练与推理。
- `pipeline.py`：需要多阶段结果的最终实验编排。
- 实现包：数据、模型、损失、指标与推理函数。
- `tests/`：已有关键行为检查。
- `requirements.txt`：任务依赖。

DAN模块的第三方许可保存在[task2/emotion_classifier/licenses/DAN-MIT.txt](task2/emotion_classifier/licenses/DAN-MIT.txt)。
