# 机器学习课程设计

四项计算机视觉实验的源码与运行说明。任务编号与《机器学习课程项目报告》的章节对应。

| 报告章节 | 任务 | 报告中的本地结果 | 阅读入口 |
| --- | --- | --- | --- |
| 第一章 · Task 1 | 植物图像五分类 | seed 42 五折 OOF micro-F1 **0.956** | [实验说明](task1/README.md) · [模型实现](task1/plant_classifier/modeling.py) |
| 第二章 · Task 2 | 人脸表情识别 | 固定分组验证准确率 **72.38%** | [实验说明](task2/README.md) · [实验流程](task2/pipeline.py) |
| 第三章 · Task 3 | 眼底中心凹定位 | 三折合并 OOF RMSE **16.4101 像素** | [实验说明](task3/README.md) · [实验流程](task3/pipeline.py) |
| 第四章 · Task 4 | 视网膜血管分割 | 四图验证转换后平均 Dice **0.851209** | [实验说明](task4/README.md) · [训练入口](task4/train.py) |

OOF 指汇总各折留出样本的预测。各任务 README 说明验证条件、模型版本和报告结果的来源。

## 运行准备

实验环境为 Windows、Python 3.12、PyTorch 2.7.0+cu126 和 RTX 4060 Laptop GPU。先按各任务说明准备课程数据，再进入对应目录安装依赖。以第一章为例：

```powershell
git clone https://github.com/lee11-star/MLdesign.git
cd MLdesign/task1
python -m pip install -r requirements.txt
python train.py --help
```

| 任务 | 初始化参数 |
| --- | --- |
| Task 1 | DINOv2-Small |
| Task 2 | ImageNet 和 DAN 公开参数 |
| Task 3 | ImageNet 预训练骨干 |
| Task 4 | 随机初始化 |

数据目录、权重路径、阶段顺序及验证命令见各任务 README。请保留四个任务目录的相对位置，部分脚本会读取共用的预训练缓存。

## 查阅实验记录

仓库提供训练、推理、验证代码和依赖清单。脚本将训练日志、模型权重、预测文件和验证结果写入各任务的 `outputs/`。报告附录列出所引用的本地记录及对应文件名，复查时可按文件名定位。原始课程数据、训练产物和提交报告保存在本地。

DAN 模块许可见 [DAN-MIT.txt](task2/emotion_classifier/licenses/DAN-MIT.txt)。