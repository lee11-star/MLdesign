# 植物图像分类

根据植物照片预测Black-grass、Common wheat、Loose Silky-bent、Scentless Mayweed、Sugar beet五类。训练集500张，每类100张；测试集378张。

## 模型结构

采用DINOv2-Small（ViT-S/14）骨干，输入504×504 RGB图像，得到384维图像特征，经过LayerNorm与384→5线性层输出五类logits。训练使用标签平滑交叉熵。最终使用seed=42、3407、2026三个全量模型的Softmax概率平均，不使用TTA。

主要实现位于`plant_classifier/modeling.py`，训练在`plant_classifier/train.py`与`train_full.py`，推理在`plant_classifier/infer.py`。

## 实验方案

先检查图片可解码性、类别映射、文件重复及提交ID，再以seed=42固定400/100分层划分筛选。入选配置完成五折分层交叉验证，每折训练400、验证100张，每张训练图仅由对应未见过它的折模型产生OOF预测。

比较常规卷积骨干、DINOv2冻结探针与微调、392/504输入、强弱增强及概率融合。micro-F1为主指标，本任务单标签分类下它与accuracy相等，另报macro-F1和逐类混淆。

## 训练过程

图像先居中补方再缩放，以保留细长叶片；训练执行翻转、轻旋转、颜色与仿射扰动，验证仅做固定缩放和归一化。AdamW骨干学习率5e-6、分类头5e-4，weight decay=0.01，标签平滑0.02，batch=8。第一轮冻结骨干，之后解冻，梯度裁剪1.0；余弦周期20轮、早停耐心6轮。

五折最佳轮次为10、13、5、10、8，取中位数10用于全量重训；余弦周期仍为20轮。数据由程序查找，课程数据可按原结构放在`2026task1/dataset-for-task2/dataset-for-task2/`，内含`train/<类别>/`和`test/`，同时保留提交范式。

```powershell
python data_audit.py --root . --output-dir outputs/audit
python train.py --help
python train_full.py --help
python predict.py --help
python validate_submission.py submission.csv
```

`train.py`配置`--model dinov2_vits14 --pretrained --image-size 504 --epochs 20 --batch-size 8 --lr 0.000005 --head-lr-multiplier 100 --freeze-backbone-epochs 1 --weight-decay 0.01 --label-smoothing 0.02 --transform-mode fine_grained --grad-clip 1 --patience 6 --seed 42`；用`--fold 0`至`--fold 4 --num-folds 5`分别训练五折。全量入口改用`train_full.py --epochs 10 --scheduler-epochs 20`，分别设置三个种子。各命令完整参数以`--help`为准。CPU快速模式会降低分辨率与轮次，正文成绩对应GPU配置。

## 分析过程与结果

Small504的OOF micro-F1为0.956、macro-F1约0.95603，500张中判错22张。Black-grass与Loose Silky-bent互判9+7=16张，是主要剩余问题。与Small392的0.950相比，504输入纠正8张、退步5张，净增3张正确。Base504为0.954，强增强Small504为0.950，均未替换Small504。已有提交反馈为96.56%，它与本地OOF分开记录。

验证标签参与了配置和轮次选择，OOF用于比较这些具体方案；最终三种子全量模型没有另做三组完整OOF。测试概率、混淆矩阵、逐类报告和提交核验由对应脚本输出到`outputs/`，本仓库不预置这些训练产物。
