# 人脸表情识别

将48×48灰度人脸图像分为Angry、Fear、Happy、Sad、Surprise、Neutral六类，标签依次为0–5。原始训练图28275张，测试图7065张。

## 模型结构

基础模型包括CNN、ImageNet ResNet18和ConvNeXt-Tiny，输出层均替换为六分类。最终新增DAN分支：ResNet18提取空间特征，四个空间与通道注意力头各输出512维特征，沿头维log-softmax求和后，经过512→6线性层与BatchNorm输出分类结果。只迁移公开AffectNet7特征与注意力参数，原七分类头丢弃。

最终概率为`0.75 × 七模型基线 + 0.25 × DAN`，采用原图与水平翻转TTA。七模型由常规ConvNeXt、两种ResNet、两种输入尺度正则化ConvNeXt和两个MixUp种子组成。模型在`emotion_classifier/models.py`及`facial.py`，训练在`training.py`，概率融合在`ensemble.py`。

## 实验方案

按文件哈希将相同图像分组，39个标签冲突组共110张整组排除，余28165张。首个分层分组折划分为22532训练、5633验证，训练与验证哈希交集为0。额外分组折用于补充验证，不作为从未使用过的独立测试集。

在七模型基线上比较DAN普通交叉熵、可信度缩放与排序正则，以及两者固定25%融合，共5项，不连续搜索融合权重。主终点固定为第30轮翻转TTA。总体准确率、各类召回、混淆矩阵、纠正与新增错误数共同用于分析。

## 训练过程

标准增强包含水平翻转、±10°旋转、±8%平移和0.9–1.1缩放，再调整模型输入。DAN输入224、batch=64、训练30轮，分类头学习率3e-4、其余参数3e-5，AdamW、weight decay=1e-4。普通DAN不启用标签平滑、MixUp或额外头dropout；可信度候选增加top70%排序间隔0.15的正则，原标签不变。

先将课程数据及提交范式按`emotion_classifier/data.py`中的`DEFAULT_DATA`、`DEFAULT_TEMPLATE`放置，再执行审计生成manifest和像素缓存。公开DAN检查点来源为[作者仓库](https://github.com/yaoing/DAN)，需自行准备到`outputs/round8/pretrained/dan_affectnet7.pth`。全量训练使用清洗后28165张图，固定30轮。

```powershell
python audit_data.py --help
python train.py --help
python predict.py --help
python pipeline.py --help
python verify.py
```

`train.py`可独立训练`--model dan --image-size 224 --epochs 30 --batch-size 64 --lr 0.0003 --backbone-lr-scale 0.1 --weight-decay 0.0001`，并通过`--pretrained`指定参数文件。`predict.py --checkpoint <模型路径> --tta`输出单模型结果。

`baseline_training.py`建立早期训练流程，`model_selection.py`和`baseline.py`构建融合基线；`feature_transfer.py`、`generalization.py`保留补充对照的必要依赖。最终`pipeline.py`依赖既有七模型概率、原折与额外折manifest及公开DAN参数，不能在缺少前置产物时仅凭一次命令重建全部历史实验。原实验中的`outputs/round4`、`round7`、`round8`为产物位置，保留以对应模型配置。

## 分析过程与结果

七模型基线原折准确率71.77%；普通DAN单模型69.86%，固定比例融合后72.38%，纠正115张旧错例并新增81张错误，净增34张正确。Fear召回由52.03%下降至49.45%，总体收益不意味着六类同时改善。可信度模型单独69.54%、固定融合72.22%，未优于普通DAN。

七模型版本的已有平台反馈为0.733，DAN融合平台成绩尚无记录。原折长期参与选参，bootstrap区间未校正选择偏差。`analysis.py`、`evaluate.py`、`verify.py`用于查看难类、复算预测和核对提交。训练不使用外部图像、伪标签或重新标注；DAN改编许可见`emotion_classifier/licenses/DAN-MIT.txt`。
