# 人脸表情识别

将48×48灰度人脸图像分为Angry、Fear、Happy、Sad、Surprise、Neutral六类，标签依次为0–5。原始训练图28275张，测试图7065张。

## 模型结构

基础模型包括CNN、ImageNet ResNet18和ConvNeXt-Tiny，输出层均替换为六分类。最终新增DAN分支：ResNet18提取空间特征，四个空间与通道注意力头各输出512维特征，沿头维log-softmax求和后，经过512→6线性层与BatchNorm输出分类结果。只迁移公开AffectNet7特征与注意力参数，原七分类头丢弃。

最终概率为`0.75 × 七模型基线 + 0.25 × DAN`，采用原图与水平翻转TTA。七模型由常规ConvNeXt、两种ResNet、两种输入尺度正则化ConvNeXt和两个MixUp种子组成。模型在`emotion_classifier/models.py`及`facial.py`，训练在`training.py`，概率融合在`ensemble.py`。

## 实验方案

按文件哈希将相同图像分组，39个标签冲突组共110张整组排除，余28165张。首个分层分组折划分为22532训练、5633验证，训练与验证哈希交集为0。额外分组折用于补充验证，不作为从未使用过的独立测试集。

在七模型基线上比较DAN普通交叉熵、可信度缩放与排序正则，以及两者固定25%融合，共5项，不连续搜索融合权重。主终点固定为第30轮翻转TTA。总体准确率、各类召回、混淆矩阵、纠正与新增错误数共同用于分析。

## 训练过程

先在仓库根目录进入task2，安装requirements.txt。课程数据、模板和ZIP路径沿用emotion_classifier/data.py与audit_data.py的默认值，可用--help查看。

主流程需要在相邻task1/outputs/torch_cache/hub/checkpoints目录准备resnet18-f37072fd.pth、convnext_tiny-983f1562.pth、efficientnet_b2_rwightman-c35c1473.pth三个ImageNet检查点。公开DAN参数来源为[作者仓库](https://github.com/yaoing/DAN)，保存为outputs/round8/pretrained/dan_affectnet7.pth。无需下载AffectNet原图。审计记录直接核对仓库中的emotion_classifier/facial.py，不要求另外准备下载源码快照。

标准增强包含水平翻转、±10°旋转、±8%平移和0.9–1.1缩放。DAN输入224、batch=64、固定30轮，分类头学习率3e-4、其余参数3e-5，AdamW、weight decay=1e-4。普通DAN不启用标签平滑、MixUp或额外头dropout；可信度候选增加top70%排序间隔0.15的正则。

从课程数据开始，按以下顺序生成前置产物；每个阶段完成后再进入下一阶段：

```powershell
cd task2
python -m pip install -r requirements.txt
python audit_data.py
python baseline_training.py
python model_selection.py
python regularization.py
python baseline.py
python generalization.py
python pipeline.py
python verify.py
python validate_submission.py submission_round8.csv
```

baseline_training.py生成CNN/ResNet基线、数据划分和初始提交；model_selection.py训练三项对照并生成optimization产物；regularization.py恢复原96/128输入正则化实验，训练两个40轮ConvNeXt并生成kaggle80选型和提交；baseline.py再生成七模型融合。generalization.py生成额外分组折、固定第30轮参考概率；pipeline.py训练DAN、比较固定25%融合并按原规则决定是否导出。历史outputs子目录名保留以对应报告中的产物。

train.py可独立训练--model dan --image-size 224 --epochs 30 --batch-size 64 --lr 0.0003 --backbone-lr-scale 0.1 --weight-decay 0.0001，并通过--pretrained指定公开参数文件。feature_transfer.py为DINO补充实验，不是上述DAN主流程的运行前提。训练会消耗多轮GPU时间，本次源码修复没有重新训练模型。

## 分析过程与结果

七模型基线原折准确率71.77%；普通DAN单模型69.86%，固定比例融合后72.38%，纠正115张旧错例并新增81张错误，净增34张正确。Fear召回由52.03%下降至49.45%，总体收益不意味着六类同时改善。可信度模型单独69.54%、固定融合72.22%，未优于普通DAN。

七模型版本的已有平台反馈为0.733，DAN融合平台成绩尚无记录。原折长期参与选参，bootstrap区间未校正选择偏差。`analysis.py`、`evaluate.py`、`verify.py`用于查看难类、复算预测和核对提交。训练不使用外部图像、伪标签或重新标注；DAN改编许可见`emotion_classifier/licenses/DAN-MIT.txt`。
