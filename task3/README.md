# 眼底图像中心凹定位

根据眼底图像预测中心凹原图像素坐标。训练80张、测试20张，提交列为`ImageID,value`，每张图的X/Y交替写入，共40行。

## 模型结构

ResNet18或ConvNeXt-Tiny提取四级空间特征，各级用1×1卷积投影到64通道，再逐层上采样相加。融合特征经过64→64、64→32卷积及单通道输出，生成定位热图。支持区域内以温度0.25的概率期望解码，按原图宽高恢复像素坐标。

采用ResNet18-384、ResNet18-512和ConvNeXt-Tiny-384三个分支，坐标权重分别为0.25、0.25、0.50。实现位于`fovea/model.py`、`supported_decoder.py`、`metric_objective.py`和`optimized.py`。热图交叉熵与宽高加权坐标MSE/2500共同训练；现有已采用模型的坐标损失使用增强后残差，未先逆旋转缩放。可选逆仿射实现保留为对照，不改写已有结果。

## 实验方案

坐标CSV为训练真值，XML框用于审计。训练图感知哈希聚为58组，重复图0039和0077同组；未提供患者ID，图像分组不等于患者隔离。初版68/12留出，后续固定三折留出26、27、27张，seed=1701，每张图一次OOF。每折固定预算，比较区域约束、解码、坐标损失、分辨率和骨干融合。

RMSE按全部坐标残差先求均方再开方，即`sqrt(mean((prediction_xy - true_xy)**2))`；它不同于平均欧氏距离，合并OOF也不能直接平均各折RMSE。另检查中位距离、最大误差、逐图贡献及排除困难图后的结果。

## 训练过程

课程数据位于detection/，保留训练图、测试图、坐标CSV、XML和sample_submission.csv。ImageNet参数需自行准备，放在相邻task1/outputs/torch_cache/hub/checkpoints/目录：resnet18-f37072fd.pth与convnext_tiny-983f1562.pth。可用train.py --pretrained指定初版权重；后续编排沿用上述相对缓存位置。

初版384输入、batch=8，训练80轮，原实验选第61轮；全量保留80轮余弦周期训练61轮。后续三折比较固定61轮，ResNet骨干学习率1.5e-4、ConvNeXt为3e-5，定位头1e-3，AdamW、weight decay=0.01、梯度裁剪5。增强同步变换图像与坐标。

从课程数据开始，依次运行：

```powershell
cd task3
python -m pip install -r requirements.txt
python baseline.py
python resolution_screen.py
python local_refinement.py
python export_optimized.py
python cross_validation.py
python decoder_comparison.py
python coordinate_training.py
python resolution_comparison.py
python pipeline.py
```

baseline.py生成records.csv、split.json、初版权重和submission.csv；resolution_screen.py生成512/640对照；local_refinement.py完成局部精定位对照；export_optimized.py比较原优化候选并生成optimization_summary.json、bundle和submission_optimized.csv。cross_validation.py由这些已生成配置建立三折审计；其后依次完成有效区域解码、坐标联合损失、384/512融合和ConvNeXt固定融合。

优化导出与分辨率比较不读取人工成绩JSON，平台反馈单独保存，不作为训练或选型的运行前提。每个编排入口均支持--help；已有最终模型后，可执行python pipeline.py --predict-only重新生成submission_backbone.csv，或用predict_ensemble.py --bundle指定融合配置。迁移既有产物时需相应调整JSON中的检查点路径。

原始数据与训练权重未随源码发布；中间产物由上述阶段生成。本次修复验证了缺失阶段的调用及导出行为，没有重新运行整套GPU训练。

## 分析过程与结果

初版12图RMSE为27.2955，640实验降至19.3751，平台反馈却从12.66810退步至13.43397，说明小留出集选型不稳定。有效区域、联合坐标损失及384/512融合后，三折OOF为17.1585；新增ConvNeXt固定融合降至16.4101，对应已有公开榜反馈11.55982。

改善集中于困难图：38/80图距离降低，中位距离从7.6701升至8.3533，同时排除0049、0072后其余78图RMSE由11.5994升至11.9360。最新JS候选虽到16.2394，排除最大误差图后退步且未达到3%采用门槛，未替换已采用方案。80个训练标签均非(0,0)，当前未验证不可见判别。验证划分多次参与开发，本地指标与平台反馈分别解释。
