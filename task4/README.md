# 视网膜血管分割

使用20张训练眼底图及二值标签学习血管分割，预测20张测试图并输出512×512掩膜与RLE提交。

## 模型结构

采用从零训练的小型U-Net。输入四通道：视野内归一化RGB，以及绿色通道减去sigma=8高斯背景的局部对比度。改进配置编码通道24/48/96，瓶颈192，解码96/48/24；同尺度特征跳跃拼接，1×1卷积输出血管logit。损失为BCE与soft Dice之和，不使用外部训练数据或预训练权重。

模型、数据和提交实现位于`vessel/model.py`、`vessel/data.py`和`vessel/submission.py`。`vessel/augment.py`保留现有几何增强实现，是否启用由训练参数控制。

## 实验方案

先按整图固定16/4划分，再从训练图取切块。验证图为8、10、15、16，seed与split_seed均为42。比较宽度16、128切块基线与宽度24、192切块配置，另比较四方向翻转、八方向D4及两模型等权融合。

实际标签为白色血管255、背景0，官方编码器取非零像素为前景。提交转换为原尺寸阈值化后双线性缩放到512，再取非零像素。实验同时报告按此转换的逐图平均Dice与原尺寸Dice，主指标固定为前者。

## 训练过程

课程数据放在`segmentation/train/image`、`segmentation/train/label`及`segmentation/test/image`，保留提交模板。文件后缀可能与实际TIFF/GIF内容不同，由图像库正常解码。

改进配置batch=12，每轮50步、上限120轮。AdamW学习率0.001、weight decay=1e-4，余弦最低学习率0.00002。每5轮完整图验证，在0.2–0.75、步长0.025的阈值中选取平均Dice最高者。第65轮和0.400阈值入选，全量20图重新训练65轮，余弦周期仍保留120轮；预测使用八方向逆变换概率平均并排除视野外区域。

```powershell
python audit_data.py
python train.py --width 24 --patch-size 192 --batch-size 12 --epochs 120 --steps-per-epoch 50 --seed 42 --split-seed 42 --tta d4 --output outputs/validation
python train.py --all-data --selection outputs/validation/summary.json --output outputs/final
python predict.py --checkpoint outputs/final/model.pt
python verify_submission.py
```

官方`segmentation/segmentation_to_csv.py`保留用于编码对照。`evaluate_checkpoint.py`复算验证，`compare_models.py`读取既有基线与改进产物比较策略。预测掩膜使用PNG，不将黑色血管显示图输入RLE编码器。

## 分析过程与结果

原基线转换后Dice为0.848591、原尺寸0.813110；大切块宽模型与D4为0.851209、0.815958，转换后增加0.002618。仅基线换D4为0.848868，两模型等权融合为0.850760，均低于所选单模型。四张图均改善，但图15增加0.005344、图8仅0.000074。

四图同时用于轮次、阈值和策略选择，数值属于开发集对照。后续本地微调候选达到0.851530，但原尺寸降至0.815883，尚无对应线上反馈；已有上一版线上反馈0.85914。此处保留差异，不以小幅本地增加宣称泛化改善。提交检查包含20张官方编码器一致和RLE解码往返。训练数据、权重及验证产物需自行生成，未随源码上传。
