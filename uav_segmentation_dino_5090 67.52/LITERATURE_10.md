# 面向本竞赛的10篇论文与实施选择

检索核对日期：2026-09-19。任务：RGB无人机八类监督语义分割，0忽略、1–8有效类别，单张租用RTX5090。用户报告DINO4090官方67.40；5090方案暂无官方成绩。相关性优先于单纯年份，兼顾近作和可直接落实的经典方法。

| # | 论文与原始来源 | 年份 | 对当前工程的启发 | 实施决定与限制 |
|---|---|---|---|---|
| 1 | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | 2023预印本 | 密集视觉表征作为基础，减少从小数据重新学语义的成本 | 保留L/14及现有LoRA；现有67.40基线支持继续做受控改动，并不证明新头必然提升 |
| 2 | [DINOv3](https://arxiv.org/abs/2508.10104) | 2025 | Gram anchoring保护预训练密集特征，值得研究更强骨干 | 暂不替换；官方权重需授权，patch16/特征提取接口及竞赛许可需另核对。Gram anchoring不能被简单监督损失冒名替代 |
| 3 | [UNetFormer: A UNet-like Transformer for Efficient Semantic Segmentation of Remote Sensing Urban Scene Imagery](https://arxiv.org/abs/2109.08937) | 2022期刊，2021预印本 | UAVid/LoveDA等遥感场景中的全局与局部互补 | 最直接的领域参考；保留DINO全局特征+卷积局部先验，不复制其ResNet编码器或完整注意力解码器 |
| 4 | [Vision Transformers for Dense Prediction (DPT)](https://arxiv.org/abs/2103.13413) | 2021 | 重组多层ViT特征，并逐级恢复密集输出 | 当前由粗到细空间融合的依据；这不是DPT完整复现 |
| 5 | [Vision Transformer Adapter for Dense Predictions](https://arxiv.org/abs/2205.08534) | 2022预印本，ICLR2023 | 卷积空间先验补充普通ViT密集预测 | 当前RGB多尺度先验已借鉴思路，未实现其可变形注意力注入；本轮不重复新增同类模块 |
| 6 | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | 2021 | 轻量跨层解码，注意精度与成本 | 旧4090为风格借鉴，并非MiT。DINO四层同网格不能称原生多尺度；不再加第二个SegFormer骨干 |
| 7 | [Rein++: Efficient Generalization and Adaptation for Semantic Segmentation with Vision Foundation Models](https://arxiv.org/abs/2508.01667) | 2025预印本 | 小规模分割数据和域偏移下的参数高效适配 | 保留场景域验证与LoRA；LoRA不等于Rein实例token。未引入SAM或目标域无监督适配，避免未经规则许可利用测试集 |
| 8 | [Memory Efficient Transformer Adapter for Dense Predictions (META)](https://arxiv.org/abs/2502.01962) | ICLR2025 | 局部分支与内存效率共同考虑 | 当前卷积分支已经覆盖局部先验需求；不复制其cross-shaped attention，5090显存/时长仍须测速 |
| 9 | [PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers](https://arxiv.org/abs/2206.02066) | CVPR2023，2022预印本 | 语义与边缘信息互补，避免融合中过度平滑 | 新增训练用1通道边界辅助头和均衡BCE。是边界监督思想的简化实现，不是PIDNet三分支或其边界引导融合 |
| 10 | [The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks](https://arxiv.org/abs/1705.08790) | CVPR2018，2017预印本 | 使用IoU的可优化替代目标 | 复用仓库已有Lovasz函数，增加可配置0.2权重，保留CE+0.3Dice；不承诺比赛IoU一定提升 |

## 本轮实际改动

当前5090已有逐级融合，因此主改动限定为两个监督信号：

- `L = L_weighted_CE + 0.3 L_Dice + 0.2 L_Lovasz + 0.1 L_boundary`。0.2/0.1为工程起始超参，非论文推荐或已调优值。
- Lovasz在有效像素上按当前batch计算，并平均当前存在的类别；与完整数据集IoU并非严格等价。排序会增加训练耗时，需要重新测速。
- 边界标签由水平/垂直相邻有效类别差异生成，同时标记两侧像素。忽略像素及周围3×3邻域不监督，裁剪边缘不当作类别边界。
- 边界正负像素分别取均值再平均；全背景/全边缘/全忽略情况均处理。该定义是独立工程设计，不声称与PIDNet完全一致。
- 辅助头接在最终1/4解码特征，监督梯度进入共享解码器；推理不执行边界头，不做边界阈值后处理。
- 保持crop448/width96、最后4层LoRA、30轮、固定数据划分和提交协议。这样本轮不再同时改变骨干和分辨率。

## 倒金字塔的判断

本工程将“倒金字塔”解释为从粗到细扩大空间网格（约1/32→1/16→1/8→1/4），通道保持一致。
它可能改善细节，但低分辨率DINO特征插值本身不能创造细节；RGB分支和监督才提供补充。现有设计没有证明优于旧4090融合头，需以相同训练口径比较。

## 最小对照矩阵（共享一份模型代码）

| 配置 | 裁剪/宽度 | 边界 | Lovasz | 目的 |
|---|---|---|---|---|
| decoder_only.json | 392/64 | 0 | 0 | 相对旧4090架构比较，继承相同训练超参 |
| pyramid_control.json | 448/96 | 0 | 0 | 本轮两项监督改动的控制组 |
| boundary_only.json | 448/96 | 0.1 | 0 | 单测边界 |
| lovasz_only.json | 448/96 | 0 | 0.2 | 单测IoU替代损失 |
| train.json | 448/96 | 0.1 | 0.2 | 联合候选，尚非验证胜出方案 |

同种子/划分/训练轮数比较domain mIoU、random mIoU、每类IoU、显存和总时长。官方67.40与本地验证分数单独记录。
有限预算优先跑控制组与联合候选；如果联合下降，再运行单项定位，不应盲目添加更多模块。
没有在本轮启动付费训练；不能把CPU反向通过当作准确率验证。若已有运行，修改代码/配置后须新实验，不可混用旧断点。

## 来源可靠性与缺口

10篇均核对论文原始arXiv记录/摘要；重点实现另核对[PIDNet作者损失代码](https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py)和[Lovasz作者实现](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py)。未完整复现这些论文。
CVF Lovasz摘要页读取失败，使用原始arXiv与作者代码核对。未使用第三方解读作为方法依据。
DINOv3权重申请与许可见[官方仓库](https://github.com/facebookresearch/dinov3)。未获得本比赛完整规则，不推断额外数据/测试时适配获准。
