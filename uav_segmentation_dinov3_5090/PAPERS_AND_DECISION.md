# 论文与当前取舍（2026-09-20）

用户在实现期间要求重新考虑DPT，本版采用独立实现的轻量UPerNet式解码器：四个深度层投影，卷积空间先验产生1/4至1/32尺度，PPM池化汇聚，FPN自顶向下融合，全层拼接输出。用GroupNorm适配小batch，无自定义CUDA扩展。

1. **DINOv3** (2025): https://arxiv.org/abs/2508.10104 。使用官方预训练骨干；Gram anchoring属于其预训练方法，本项目不重新实现大规模预训练。
2. **Unified Perceptual Parsing for Scene Understanding / UPerNet** (ECCV 2018): https://arxiv.org/abs/1807.10221 。采用PPM+FPN语义分割思想；RGB先验、门控和GN属于本工程改动，不宣称复现原模型。
3. **MM-DINO** (TGRS 2026): https://github.com/KimotaQY/MM-DINO ，论文DOI https://doi.org/10.1109/TGRS.2026.3677346 。直接相关的DINOv3遥感分割工作；参考适配与细化，不把其遥感数据集分数当成本竞赛预测。
4. **UNetFormer** (ISPRS 2022): https://arxiv.org/abs/2109.08937 。有UAVid/LoveDA上的验证；可作为后续全局-局部注意力解码器对照，本版不实现其窗口注意力。
5. **SegFormer** (NeurIPS 2021): https://arxiv.org/abs/2105.15203 。低开销多层融合可作为基线；用户希望换解码器，本版改为PPM+FPN。
6. **Vision Transformers for Dense Prediction / DPT** (ICCV 2021): https://arxiv.org/abs/2103.13413 。能够用于ViT密集预测，不应认为“只适合深度估计”。用户选择重新评估，因此本轮不采用。重采样金字塔有效与否要实验验证，插值本身不产生新细节。
7. **ViT-Adapter** (ICLR 2023): https://arxiv.org/abs/2205.08534 。参考局部空间先验动机；本版卷积分支不是原论文的可变形注意力交互适配器。
8. **Mask2Former** (CVPR 2022): https://arxiv.org/abs/2112.01527 。DINOv3官方分割代码明确提供Adapter+Mask2Former路径；并非“不适合8类语义分割”。暂缓原因是训练匹配、像素解码器与依赖适配更复杂，需要单独成本实测。

官方实现与权重要求：https://github.com/facebookresearch/dinov3 。Web/SAT具有不同归一化；ViT-L的SAT结构还启用local_cls_norm。调用枚举选择结构，用原始官方state_dict strict加载。官方源固定到6876159a11b4df116f30f667f8c9888617df0751，文件SHA见reference/dinov3_source.json，许可证见vendor/dinov3/LICENSE.md。

选择UPerNet式头是工程假设，尚无证据证明超过DPT、SegFormer、旧DINO或本竞赛官方分数。当前缺少已授权DINOv3权重及服务器测速。保持既有数据划分与增强便于比较，但主干与解码器同时变化时不能归因于单一模块。
