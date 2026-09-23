# 第三方来源与实现差异

## OffSeg

`aic/model.py:OffsetLearning` 根据作者的双偏移公式和 PyTorch 实现移植。

- 作者仓库：https://github.com/HVision-NKU/OffSeg
- 固定提交：`a203f52fb66399517c49f5acda3aaf931804036e`
- 对照文件：`mmseg/models/decode_heads/offset_learning.py`、`segformer_head_offset_learning.py`。
- 作者声明见 `third_party/OffSeg_LICENSE.txt`：Apache 2.0 文本之外，作者另列专利与非商业/学术/研究用途限制；不可将其简写为不受限制的 Apache 商用授权。
- 修改：去掉 MMSeg 注册与初始化依赖，以普通 PyTorch 层表达同一双偏移计算；矩阵乘、softmax 强制 FP32；类别为本赛题的 8 类。
- 不声称完整复现：本工程使用 Hugging Face MiT、统一 GN、小批量、统一 dropout 0.1。作者的 offset head 有额外 maxpool2，默认公平对照的两个 head 都使用 1/4 输出；另有成对的 pool2 配置用于检验这一差异。
- 单元测试独立使用 einsum 表达公式并对照输出，测试梯度；这验证移植方程，不等于复现论文的精度。

## MiT / SegFormer

骨干调用 `transformers==4.56.2` 的 `SegformerModel`；公开 ImageNet 预训练权重来自 NVIDIA 的 `nvidia/mit-b4`、`nvidia/mit-b5`。
模型下载来源、固定 revision、文件长度、上游 LFS SHA256 全部在 `aic/weights.py`；运行写出下载收据。
严格加载全部骨干键，仅明确排除原 ImageNet 分类器的 weight/bias，不允许静默随机初始化骨干。
`features()` 为 Hugging Face 编码器添加非重入激活检查点；有与其原生输出一致性的测试。
解码头是小批量 GN 适配版：四层投影包含 Conv/GN/ReLU，融合层 256 通道。它不是原论文所有训练细节的逐项复现，结果应标为本项目基线。

## 金字塔与轴向注意力

`fpn`、`ppm`、`msfe` 是工程消融选项。`ppm` 是普通金字塔池化；`msfe` 这个配置名指 FPN+普通 PPM+ELA 启发的轴向局部注意力。
未核实矿区论文的完整 SFA-PPM 与 MSFE-FPN 细节，因此本实现不使用“原版 SFA-PPM”或“完整复现 MSFE-SegFormer”的称谓，也不套用其论文增益。
FreqFusion、PointFlow、PPTFormer、HRDA 均未被包装成已实现模块。

其余运行依赖遵循各自发行包许可证。项目未对论文作者背书、赛题成绩或任何商业授权作额外声明。
