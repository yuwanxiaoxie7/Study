# 论文与实施选择（2026-09-18）

本报告比较论文与作者代码，性能判断是针对当前工程的工程推断，不是本赛事已完成实验。没有足够证据宣称任何组合“最合适”或必涨分。

## 候选比较

| 候选 | 依据与潜在收益 | 显存/速度及风险（未实测） | 决策 |
|---|---|---|---|
| DINOv2-S + 原解码器 | [DINOv2，2023](https://arxiv.org/abs/2304.07193)，延续旧模型；约21M骨干 | 相对最低，旧成绩可作参考，但新配方不等于旧续训 | 提供small_legacy对照 |
| DINOv2-L + 原解码器 | 同论文和[官方实现/权重](https://github.com/facebookresearch/dinov2)，更大语义表征 | 约300M骨干；显存/时间明显增加；LoRA限制优化器开销 | 提供large_legacy，隔离骨干变化 |
| DINOv2-L + SegFormer风格融合 | [SegFormer，NeurIPS2021](https://arxiv.org/abs/2105.15203)，轻量多层线性投影与融合 | head开销较小，空间细节仍受patch14限制；能否优于旧FPN未知 | 主候选，真实GPU实测后才能长训 |
| DINOv2-L + DPT | [Vision Transformers for Dense Prediction，ICCV2021](https://arxiv.org/abs/2103.13413)、[官方DPT](https://github.com/isl-org/DPT)；token重组、多分辨率逐级融合 | 比简单融合复杂；需要额外训练参数和消融 | 后续候选，不在本次叠加 |
| DINOv2-L + UPerNet | [Unified Perceptual Parsing，ECCV2018](https://arxiv.org/abs/1807.10221)、[作者实现](https://github.com/CSAILVision/unifiedparsing)；多尺度上下文聚合 | DINO仍须构造金字塔；head较复杂，不能直接套Swin接口 | 暂不采用 |
| DINOv3-L + 分割头 | [DINOv3，2025](https://arxiv.org/abs/2508.10104)、[官方实现](https://github.com/facebookresearch/dinov3)；强调高质量稠密特征，另有卫星预训练 | L约300M，patch16；申请权重、自定义许可、新环境；卫星域不等于低空域 | 已核实存在，暂不作为默认部署 |
| UNetFormer | [论文，ISPRS2022](https://arxiv.org/abs/2109.08937)、[GeoSeg官方代码](https://github.com/WangLibo1995/GeoSeg)；局部/全局解码和遥感适配 | 换整套结构影响因素多；GeoSeg GPL-3.0 | 不直接移植代码 |
| AerialFormer | [论文，2023预印本](https://arxiv.org/html/2306.06842v2)、[作者代码](https://github.com/UARK-AICV/AerialFormer)；多分辨率和局部信息 | 更换编码/解码方案，依赖适配和预算风险较高 | 研究参考，未实现 |

以上显存只作相对排序。L骨干FP32参数约1.2GB，但激活、LoRA反向、适配器梯度、workspace和加载临时副本远超过参数本身；不提供未经测量的整网GiB或小时数字。正式门槛：24GB卡、峰值allocated≤22GiB、总时长估算×1.2≤18小时。以benchmark.json为准。

## 为什么主选 DINOv2-L + SegFormer风格融合

SegFormer论文的MLP解码器用多级特征投影与融合降低解码复杂度；官方完整网络的MiT是层次化编码器。DINO并非MiT，因此本项目独立实现适配：取L的第6/12/18/24层，CLS/register排除后恢复同尺寸二维网格；四组1024通道投影到64后拼接融合。上采样至1/4尺度并加原DINO图像细节分支，最后输出8类。这是多深度融合，不声称上采样凭空生成多尺度细节。

保留原空间适配器与细节分支，是减少变化因素的工程选择。采用GroupNorm/GELU而非照搬官方batchnorm，适应batch1。论文不证明该改造在本比赛必优于原FPN；必须用L+legacy同配方对照。官方源码的研究用途条款见[NVIDIA SegFormer许可](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)；没有复制其源码或加载其分割预训练。

## DINOv3核实与取舍

DINOv3已公开，官方仓库包含语义分割线性探测和多种骨干。官方README说明需申请模型下载地址；有LVD-1689M和SAT-493M来源，二者不可混淆。本次未取得用户获批权重。其[setup.py](https://github.com/facebookresearch/dinov3/blob/main/setup.py)要求Python≥3.11，与本次Python3.10部署约束不同；采用[DINOv3专用许可](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)。这不是“模型不成熟”的结论，而是当前交付条件的选择。DINOv2标准自然图像模型官方代码与权重声明Apache-2.0，下载无需本次申请流程。

## 权重与来源

- L默认官方地址：https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
- S对照官方地址：https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
- 两者来自官方backbones.py路径构造及README。只用骨干，不用ADE20K等外部分割标注head。
- 本地已有官方DINOv2 checkout commit：7764ea0f912e53c92e82eb78a2a1631e92725fc8；复制需要的Python文件，保留LICENSE，未改vendor源码。原checkout git status为空。
- 下载记录URL、字节数和SHA256。公开上游独立SHA256未取得，局部摘要只能验证后续未改动；骨干load_state_dict(strict=True)拒绝漏键和错维度，不退回随机初始化。

## 实验与证据边界

固定5178训练、1049随机验证、769域验证；类别权重/稀有池只读训练部分。保留旧增强和损失；三个配置都是30轮从公开骨干初始化。S历史最佳经历过前序训练加3轮微调，不能直接声称新30轮是单变量复现。先比较L+legacy与L+segformer，再看官网提交。

官网65.82属于Swin工程。旧DINO的约66分只收到用户回忆，尚无可定位的提交记录；不拿本地72.12或64.20冒充官网分数。

已访问[赛事页面](https://www.aicomp.cn/tracks/tracks-1/3708.html)，网页工具未取得预训练/集成规则完整条文，故没有宣称已验证比赛许可。保持仅比赛训练图、官方自监督骨干、单模型，不做测试标签调参。若正式规则对大规模公开预训练有限制，需按规则调整。

检索失败/缺口：未取得本赛事逐类官方反馈；未取得原66分提交映射；DINOv3权重未申请；没有4090实测。此前Wigolo未连接，本次使用网页工具核对论文和作者仓库。
