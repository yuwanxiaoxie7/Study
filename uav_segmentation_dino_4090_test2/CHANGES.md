# 相对旧DINO最佳版的变化

- 来源：uav_segmentation_3050/runs/restart_20260915_120100_012816/E1_sampling02；原始配置存于reference/source_config.json（仅历史记录，旧绝对路径不会作为运行输出）。没有复制历史checkpoint。
- 骨干S/14→L/14：384→1024维、12→24层，层索引[2,5,8,11]→[5,11,17,23]。公开预训练严格加载全部键；旧S权重不能用于L。
- 仍在最后4个block的QKV上使用rank8、alpha16 LoRA。骨干其余参数冻结；最后两层抽取特征仍使用旧SpatialAdapter。
- 默认解码器改为每层1×1投影→同网格拼接→1×1融合、GroupNorm/GELU→上采样→旧图像细节分支→8类输出。宽度保持64。是受SegFormer启发的DINO适配，并非官方MiT+SegFormer或完整论文复现。GroupNorm适合batch1。未虚构天然多尺度。
- 原legacy解码器仍可通过配置选择；这是一个源码实现内的对照开关，不是双份工程。
- 归一化、随机缩放/旋转/色彩增强、20%稀有采样、训练集类别权重、CE+0.3Dice均沿用旧DINO。Ignore的外层控制器使用255，在旧损失入口转换为-1。
- 训练crop392、有效batch8、tile448/stride224、Hann下限0.05、logit融合均保留。
- 旧3轮LR3e-5/3e-6是续训；新head随机初始化，使用30轮LR3e-4/3e-5、warmup100。与旧历史分数的差异不能全部归因于解码器。三个新配置用同一训练配方对照。
- Linux workers4；保持完整训练集稀有采样池，样本种子基于全局索引，恢复后不会换采样池。隔离数据增强对主进程Python随机状态的影响。
- AMP优先BF16；硬件不支持时auto使用FP16，真实算子不兼容则停止并要求显式更换precision。trainable blocks启用非重入activation checkpoint。官方attention用PyTorch SDPA，不安装xFormers。
- 工程控制器复用4090版经过测试的日志、原子保存、恢复、双验证、ZIP检查流程；模型、数据增强、损失与旧DINO同源。
- 时间实测估算加20%余量；超过18小时或实测allocated显存超过22GiB停止。CUDA上下文和其他进程还会占显存，不承诺22GiB以内一定可运行。
- 没有添加边界loss、Lovasz、EMA、TTA、模型集成或测试集拟合。
