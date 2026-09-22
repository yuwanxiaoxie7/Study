# 当前执行提示词（采用用户最新解码器方向）

在独立uav_segmentation_dinov3_5090目录实现DINOv3 ViT-L/16固定8类低空图像分割，原工程保留。用户已要求重新考虑DPT，因此本轮采用轻量UPerNet式解码器；不要回退到已放弃的DPT实现。

使用官方DINOv3固定提交，支持SAT/Web权重及对应结构、归一化。四个中间层[5,11,17,23]由官方API提取，处理CLS/storage tokens和RoPE。冻结主干，在后4块融合QKV中的Q/V切片加入rank8/alpha16 LoRA，K及其masked bias保持原样。检查LoRA梯度和零初始化等价性。

解码器采用128通道投影，RGB卷积细节金字塔，sigmoid(-2)初始门控，深层PPM(1,2,3,6)，FPN自顶向下融合及全层拼接。GroupNorm适配小batch；称为UPerNet式独立实现，不冒称论文精确复现。参考DINOv3、UPerNet、MM-DINO、UNetFormer、SegFormer、DPT、ViT-Adapter、Mask2Former并说明取舍，不声称哪一种必然更高。

初始448裁剪、batch2×累积4、30轮、每2轮完整验证、最终强制验证；BF16、训练块checkpoint；AdamW解码器2e-4、LoRA2e-5、weight decay .01，偏置/归一化无衰减，warmup5%+cosine最低1%，grad_clip1。CE+.3Dice+.1Lovasz+.05Boundary，提供关闭辅助损失配置。保持原划分和增强；训练专用边界头推理不执行。

保留一键环境安装→数据校验→权重校验→GPU测速→训练/断点恢复→验证→预测→提交打包。无卡模式可安装/下载。支持prepare-only、check-only及只测速的448/512/batch1/checkpoint对照，不自动选参数或启动完整训练。实际RTX5090型号、显存、驱动、CUDA/PyTorch和BF16算子须实测；预算18小时，超限需明确接受。支持显式兼容环境复用，下载进度、安全续传、重试、验证及授权链接脱敏。

权重必须官方授权或本地原始state_dict；本地SHA与官方摘要分开。训练身份绑定权重内容、配置、划分及训练代码；下载脚本修复不强迫丢弃断点。保存模型、优化器、调度位置、随机状态、日志、逐类指标、环境及源码。

完成CPU单测、真实ViT-L小输入前后向、下载错误/续传测试、恢复一致性、提交校验。服务器显存、速度、预训练加载和完整成绩未测就明确说明。未经授权不启动付费完整训练。提供screen/SSH命令和源码ZIP，不依赖PyCharm同步。核查竞赛规则，缺失完整提交通知时标注未确认，不把源码ZIP当作预测包。
