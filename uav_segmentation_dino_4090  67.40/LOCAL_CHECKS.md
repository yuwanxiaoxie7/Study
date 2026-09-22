# 本地验证记录

环境：Windows、D:/Anaconda3/envs/pytorch/python.exe、PyTorch2.5.1。所有模型测试在CPU运行，未使用本机GPU长训。

- 9项测试通过：两种解码器尺寸/梯度；LoRA零初始化与冻结；官方ViT含0/4 register token特征与原接口一致；checkpoint反向；Ignore全忽略零梯度；固定划分/三个配置；Hann完整覆盖；ZIP拒绝错误尺寸/类型/标签；优化器学习率分组；训练控制器模拟验证中断恢复及最佳checkpoint选择（部分项目合并在单项测试）。
- 真实完整L结构CPU构建、28×42随机输入前向和反向已通过：总参数306,918,216，可训练参数2,549,576，末层LoRA梯度有限。此项未加载L预训练，不是训练精度验证，也不代表392crop的CUDA显存已验证。
- 项目Python源码AST解析与三份配置解析通过。
- 原始split.json、domain_split.json、data_manifest.json与DINO源工程逐字节SHA256一致；真实数据采样在恢复前后像素一致，稀有类别池保持完整训练集。
- 模拟训练中途（非仅验证边界）中断，恢复后的最终模型与不间断运行逐参数完全一致。
- run_all.sh和setup.sh经Git Bash `bash -n`语法检查。
- Windows打包脚本调用的是标准库Python打包器；包内容按白名单过滤，CRC检查，排除数据、权重、缓存、环境、runs。

待云端：公开L权重真实下载与严格加载、CUDA/BF16算子兼容、392crop真实峰值显存与速度、完整训练/验证、官方分数。脚本自动执行前置检查，失败会停止。
