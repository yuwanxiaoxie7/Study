# SegFormer B4/B5 复赛运行手册

独立工程，从公开 ImageNet MiT 权重开始，不需要旧 best.pt。建议先跑 B4 基线，再跑 B4 offset 对照，最后考虑 B5。论文与实现差异见 RESEARCH.md、THIRD_PARTY.md。本地检查记录见 LOCAL_VERIFICATION.json；5090 显存、速度、完整训练及比赛成绩均未实测。

## 上传与数据

上传 dist/uav_segmentation_segformer_test2_cloud.zip 到自己的 Linux 5090 服务器，解压并进入项目目录。代码包不含数据、权重、本地虚拟环境或训练结果。数据单独放置：

```text
比赛数据目录/
  Label.txt
  test_2.zip
  train/train/images/*.png
  train/train/masks/*.png
```

text2 对应实际文件 test_2.zip，1300 张 RGB 1024×1024 图像，不会回退到 test_1。训练集6996对，固定 train/val/domain_val 为5178/1049/769。原始标签0忽略，1背景、2建筑、3道路、4水体、5荒地、6植被、7农田、8车辆。内部有效类别0–7、忽略255；输出映射回1–8。

## SSH 命令

推荐 Python 3.11（支持3.10–3.12）。脚本建立独立 .venv，复用兼容依赖，否则安装 PyTorch 2.7.1/cu128 和固定依赖；需要兼容驱动。替换下面的数据路径。

先检查，包含有限 GPU 测速，不启动完整训练：

```bash
cd /workspace/uav_segmentation_segformer_test2
bash run_all.sh --config configs/b4_baseline.json --data-root '/workspace/比赛数据目录' --check-only
```

检查模式会校验全量数据和权重，并做2次预热、3次有效优化更新测速，结束恢复模型和随机状态，不保存训练 best/last。输出在 checks/<时间>/。追加 --device cpu 则只做有限前向，GPU预算显示未通过代表未测，不是数据错误。

决定正式训练后，在已安装 tmux 的服务器运行：

```bash
tmux new -s segformer-b4
cd /workspace/uav_segmentation_segformer_test2
bash run_all.sh --config configs/b4_baseline.json --data-root '/workspace/比赛数据目录' --run runs/b4_baseline --stage all
```

按 Ctrl+B 再按 D 离开，SSH断线后继续训练；返回用 tmux attach -t segformer-b4。实例关机或回收时 tmux 不会继续运行。

进程真正退出后，从已有断点恢复：

```bash
bash run_all.sh --config configs/b4_baseline.json --data-root '/workspace/比赛数据目录' --run runs/b4_baseline --stage all --resume
```

已有断点却未加 --resume 会报错；指定恢复但没有 last.pt 也报错。不要删除 runs。首次断点产生前退出，则按原新训练命令重新运行。

仅训练有限2次更新以检查服务器训练链路：

```bash
bash run_all.sh --config configs/b4_baseline.json --data-root '/workspace/比赛数据目录' --run runs/b4_baseline --stage train --stop-after-updates 2
```

之后去掉更新限制并加 --resume 可继续原30轮计划。正式 all 阶段顺序为数据/权重检查、GPU测速、训练与定期验证、加载best、验证、test_2预测、打包；环境安装由外层脚本完成。

单独验证或预测：

```bash
bash run_all.sh --config configs/b4_baseline.json --data-root '/workspace/比赛数据目录' --run runs/b4_baseline --stage validate
bash run_all.sh --config configs/b4_baseline.json --data-root '/workspace/比赛数据目录' --run runs/b4_baseline --stage predict
```

## 实验与资源

| 配置 | 用途 |
|---|---|
| b4_baseline.json / b4_offset.json | B4基线及双偏移对照 |
| b5_baseline.json / b5_offset.json | B5基线及双偏移对照 |
| b4_baseline_pool2.json / b4_offset_pool2.json | 配对检查额外降采样的影响 |
| b4_fpn.json / b4_ppm.json / b4_msfe.json | FPN、普通PPM、轴向注意力逐项消融 |
| b4_offset_dice.json / b4_offset_lovasz.json | 分别测试损失项 |
| b4_offset_sampling.json / b4_offset_weighted.json | 分别测试采样和类别加权 |
| b4_offset_640.json / b4_offset_768.json | 更大训练裁剪和预测窗口 |

每个配置使用不同 --run。例如 offset 使用 --config configs/b4_offset.json --run runs/b4_offset。不要一次运行所有实验。

默认512裁剪，microbatch1、累积8次，GN，全骨干微调，骨干/头学习率6e-5/6e-4，AdamW、warmup+poly、激活检查点，自动BF16/FP16。训练前实测更新、预测、显存和写盘成本，外推总耗时加20%余量；默认预算18小时、reserved显存28GiB。超预算则停止，需显式建立更轻配置或调整自己接受的预算，不静默更改训练参数。估计不包括全部安装、下载、审计、压缩及服务器竞争开销，不保证用时。

## 日志、选模、恢复

runs/<实验>/ 保存 terminal.log、resolved_config.json、identity.json、environment.json、audit.json、benchmark.json、weight_loading.json、last.pt、best.pt、best_metrics.json、metrics/、completed.json。

每100个完整更新及每轮保存last，最多损失未写盘的更新。断点保存模型、优化器、scaler、学习率进度、epoch、样本游标和随机状态。恢复必须保持训练代码、配置、数据清单、精度及核心运行版本一致；跨硬件不保证位级一致。只加载可信的本工程断点。

沿用原项目口径，按domain_val mIoU选best，同时报告val和8类指标。domain_val是已有聚类留出，不能保证航线级独立。每2轮和最后一轮验证；指标0–1尺度。可视化依次为原图/真值/预测/错误图；道路FN仅代表漏检像素，不是连通性指标。

## 提交包

输出 runs/<实验>/submission_test2/submission_test2.zip：1300张平铺同名、1024×1024、8位灰度PNG，像素1–8。逐图及ZIP校验，保存SHA256和模型/输入身份收据。同一身份下预测可续作；损坏图重算，身份变化禁止混合。仅一个best模型，滑窗空间融合，无多模型集成、无测试集训练，不自动提交平台。

代码上传包不是比赛预测包。Docker与技术方案PDF等正式材料应结合最终训练和消融结果按赛事平台要求制作，当前不宣称已有这些结果。

## 本地交付状态

已完成的数据检查、7项合成CPU测试和B4/B5真实权重64×64 CPU前后向结果收录在LOCAL_VERIFICATION.json。没有运行5090测速或完整训练。67.40属于用户确认的旧DINOv2项目，新SegFormer工程暂无官方成绩；不引用错误67.52标记。

重新生成代码上传包（不启动模型）：

```bash
.venv/bin/python scripts/package_cloud.py
```
