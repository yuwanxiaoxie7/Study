# DINO 4090 独立工程

复赛测试集2仅预测：见 [TEST2_README.md](TEST2_README.md)，使用 `run_test2.sh` 加载初赛best.pt，输出1300张预测结果ZIP。

基于 `uav_segmentation_3050` 的 E1_sampling02 数据、损失与适配器，新建 DINOv2-L/14 + SegFormer 风格轻量解码器。原工程不修改。当前是待云端验证的候选，尚无新官网分数。

## 上传和运行

Windows 双击 `package_cloud.bat`，生成 `uav_segmentation_dino_4090_cloud.zip`。上传、解压到租赁服务器。数据独立上传，目录内应有 `train/train/images`、`train/train/masks`、`Label.txt`、`test_1.zip`。

服务器选 Linux x86_64、Python 3.10、RTX 4090 24GB，数据以外预留至少 25GB 磁盘和 16GB 系统内存。在新工程目录执行：

```bash
bash run_all.sh --data-root "/实际路径/比赛数据目录"
```

自动：参数/路径检查 → 隔离环境安装 → 语法与单元测试 → 数据哈希检查 → 官方骨干权重下载 → 真实CUDA前后向/滑窗测速 → 30轮训练与双验证 → 最佳checkpoint预测500张 → ZIP校验。

只检查云端兼容性和速度：追加 `--check-only`。正式训练前的估算包含双验证与500张测试预测，再乘1.2余量；超过18小时停止。明确接受更长租赁时间时追加 `--skip-budget-limit`。18小时不是保证时长或自动关机时间，首次下载/安装另计。

## 中断恢复

重跑同一命令即可；`last.pt` 保存模型、优化器、AMP与随机状态以及读取位置。修改配置/源码须加 `--new-run`。每个项目只允许一个训练进程。OOM不会偷偷缩小模型，明确修改crop为14的倍数（如336）或width（如32）后再新建实验。

## 配置与对照

默认 L + SegFormer风格融合：`configs/train.json`。

L + 原解码器对照：

```bash
bash run_all.sh --data-root "/实际路径/比赛数据目录" --config configs/large_legacy.json --new-run
```

S + 原解码器对照：把配置换成 `configs/small_legacy.json`。这些对照不会自动排队启动。公平判断SegFormer收益，应比较L+legacy和L+segformer的固定域验证与官网分数；不能只和旧S的历史记录比较。

三者共用30轮从公开骨干预训练开始的配方：crop392、resize512/768/1024、batch1×累积8、head LR3e-4、LoRA LR3e-5、warmup100、CE+0.3Dice、稀有采样0.2、每轮双验证、tile448/stride224/Hann、无TTA。旧最佳是已训练模型的3轮低学习率续训，新模型不能直接套用该时长。

## 最终产物

控制台和 `latest_submission.json` 显示唯一提交包：`runs/<实验>/submission/submission.zip`。比赛只上传该ZIP，必须500张平铺的8-bit灰度1024×1024 PNG，预测类ID1–8。

另外请保存 `best.pt`、`last.pt`、`config.json`、`identity.json`、`pretrained.json`、`environment.json`、`pip-freeze.txt`、`best_metrics.json`、`history.jsonl` 和当前工程代码。best按769张域验证选；1049张随机验证辅助观察；5178张实际训练。数据划分文件从DINO项目原样复制。

网络使用DINOv2官方Apache-2.0代码和公开LVD-142M骨干，不使用外部分割标注权重。赛事网页本次未返回完整预训练/集成规则条文，不能据此宣称已得到赛事许可；参阅 `PAPER_RESEARCH.md` 的证据边界。
