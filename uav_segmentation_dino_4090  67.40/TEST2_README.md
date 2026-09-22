# 复赛测试集2：仅预测

本入口不启动训练。使用初赛保存的完整 best.pt，沿用 checkpoint 内的模型配置、tile/stride、精度策略、无TTA和类别映射。测试集2包含1300张1024×1024图片。

在原RTX 4090服务器上，将本工程、原 best.pt 和官方 test_2.zip 放好后运行：

```bash
bash run_test2.sh --test-zip "/数据目录/test_2.zip" --checkpoint "/原实验目录/best.pt"
```

默认输出 `submission_test2/submission.zip`，只提交此ZIP。内含1300张平铺8位灰度PNG，与输入同名、尺寸1024×1024，像素类别ID为1–8。旁边的 `submission.validation.json` 是校验记录，不放入提交包。输出已存在时会停止，可用 `--output /其他目录/submission.zip` 指定新位置。

只检查测试集（无需安装深度学习依赖）：

```bash
python predict_test2.py --test-zip "/数据目录/test_2.zip" --check-only
```

权重必须匹配本工程保存的初赛 `submission/validation.json` 中的SHA256：`b641d94455757fe9296cc8e748076ded12d13402a9da6e4c2c08bc55353a9333`。该记录对应本地保留的初赛提交包；仍需用户确认它就是官网67.40分的提交。推理还会校验原模型源码和数据划分记录，避免混用其他模型。

本地目前没有 best.pt，不能从初赛预测PNG还原权重。须从原云端实验取回；不要用公开DINO骨干权重代替训练好的分割模型。无需另行下载骨干、解压训练集或重训。

旧 `run_all.sh` 和训练配置保留初赛行为。此次新增独立复赛入口，不影响原实验恢复。Docker及技术方案PDF后续准备。
