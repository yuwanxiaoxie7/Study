# 4090复赛：重新训练并生成测试集2提交ZIP

适用于初赛best.pt已删除的情况。本工程从4090初赛工程独立复制，保留DINOv2-L/14、LoRA、SegFormer风格解码器、固定训练/验证划分以及原30轮训练参数。训练从公开DINOv2骨干预训练开始，自动用域验证选择best.pt，再预测测试集2的1300张图。重新训练不能保证复现官网67.40分。

## 一键运行

将本工程代码包上传Linux x86_64、Python 3.10、RTX 4090 24GB服务器并解压。数据单独上传，解压train.zip后保持目录如下（不要重复套一层train）：

```text
比赛数据目录/
  Label.txt
  test_2.zip
  train/train/images/*.png
  train/train/masks/*.png
```

在代码目录执行：

```bash
bash run_test2.sh --data-root "/实际路径/比赛数据目录"
```

脚本自动安装隔离环境、运行检查、核对数据、下载公开骨干、GPU测速、重新训练30轮、选择最佳权重、预测1300张并校验ZIP。不需要旧best.pt或test_1.zip。完整训练耗时取决于服务器；测速会打印估算，超过原18小时预算会停止。接受该估算后可追加 `--skip-budget-limit` 再运行。

只检查环境与估算耗时：在命令末尾加 `--check-only`，正式运行时去掉。中断后重跑同一命令即可恢复训练；不要删除runs或追加--new-run。若在预测阶段中断，重跑会复用已完成的训练并重新生成预测包。

## 提交结果

完成后控制台及latest_submission.json给出结果路径：

```text
runs/<实验名称>/submission/submission.zip
```

只上传这个ZIP：1300张平铺、与测试图片同名、1024×1024、8位灰度PNG，像素类别1–8。代码上传包不是比赛结果。代码包不包含训练集或测试集，请单独上传。

务必保存整个runs目录，特别是best.pt、last.pt、配置、环境和验证记录，后续制作Docker与技术方案需要使用。此次不制作Docker及PDF。其他继承的研究/检查文档记录的是初赛背景，复赛运行以本说明为准。
