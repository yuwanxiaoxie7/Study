# DINOv3 ViT-L/16 + UPerNet式轻量解码 / RTX 5090

本工程根据用户后续要求改用UPerNet式PPM+FPN解码，保留卷积细节金字塔；不是DPT或原版UPerNet的完整复现。单一模型实现，SAT/Web通过配置切换。此前报告的官方成绩未与实验产物核实，不在这里归属任何目录。

默认：SAT-493M、后4块Q/V LoRA rank8/alpha16、width128、448裁剪、batch2×累积4、30轮、每2轮双验证、最后一轮强制验证。解码学习率2e-4，LoRA 2e-5；5% warmup+cosine至1%；CE+0.3Dice+0.1Lovasz+0.05Boundary。所有显存、速度和正式分数仍须服务器实测。

## 1. 上传与解压

源码包上传至`/root/autodl-tmp/Files`，新目录不存在时执行：

```bash
cd /root/autodl-tmp/Files
unzip uav_segmentation_dinov3_5090_cloud.zip
cd uav_segmentation_dinov3_5090
```

已有同名目录时先核对，不要直接覆盖运行中的代码。无需配置PyCharm自动同步。

## 2. 环境准备（无卡模式可执行）

```bash
PYTHON=/root/miniconda3/bin/python3 bash setup.sh
```

安装目标PyTorch 2.7.1/cu128；支持Python 3.10–3.12。官方仓库完整开发环境与本项目仅主干运行环境不同；实际核验算子后才能训练。

已安装兼容环境时，可显式复用而不重装大包：

```bash
export DINOV3_ENV=/root/autodl-tmp/Files/uav_segmentation_dino_5090_v2/.venv
PYTHON=/root/miniconda3/bin/python3 bash setup.sh
```

先确认该路径存在且没有别的任务在修改依赖。`PIP_INDEX_URL`可配置通用包源，`TORCH_INDEX_URL`独立配置CUDA轮子源；不要把不提供cu128的普通PyPI镜像当作PyTorch CUDA仓库。脚本不自动升级已匹配的torch，不自动删除缓存或结果。安装可能需数GB空间，建议先检查`df -h /root/autodl-tmp`。

## 3. 官方权重（无卡模式可准备）

访问 https://github.com/facebookresearch/dinov3 的Pretrained models申请官方访问权限。
SAT：ViT-L/16 SAT-493M；Web：ViT-L/16 LVD-1689M。必须是原始PyTorch骨干state_dict，不能直接把Hugging Face的转换格式当作本工程权重。

两种选择：

本地已下载的官方SAT权重：
```bash
export DINOV3_WEIGHTS='/root/autodl-tmp/实际路径/官方SAT权重.pth'
bash run_all.sh --prepare-only
```

或从官方授权链接下载，终端输入时隐藏链接，不写进命令历史：
```bash
read -r -s -p '粘贴官方授权权重URL后回车: ' DINOV3_WEIGHT_URL
export DINOV3_WEIGHT_URL
bash run_all.sh --prepare-only
unset DINOV3_WEIGHT_URL
```

脚本显示进度、速度/ETA，支持Range续传；403/404时保留半成品，提示检查权限，不记录授权URL。验证结构及所有参数形状、有限值后再改为正式文件名，重复运行复用权重。如果用户有可信的完整SHA256，可设置`DINOV3_SHA256`额外核验。默认SHA256是本地完整性记录，不冒充官方摘要。

使用Web权重时，每条命令保留`--config configs/web.json`。脚本分别使用官方SAT/Web归一化及结构，不能交叉加载。SAT不能被预先认定在低空图像上更优。

## 4. 切换有卡模式，测速但不训练

```bash
bash run_all.sh --data-root '/root/autodl-tmp/Files/2026-低空图像语义分割赛道-训练集' --check-only
```

可选对照测速，独立进程分别测试448默认、512、batch1×累积8、关闭checkpoint，绝不自动挑选配置或训练：

```bash
bash run_all.sh --data-root '/root/autodl-tmp/Files/2026-低空图像语义分割赛道-训练集' --compare-profiles
```

结果在`runs/profiles_*/comparison.json`，包含显存、吞吐、保存耗时、预计训练总时长；失败的配置明确记录。比较也会使用GPU时间。选择512配置后用`--config configs/crop512.json --check-only --new-run`重新创建实验；选择batch1同理使用`configs/batch1.json`。默认不会悄悄降分辨率、batch或通道。

## 5. 用户确认后正式训练

```bash
screen -S dinov3
cd /root/autodl-tmp/Files/uav_segmentation_dinov3_5090
bash run_all.sh --data-root '/root/autodl-tmp/Files/2026-低空图像语义分割赛道-训练集'
```

如果使用自定义环境/本地权重路径，进入新终端后重新export相应变量。训练使用与检查时完全相同的`--config`，去掉`--check-only`和`--new-run`。

Ctrl+A然后D离开；`screen -r dinov3`重新进入。断开SSH不影响screen中的进程，服务器关机仍会停止。若screen不存在，先安装或使用已有tmux。

预算默认18小时。超过时先停止，用户接受费用后追加`--skip-budget-limit`，不要加`--new-run`。未完成GPU测速前没有固定时长承诺。

## 6. 恢复与交付

相同训练命令自动恢复`runs/active.json`对应实验。每100次更新及每轮保存`last.pt`，同时保留优化器、调度位置、随机状态等；最佳域验证模型保存为`best.pt`。中断验证后重做该轮验证，不重复该轮训练。

训练关键代码/配置、数据划分、权重SHA变化会要求新实验；只改下载实现不影响训练身份。运维代码版本单独记入`operations.jsonl`。改变模型结构不能恢复旧DINOv2断点。

完成后查看：
```bash
cat latest_submission.json
```

下载所指向的`runs/实验/submission/submission.zip`及`validation.json`、`submission.sha256`。源码`*_cloud.zip`不是提交结果。备份best.pt、last.pt、config.json、environment.json、source和验证日志。

提交程序遵循当前代码和本地Label.txt所支持的协议：500张1024×1024、8bit灰度PNG、原文件名、ZIP根目录、输出ID1–8，0为训练忽略标签。完整竞赛提交通知未获得，外部预训练许可、上传方式等必须另行核对；不能宣称已验证所有规则。

## 7. 本地验证

```bash
python scripts/local_checks.py
python -m unittest discover -s tests -v
python scripts/smoke_model.py
python scripts/package_cloud.py
```

smoke_model使用真实ViT-L结构、随机初始化、小输入CPU前后向；不需要权重，不代表预训练模型、实际GPU或完整训练已验证。详细记录见LOCAL_CHECKS.md。
