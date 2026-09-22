# DINOv2-L / RTX 5090

基线：用户报告 DINO 4090 工程官方分数 **67.40**。本工程尚无官方成绩。
单一实现：DINOv2-L + 最后4层QKV LoRA + 逐级空间先验解码器。
本轮增加训练专用边界辅助监督和Lovasz损失，默认系数0.1/0.2；推理不执行边界辅助头。
10篇论文及对照方案见 LITERATURE_10.md，可复用执行提示词见 OPTIMIZATION_PROMPT.md。
原4090工程保留；本文件夹不包含旧版解码器、旧训练结果或虚拟环境。

## 服务器使用

选择 Linux x86_64、RTX 5090 约32GB、Python 3.10–3.12，驱动必须支持所选GPU与CUDA运行时。
上传并解压 `uav_segmentation_dino_5090_cloud.zip` 到 `/root/autodl-tmp/Files`。
不需要配置PyCharm远程解释器或自动上传；在SSH终端执行。

只安装环境：
```bash
cd /root/autodl-tmp/Files/uav_segmentation_dino_5090
PYTHON=/root/miniconda3/bin/python3 bash setup.sh
```

检查、下载权重及GPU测速（不进行完整训练）：
```bash
cd /root/autodl-tmp/Files/uav_segmentation_dino_5090
PYTHON=/root/miniconda3/bin/python3 bash run_all.sh --data-root '/root/autodl-tmp/Files/2026-低空图像语义分割赛道-训练集' --check-only
```

检查通过后开始训练：
```bash
screen -S dino5090
cd /root/autodl-tmp/Files/uav_segmentation_dino_5090
PYTHON=/root/miniconda3/bin/python3 bash run_all.sh --data-root '/root/autodl-tmp/Files/2026-低空图像语义分割赛道-训练集'
```
Ctrl+A再按D离开会话；`screen -r dino5090`重新进入。服务器关机仍会停止训练。
意外退出后运行同一命令自动续训；不要加 `--new-run`。修改配置后必须使用 `--new-run`。
若没有screen，使用服务器镜像提供的终端保持工具，或先安装screen。

默认30轮、每2轮完整验证一次，crop448、width96、batch1×累积8、tile448/stride224；保持原划分、增强、LoRA与学习率，CE+Dice基础上增加两项辅助目标。
显存28GiB上限、估计18小时上限；实际速度和显存只能在服务器测速后确认。
不自动调小模型掩盖OOM。若超限可使用 `--config configs/decoder_only.json --new-run --check-only` 检查392/64配置。

权重可以从原4090工程复制整个 `pretrained` 文件夹到本工程，含 `.pth` 与下载校验 `.json`；脚本核对哈希后复用。
不要复制旧 `.venv` 或 `runs`。新解码器不能直接恢复旧4090训练断点。
安装不保留pip大轮子缓存，避免50GB数据盘同时保存已安装的CUDA包和下载副本；如中断后显示空间不足，只可清理本工程`.cache/pip`和`.cache/tmp`，不要删除`.venv`、`pretrained`或`runs`。

## 文件保留与提交

训练完成显示 `runs/<实验>/submission/submission.zip`，这是预测结果包。
云端源码包 `_cloud.zip` 不能当预测提交包。
下载 `submission.zip` 和旁边的 `validation.json`、`submission.sha256`。
Windows PowerShell用 `Get-FileHash '实际下载路径\submission.zip' -Algorithm SHA256` 与报告比对，确认传输一致。
另外备份整个实验目录（尤其best.pt、last.pt、config.json、source、验证日志）及当前项目代码。
ZIP生成前校验PNG名称、数量、灰度、尺寸、像素范围、CRC，并输出字节数与SHA256。
沿用4090的500张1024×1024单通道PNG、根目录同名、预测ID1–8协议。
本地Label.txt已核对；尚未获得完整官方提交通知，上传方式须以竞赛页面要求为准。
平台“下载文件不是有效ZIP”也可能是下载链接返回网页或传输不完整，不能仅据此认定PNG格式不符。

研究与比较计划见 PAPER_RESEARCH.md；验证范围见 LOCAL_CHECKS.md。

新增消融配置：pyramid_control（两项关闭）、boundary_only、lovasz_only和默认train（两项开启）。
例如检查控制组：`bash run_all.sh --data-root /实际数据目录 --config configs/pyramid_control.json --new-run --check-only`。
检查后训练同一实验保留--config参数，去掉--check-only和--new-run。不同配置切换必须新建实验。
新增代码不能续接修改前的5090运行；开始本轮时如已有runs记录，使用--new-run检查一次。
