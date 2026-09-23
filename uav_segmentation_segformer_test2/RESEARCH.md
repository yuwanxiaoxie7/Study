# 论文依据与实现边界

本工程采用B4基线→双偏移→分辨率/损失独立消融→B5比较的顺序。下列文献来自前期核对的一手来源；2024–2025论文作为较新依据，2021–2022文献用于机制交叉参考。其他数据集的提升不能作为本赛题预计涨分。

| 论文与来源 | 与赛题的关联和实现状态 |
|---|---|
| [SegFormer](https://arxiv.org/abs/2105.15203)，NeurIPS2021；[作者代码](https://github.com/NVlabs/SegFormer) | 真实MiT-B4/B5骨干的直接依据。已实现公开预训练骨干严格加载；GN解码头为工程适配，不是论文训练细节完整复现。 |
| [OffSeg：Learning Offsets for Better Spatial and Class Feature Alignment](https://arxiv.org/abs/2508.08811)，ICCV2025；[作者代码](https://github.com/HVision-NKU/OffSeg) | 双偏移处理类别表示与图像特征不匹配，作者有SegFormer迁移实验。已移植核心方程并测试；通用分割证据不能证明本赛题必涨分。 |
| [Frequency-aware Feature Fusion for Dense Image Prediction](https://arxiv.org/abs/2408.12879)，TPAMI接收2024；[作者代码](https://github.com/Linwei-Chen/FreqFusion) | 类内一致性与边界、频率感知融合，支持改善融合质量这一研究方向。尚未移植其滤波及重采样算子，普通FPN不是FreqFusion。 |
| [无人机矿区图像的改进SegFormer研究](https://doi.org/10.3390/s25123827)，Sensors2025 | 无人机场景、多尺度融合和局部注意力相关。已提供FPN、普通PPM和轴向注意力消融；完整SFA-PPM细节未核实，不能称原版MSFE复现。 |
| [PPTFormer](https://www.ijcai.org/proceedings/2024/99)，IJCAI2024 | 无人机视角变化与泛化相关。未实现其专门视角建模；普通旋转、尺度增强不等于复现PPTFormer。 |
| [SegFormer无人机图像研究](https://arxiv.org/abs/2410.01092)，2024预印本；[2025出版会议论文集](https://doi.org/10.1007/978-3-031-82156-1_9) | 提供B0/B3/B5在无人机任务上的应用参考，不推出B5必胜B4；不采用其中的模型集成路线。 |
| [PointFlow](https://openaccess.thecvf.com/content/CVPR2021/html/Li_PointFlow_Flowing_Semantics_Through_Points_for_Aerial_Image_Segmentation_CVPR_2021_paper.html)，CVPR2021 | 航空图像的细节、边界及多尺度语义传递。用于交叉支持保留浅层信息，未实现点流模块。 |
| [UNetFormer](https://eprints.lancs.ac.uk/id/eprint/173313/)，ISPRS Journal2022 | 遥感场景中全局上下文与局部细节联合建模。支持上下文消融动机，不作为B4/B5直接涨分证据。 |
| [HRDA](https://github.com/lhoyer/HRDA)，ECCV2022 | 高分辨率、上下文与域差异。仅作为裁剪分辨率的机制参考，不实现目标域自训练，不使用复赛图像适应。 |

## 比较方法

先比较b4_baseline与b4_offset，保持其他参数一致。额外pool2应成对比较，避免分辨率差异混入模块结论。FPN→普通PPM→轴向注意力是另一条逐项路线，不要同时堆叠所有模块。Dice、Lovasz、稀少类采样、类别加权分别测试。

640/768配置同时更改训练裁剪和预测窗口，属于成套资源方案，不是纯训练裁剪单因素实验。B5在最后进行，先用服务器实测判断时间和显存成本。

固定相同训练划分和domain_val选模口径，同时报告普通val、8类IoU/Precision/Recall、混淆矩阵、车辆/道路漏检、植被与农田混淆。若提升很小，应在预算允许时重复随机种子，不能凭单次小差异认定有效。

## 赛题与实现限制

[官方赛题](https://www.aicomp.cn/tracks/tracks-1/3708.html)要求公开学术预训练、官方数据、单模型和复赛test_2预测。本工程只对官方训练划分反向传播；测试图用于完整性检查、推理测速和最终预测，不参与伪标签、选模或拟合归一化统计。最终提交材料及节点以[复赛通知](https://www.aicomp.cn/notice/notice-1/5278.html)和平台为准。

OffSeg移植来源、固定提交、作者附加许可证条款，以及GN/dropout/pool适配差异详见THIRD_PARTY.md。msfe只是本工程配置名称，指FPN+普通PPM+ELA启发的轴向注意力，不声称原论文完整复现。

domain_val来自既有聚类留出，不保证独立航线；精确重复和感知哈希碰撞记录在审计报告，感知碰撞只作复查候选。当前没有新的完整训练成绩，不能把历史67.40算作SegFormer基线。
