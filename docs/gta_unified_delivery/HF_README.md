---
language:
- en
- zh
tags:
- video-generation
- world-model
- reference-memory
pretty_name: GTA RGB and Whitebox Reference Memory
---

# GTA 参考记忆训练数据

完整包：`GTA_memory_training_unified.zip`。解压后是一个统一数据根目录，无需旧包。

验收小包单独线下发送，不在本仓库上传。

完整池包含 6,971 条任务记录：1,987 背景、3,539 人物、1,191 车辆、254 混合物体，对应 6,062 个不同目标视频窗口。训练/验证/测试为 6,556/197/218 条，按 episode 隔离。

输入是历史 RGB/白模参考帧对、后续白模视频、提示词及可选物体关系；监督为后续对齐 RGB 视频。每段 120 帧。目标 RGB 不应作为输入条件。

包内提供中文说明、字段文档、统一 dataloader、完整划分清单、均衡采样清单和审核记录。解压后运行 `pip install -r requirements.txt`、`python example.py`。

数据完成解码检查、配对和抽帧审核，但未经过模型训练效果验证。白模是粗几何，物体关系不是穷尽标注，SAM 掩码不是人工真值，附近历史不是长期重访。Episode 隔离不保证 GTA 地图位置隔离。验收包全部来自训练划分，不是独立评测集。

本仓库不包含模型权重，不预设 LoRA 或其他训练结构。原始录制素材的许可未在这次数据整理中重新确定，因此不额外声明开放许可证。
