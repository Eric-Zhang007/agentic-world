# 训练输入与监督

统一清单每行是一条 JSON，含 sample_id、sample_root、task、split、episode_id、balance_group。dataset 固定为 data，sample_root 相对于解压目录。source_release 仅保留来源追溯，不用于配置多个数据根目录。

## 训练信息流

```text
历史 RGB + 对应白模参考帧 ─┐
目标时间段白模视频 ───────┼→ 待训练模型 → 预测 RGB 视频
参考提示词 / 可选物体关系 ─┘                 ↓
                                与目标 RGB supervision 比较
```

目标 RGB 只能作为监督，不放入条件。QA 报告、未来生命周期信息和目标 RGB 描述也不作为条件。每条背景参考时间早于目标时间；时钟、相机和状态可从样本 JSON 追溯。目标片段 120 帧、24 fps。历史 RGB 与白模帧必须成对使用，不应将不同时刻的两张图片拼成一对。

## load_dataset 返回值

- conditions.prompt：样本提示词。
- conditions.reference_pairs：历史参考列表，每项有 reference_id、rgb、whitebox；图像为 HWC uint8 RGB。
- conditions.target_whitebox：THWC uint8 RGB 白模视频。
- conditions.object_references：物体任务的对应关系；背景任务为空列表。
- supervision.target_rgb：THWC uint8 RGB 目标视频。
- supervision.valid_pixels：HW float32，0 到 1，适用于监督损失；不代表精确可见性。
- temporal_indices：本次选择的视频帧索引。
- source_fps：原帧率。

object_references 保留 object_id、category、reference_id、reference_bbox_xyxy、observed_at_ns、used_at_ns、used_until_ns、reference_mask 和所选时刻的 target_projected_bboxes。投影框不等于无偏的可见性真值，SAM 掩码也不是人工精标。reference_pairs 可能有多项，帧引用根据 reference_id 对应。

每个 samples/<sample_id>/sample.json 是该样本的原始文件索引。通过 conditions 和 supervision 中的相对路径访问文件，不要假定两类任务的全部文件名一致。相机、逐帧世界状态及审核辅助材料在各样本目录，训练代码应显式选择所需字段。

## 模型适配时需要决定

本包不指定 H3 的训练结构。需要由训练实现决定参考帧打包、白模视频编码、文本模板、物体关系如何编码，以及只训练 LoRA 还是其他模块。不要因为清单内有参考关系就假设模型已经有专用记忆分支。无参考、错误参考、正确参考的对照应在实际训练验收中验证。
