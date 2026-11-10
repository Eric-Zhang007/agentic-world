# GTA 数据恢复与重建

## 已核实的范围

2026-10-26 本地 30 个不连续编号的 non_jsonl 分片已全部解包。
这不代表完整源包恢复：这些分片没有包含逐帧 JSONL 和原始 spool，且 RGB 也不齐。
先前把历史审计中的 610GB 等同于本地解包内容的说法不成立。

实际盘点 OSS 得到 108,510 个对象、833,513,752,209 字节。这包含可再生缓存、
历史证据和 proxy，不等于需要下载的字节，更不等于训练数据量。
补齐程序排除 derived 中的 scene 缓存、预览、日志和 proxy；初次计划需恢复
465,266,063,170 字节，其中旧目录可复用 199,857,996,982 字节，
需从 OSS 下载 265,408,066,188 字节。续跑清单会扣除已完成项。

## 正在执行

- 本地工作目录：`E:/GTA_555_complete_rebuild_20261025`。
- 原生解包：`scripts/extract_gta555_native.py`，30/30 完成。
- 远端盘点：`scripts/inventory_gta555_oss.py`，完整清单在 `recovery/oss_objects.jsonl`。
- 缺项补齐：`scripts/recover_gta555_source.py`，复用旧文件或下载到临时文件后改名，已有冲突不覆盖。
- 文件就绪审计：`scripts/audit_gta555_readiness.py`。仅判断文件存在性，不代表视觉验收。
- 服务器：`/mnt/hdd/gta_source_rebuild`，独立 `render-env`，不修改 H3 环境。
- 真实渲染验证：`scripts/bootstrap_gta555_render.py` 下载选定片段和静态场景，
  校验原始 spool、场景，再接 `scripts/run_gta555_full_rebuild.py` 调原包 `tools/reproduce.py`。

`run_gta555_full_rebuild.py` 已替换旧占位实现：读取真正的嵌套 YAML，执行渲染，
逐帧解码核对视频数量、元数据数量和 usable frames，保存可恢复的逐任务结果。
它目前负责重建与解码检查，不会伪称已经执行后续训练数据筛选、视觉审核和打包。

## 后续工作

最新实测：ep1 / agent_000 / segment_00000 的 1,200 帧重建耗时约 159 秒，
完整解码后与 OSS 原版逐帧像素一致（MSE=0），原版发布哈希匹配。
OSS 实际保留全部 1,992 个 proxy，共 28,413,708,619 字节。
因此改为先用 `cache_gta555_proxies.py` 校验原版哈希并完整解码，只有异常片段才重建，
不再无条件重渲染全部视频。服务器结果位于 `original_proxies/PROXY_STATUS.json`。
这个结论仅证明视频复现正确，不代表 RGB 与白模的几何误差已经通过审核。

全部 1,992 个原版 proxy 已通过发布哈希核对及逐帧解码：2,782,278 帧、零失败。
本地 `sync_gta555_proxies.py` 会复用旧目录中内容一致的视频，只传回本地缺项。
`adapt_gta555_record_indices.py` 会校验原始 spool 内容并将已有字节索引适配到现有 Records API。
`qualify_gta555_sources.py --watch` 持续消费已就绪输入，完成 RGB/元数据核对、
实际相机/world 查询、配对视频完整解码及轨迹采样。帧缓存和工作索引位于
`D:/GTA_source_work_cache`；原始数据和最终样本仍在 E 盘。

同一真实片段的预处理从 76.4 秒降到 20.3 秒。两次的相机、时间、投影框数值一致，
投影框集合的排列次序不具有语义；后续遍历已固定顺序。

`continue_gta555_training.py` 等完整源预处理结束后自动执行 survey、背景匹配与候选样本导出。
它不自动冒充视觉验收，不调用付费模型，也不覆盖已交付 ZIP。
已用 19 段、227 个轨迹点实际验证该链路，得到 45 个背景参考候选，
并导出邻近历史和约 120 秒后返回同位置的两个样本。证据在
`E:/GTA_555_complete_rebuild_20261025/pipeline_exercise`。

1. 核验补齐内容与原包索引对应，保留缺失、冲突、排除记录。
2. 完成真实短片段和完整片段重建测试，实测 CPU 耗时，再扩大批处理。
3. 沿用原始时间戳、clock domain 与缺失区间，对齐 RGB、proxy、相机、动作和状态。
4. 接回场景重访采样与现有 object 样本，明确背景、人物、车辆、混合场景配比。
5. 验证参考帧早于目标，划分无泄漏，视觉审核通过后合并交付。

源包路由 `data/episodes.json` 在本地分片中缺失。服务器测试依据原包
`method/original_sources.json` 和 `data/index.json` 重建路由，并另存来源说明。
这不能补充或冒充缺失的原始 `audit/files.jsonl`。

旧训练 ZIP 和验收 ZIP 保持原样。没有完成新一轮样本验收前，不替换旧交付物。

## 查看实际进展

本地：`EXTRACTION_STATUS.json`、`recovery/RECOVERY_STATUS.json`、
`recovery/recovery_results.jsonl`、`recovery/recovery_conflicts.json`。

服务器：`BOOTSTRAP_STATUS.json`、`bootstrap.log`、`rendered/PIPELINE_STATUS.json`。
比较更新时间和完成字节，不能仅凭进程存在宣称推进。

耗时估算须在真实吞吐和单段渲染测量后更新；目前不承诺整个训练集的完成日期。
