# H3 Geometry Conditioning 环境评估

Update: the later ComfyUI source inspection found native Union 2.0 support and
explicit zero control rows for reference/keyframe tokens. The conclusions below
apply to the installed DiffSynth path, not all H3 implementations.
See `episode_payload_and_comfyui.md` for evidence and remaining checkpoint checks.

实查时间：服务器当地时间 **2026-10-03 01:10:38 +08:00**（UTC 2026-10-02 17:10:38）。
范围：只读文件/目录、源码静态检查、发布方模型元数据与模型卡。没有运行推理或加载模型。

## 结论

**当前 full INT8 Ref2VA 不能直接接入 depth/canny ControlNet；semantic-ID 没有现成支持。**
服务器有旧版 Union ControlNet 的实现和完整性初筛通过的 FL2VA 基座文件，但已知权重/缓存目录没有发现 ControlNet checkpoint。
最小后续路径是独立 FL2VA + 旧版 Union 分支验证，不替换当前 Ref2VA，不升级既有环境。
这条路径仍需单独授权下载约 6.81 GB 权重和运行 GPU 验证；本次未执行。

“源码是否仅 FL2VA”的准确答案：

- **官方仓库现有推理/训练示例和注册配置针对 FL2VA + v1 Union。**
- 公共 pipeline 暴露 `control_video`/`control_scale`，没有显式拒绝 Ref2VA 的模式判断，不能描述成 API 硬编码只允许 FL2VA。
- 但当前参考视频 Ref2VA 的 packed video rows 包含参考帧，而控制编码器只有目标帧 rows；ControlNet 检查二者必须相等。按当前源码直接组合会触发行数不匹配。这是静态代码结论，不是本次 GPU 实测异常。
- 即使移除参考输入避免行数问题，也不能把 Ref2VA checkpoint 自动视为已验证的 FL2VA 控制基座。

## 能力边界

| 信号 | 发布方 v1 Union | 服务器现有代码/部署 | 判断 |
| --- | --- | --- | --- |
| Canny | 支持边缘视频 | 有 FL2VA 控制示例；缺控制权重；当前 Ref2VA runner 无控制入口 | 可独立验证，不是当前已接入 |
| Depth | 支持深度图视频 | 同上；控制帧转 RGB 后经视频 VAE | 可独立验证；不是米制深度张量接口，不保证几何精确约束 |
| HED / MLSD / Pose | 支持 | 同一个 Union 分支 | 未做本机推理验证 |
| semantic-ID / instance-ID | 未列为训练条件 | 无类别词表、ID embedding 或整数标签输入路径 | 当前不支持；不能把着色 ID 图当成已支持的语义控制 |
| Layout | v1 未支持；2.0 发布方支持 | 现有 H3 注册为 5 控制块，不是 2.0 的 10 块 | 2.0 的彩色 bbox/白底 RGB 视频不是逐像素 semantic-ID |
| depth + canny 同时输入 | 未见本地多分支接口 | 单一 `control_video` 和标量 `control_scale` | Union 表示多类型共用模型，不等于已支持双路并行融合 |
| Inpaint | 发布方 VideoX-Fun 宣称支持 | 本地控制编码器仅编码控制视频并补齐通道 | 不能由模型卡推断此 DiffSynth 路径已支持完整 inpaint 配方 |

控制输入不是任意语义张量：`MiniMaxH3Unit_ControlNetEncoder` 将帧转 RGB、LANCZOS resize，
截断/末帧补齐到模型帧数，再经 VAE 和 patchify，列补到 `49 * 4 = 196`。
空帧序列没有友好校验，会访问 `frames[-1]`。后续适配层需先校验非空、尺寸、帧率、
真实帧数和取值约定。depth 的归一化/远近方向需对照训练预处理；semantic-ID 必须留在
world state/校验层，不能经过 RGB 插值后仍宣称保留精确 ID。

## 实际资产检查

扫描方式为路径名称/文件 stat 和小型 index JSON，不读取大权重内容，不做完整 SHA256。
下列已存在目录均扫描到结束，无错误、无条目截断、没有 ControlNet 候选权重：

| 根目录 | 扫描条目 | 备注 |
| --- | ---: | --- |
| `/mnt/hdd/MiniMax-H3` | 359 | FL2VA、Ref2VA 和下载缓存 |
| `/mnt/hdd/MiniMax-H3-Quant` | 6 | 四个现有量化权重 |
| `/mnt/hdd/h3-models` | 2 | `MiniMax/MiniMax-H3` 为指向 `/mnt/hdd/MiniMax-H3` 的软链接，目标已独立扫描 |
| `/mnt/hdd/agentic-world-models` | 372 | 含其他模型/工具缓存；跳过目录软链接 |
| `/home/star/.cache/huggingface` | 329 | 未发现 ControlNet 模型缓存目录或命名权重 |
| `/home/star/.cache/modelscope` | 6 | 未发现 ControlNet |
| `/home/star/minimax-h3/run` | 120 | 未发现 ControlNet |

`/mnt/hdd/DiffSynth-Studio/models` 不存在。脚本还会加入服务器进程中设置的
`HF_HOME`、`HF_HUB_CACHE`、`HUGGINGFACE_HUB_CACHE`、`MODELSCOPE_CACHE`；
本次无额外根目录。一次较广的名称搜索只见源码/库文件，`lost+found` 无权限；
首轮全盘脚本扫描达到 300000 条上限，随后改为上述定向完整扫描。
结论不覆盖任意改名 checkpoint、未声明的其他用户缓存、跳过目录或无法访问的位置。

FL2VA 与 Ref2VA 各自的文件统计如下；字节数相同不代表权重内容相同：

| 组件 | 每个变体文件数 | 每个变体文件总字节 |
| --- | ---: | ---: |
| transformer | 13 | 66,280,485,936 |
| text_encoder | 14 | 66,714,912,872 |
| video_vae | 1 | 10,415,548,320 |
| audio_vae | 1 | 605,429,308 |

两种变体的 DiT/text index 所列分片均存在且非零。只证明资产存在，不证明所有 tensor 可加载。
当前 full INT8 DiT 为 `minimax_h3_ref2va_int8_convrot.safetensors`，
34,038,894,550 bytes；文本编码器为 `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`，
27,141,342,152 bytes。另有 pruned INT8/FP8 DiT，均不是 ControlNet。

## 元数据与下载条件

优先在服务器请求 `hf-mirror.com`，只请求 API JSON 和固定 revision 的 README，
没有请求 safetensors 正文或演示视频。HF 使用发布方 `alibaba-pai` 命名空间，
代码示例中的 `PAI` 是 ModelScope 命名空间，不能原样替换 HF endpoint。

| 分支 | 文件大小 | HF revision | 本地接入状态 |
| --- | --- | --- | --- |
| v1 `MiniMax-H3-Fun-Controlnet-Union.safetensors` | 6,806,843,904 B；6.807 GB；约 6.339 GiB | `9df91437f0890b20f8a0b07c1d9500648cfcbf44` | 本地有 5 块注册/转换器，缺权重 |
| 2.0 `MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors` | 13,551,637,688 B；13.552 GB；约 12.621 GiB | `7d2c95de2e351ed6a0b360af45f62daa04c26f88` | 本地未见 2.0 专用注册/示例，不可直接替换 |

v1 LFS SHA256：`919a48acb525dc8fc70287fcd94ec1f5e5e289a77f1df14d01099c6ce204eb02`。
2.0 LFS SHA256：`ecaf457aacc18ca6ed9889a276a36722e88c09962549bcd9e153b97e35ea921e`。
这些是远端元数据声明的校验值，本次没有下载后验算。
ModelScope 发布方 API 独立返回相同 v1 大小和 SHA256，权重文件 revision
`dd2239b4d127652c6d6b86ca0f38bd157905b857`。

实测网络结果：镜像 `PAI/...` 返回 401；镜像 `alibaba-pai/...` 的两个模型 API 和
固定版本 README 均 200；服务器直连 `huggingface.co` 443 被拒；ModelScope API 200。
这说明镜像元数据可达，不等于多 GB 权重下载已验证。发布方将 v1 标记为 legacy 并推荐
2.0；本评估优先 v1 仅因它匹配当前安装代码，不能将其称为最新推荐版本。

发布方资料的请求地址（来源为作者模型仓库，不使用第三方博客）：

```text
https://hf-mirror.com/api/models/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union?blobs=true
https://hf-mirror.com/api/models/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0?blobs=true
https://hf-mirror.com/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/raw/9df91437f0890b20f8a0b07c1d9500648cfcbf44/README.md
https://hf-mirror.com/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/raw/7d2c95de2e351ed6a0b360af45f62daa04c26f88/README.md
https://modelscope.cn/api/v1/models/PAI/MiniMax-H3-Fun-Controlnet-Union/repo/files?Revision=master&Recursive=true
```

## 源码证据和阻碍

DiffSynth HEAD：`974cfa37f27ac55eba3b6d10efa21f876900572d`。
以下行号相对服务器实际文件，不能只凭 HEAD 假定工作树无本地改动：

- `diffsynth/pipelines/minimax_h3_audio_video.py:121`：公共 control 参数；`:807` 将 reference 与 target 视频位置拼接；`:851` 控制帧预处理；`:864` 仅编码目标控制帧；`:1015` 调控制分支。
- `diffsynth/models/minimax_h3_controlnet.py:34`：v1 层位 `(0,10,20,30,40)`；`:83` 要求每个 packed video row 都有控制行；`:101` 默认不直接对音频 token 加控制残差。
- `diffsynth/configs/model_configs.py:1633`：仅查到 v1 H3 ControlNet 注册，`control_in_dim=49`、5 个控制块。2.0 要 10 个块 `(0,5,...,45)`；不能依赖非严格加载忽略多余权重。
- `diffsynth/utils/state_dict_converters/minimax_h3_controlnet.py:45`：提取控制分支，并重排 QKV、SwiGLU 权重；并非任意 checkpoint 的无条件兼容加载。
- `examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Fun-Controlnet-Union.py:18`：ControlNet 因格式转换不支持该示例的 disk offload，使用 CPU BF16；`:32` 起全部引用 FL2VA。
- `diffsynth/models/minimax_h3_dit_comfy.py:60`：full Comfy DiT 继承基础 DiT；INT8 不等于完全没有结构接口，但继承关系不能证明 Ref2VA 联合控制正确。
- `/home/star/minimax-h3/deploy/h3_quant_single.py`：固定 Ref2VA 示例、仅既有量化变体，不加载 ControlNet，不提供 prompt/input-video/control-video CLI；其设备迁移逻辑对 tensor 生效，不自动证明新增 ControlNet 模块位于 DiT 卡。

## 可行路径与准入条件

1. **当前即可做的工作**：保持 full INT8 Ref2VA 配方及结果不动。外部 world state 可产出 depth/canny 和保留 semantic-ID，用于几何检查/后续适配，但不能声称已经对 H3 去噪施加控制。
2. **最小控制验证候选**：另建任务目录/入口，使用已有 FL2VA BF16 基座、本地绝对路径 ModelConfig 和 v1 Union；不运行原示例的自动下载逻辑。只缺控制分支的已知大文件，不需要重复下载整套 FL2VA。独立入口先做空输入/帧尺寸检查及严格权重匹配，再经用户批准启动短测。
3. **2.0 路径**：若需要 bbox Layout 等新条件，单独环境/工作树按作者 VideoX-Fun 配方或经审计的 10 块适配实现，不在既有 H3 就地升级。需额外 13.55 GB checkpoint、正确层映射，inpaint 还需 post-norm 配方。
4. **保留 Ref2VA 参考视频并加控制**：属于后续研发，不是开关。需定义参考/目标控制 rows 的映射、参考段 residual 屏蔽策略、时间/位置编码和跨卡模块放置；只补零满足 shape 不足以证明语义正确。还须验证训练基座匹配及量化后质量，不承诺现成兼容。

资源只是条件，不是可运行保证：本次 `/proc/meminfo` 显示 MemTotal 263,687,584 kB，
MemAvailable 256,901,380 kB，磁盘可用 10,977,418,567,680 B。
部署 README 的历史 3 秒 INT8 峰值为 DiT 36.991 GiB、辅助卡 26.249 GiB，
并非本次测量，也不能直接加上文件大小推断可放入 GPU。控制分支还会增加激活、
控制 VAE 编码、转换临时内存与传输。FL2VA BF16 路径需要独立评估 offload/设备安排；
未验证时延、峰值、画质、depth 几何精度或多模态一致性。
后续下载/使用前另行确认发布方许可与适用范围，本次不作法律判断。

## 复现与不变性

本地工作目录 `C:/Users/zjc/omgd`：

```powershell
python -B agentic_world/scripts/inspect_h3_conditions.py --server-exec C:/Users/zjc/omgd/server_exec.py --metadata
```

已实际通过上述入口运行，退出码 0。脚本将自身源码编码后经现有 SSH helper 送入
服务器 `python3 -B -c` 内存执行，不 SFTP 上传，不生成服务器脚本/报告/cache 文件。
仅标准库；不导入 torch、DiffSynth 或模型，不调用 GPU API，不安装依赖。
元数据每次响应上限 2 MiB、25 秒超时、拒绝重定向；不请求权重正文。
JSON 输出记录扫描错误/截断和源码匹配行；网络失败作为诊断字段报告，不代表兼容性测试失败。

检查前后 8 个被审阅源码/部署文件的 SHA256 均一致。
主要指纹：部署 README `8a01a23f288150c09264b24d4dd68b608ab174b8800b24726d14c036643c57bf`；
pipeline `7f5ff890d796dcfac3551e235741bdfbd4dbcae91aae3b6e4c3817dce38eecd6`；
ControlNet 模型 `7d6782bc7947f52efc02544232a7f37f72773dc94fcc80ba84e4d5c8e6eea78a`。
本任务仅创建本报告及 `scripts/inspect_h3_conditions.py`；未修改既有 H3 或其他协作者文件。
