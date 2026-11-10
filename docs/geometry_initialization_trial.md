# RGB 到粗白模：实际实验记录

## 后续核查与修正

用户指出 comparison 的数据集白模与 RGB 不匹配。复核发现源 RGB 与 proxy 均为第 480 帧，与训练样本逐像素一致，没有选错帧。原 proxy 人物框与 RGB 检测框水平中心仅差约 1.99 像素，但 proxy 固定张臂人形宽约 214 像素、RGB 宽约 99 像素。背景取自碰撞几何，编译清单明确 `static_session_match_verified=false`，不能当作录制画面的精确视觉 GT。具体背景差异是否来自地图状态还没有定论。

该样本确实出现在后续 unified_delivery 的 train 清单第 5305 行，旧审核把它标为 `alignment_ok=true`；这不能再解释为严格几何对齐通过。此发现没有证明全包错误，也没有完成全包重新验收。原交付文件未修改。证据在 `outputs/geometry_source_audit_20261103/audit.json`。

继续优化已实现：重复 mask 去重，建筑按局部可见立面分块，人物遮挡处用两侧地面拟合补全，旧区域逐项诊断而不整体覆盖。三帧到四帧的对齐内点比例约 91.77%，18 个静态区域中 10 个保持一致、1 个候选平移待审、7 个形状冲突。该检查不等于已经允许世界合并。

`refinement_comparison.jpg` 展示修正标注和优化前后。`refined_assets` 含 61 个立面片、71 个粗实体，移除 1 个重复 mask；补全 1,827 像素地面，青色标记推断部分。部分建筑覆盖率仍低，未拟合区域保持未知，不意味着建筑消失。最新 compiler 会把原建筑候选单独保存在 `source_candidates.json`，不会把它们强行加入碰撞。新模块已部署；资产、配准、局部检查、地面补全、立面分块合计 55 项本地测试通过，其中新增地面、立面、局部检查的 33 项也在服务器通过。

## 本轮做了什么

在 GTA urban 样本上跑通了：RGB → VGGT 深度与相机 → GroundingDINO + SAM2 分割 → 点云区域 → 粗资产 → Godot 渲染。另做了单帧、双帧、三帧重建，以及利用共同帧把多视角重建对齐到单帧坐标系的实验。

输入来自 `ep_20260918_132805_7503_agent_000_segment_00000_f000480`。模型只看真实 RGB。数据集白模、相机记录独立保存在 evaluation，只作验收。三帧分别是目标首帧、历史参考帧、目标视频第 60 帧。

本地结果：`outputs/geometry_init/urban/`。
服务器：`/home/star/agentic_world/outputs/geometry_init/urban/`，VGGT 原始输出在 `outputs/vggt_20261103/`。

先看 `comparison.jpg` 和 `multiview_comparison.jpg`。单张图可打开 `coarse_assets/camera.png`，三帧版本是 `multiview_assets/camera.png`。初次失败的地面叠加结果保存在 single_assets，排查过程保留但不作为最终候选。

## 结果

- 单帧得到 21 个分割区域，转换为 19 个粗物体和可见地面三角面。道路与人行道作为地面，不重复导出成覆盖地形的大 box。
- 分割复用服务器本地 GroundingDINO、SAM2.1、DINOv2。道路在联合词表下出现混合标签，补跑独立 `road.` 检测后得到完整道路 mask。没有人工框点；补跑选择是本次实验操作，尚未做成通用自动发现策略。
- 建筑用可见点的稳健有向包围盒；人物用已有 gta_human 粗比例。人物朝向未知，默认方向仅供预览，行为尚未规划。没有重建骨骼。
- 单图没有可靠米制尺度。本次用前景人物可见高度 1.7 米作先验；这一选择明确记录在 report.json，没有借用相机真值定尺度。
- 单帧 VGGT 前向约 0.58 秒，加载、预处理和前向合计约 10.83 秒，不含首次权重下载。三帧整条推理命令实测 14.68 秒，含进程启动及输出落盘；不是完整初始化耗时。
- 本次原图分割两次检测合计约 3.93 秒，SAM 约 0.74 秒，embedding 约 1.19 秒。这些是脚本计时阶段，不包含全部进程及人工验收时间。
- 双帧相机相对旋转误差约 0.88 度，平移方向误差约 3.95 度。数据相机由记录插值而来，时间对齐不是精确真值。
- 三帧三个相机对的平均旋转误差约 0.62 度，平均平移方向误差约 2.67 度；竖直焦距平均相对误差约 10.58%，说明预测相机仍有系统性偏差。第三帧的导出文件名不同，验收通过显式映射和原训练视频解码核对来源，没有按文件名盲猜。
- 单帧粗资产投影与同一批 SAM bbox 的中位 IoU 约 0.768，中心误差约 4.02 像素，画幅 518×294。这是内部投影一致性，不能当作独立三维精度。

## 新视角对齐暴露的问题

对同一张 target RGB 在单帧和双帧重建中的逐像素点做 Sim(3) 拟合，先排除人、车与 HUD，使用已分割的静态背景。相同 RGB 和预处理网格经过核对，不用未知对应点盲目 ICP。

双帧约 66.2% 的静态点进入全局对齐内点。近景内点比例约 99.99%，远景只有约 8.15%。这里的固定绝对阈值会对远景更严格，因此另外报告按每个像素深度归一化的误差：近景中位约 2.09%，远景约 8.59%。这些是两次模型重建之间的差异，不是对真实世界的精度。

三帧也能拟合全局坐标变换，但内点比例约 63.8%，并未解决全场景一致性。两个实验均为 `world_merge_ready=false`。只获得可供局部检验的坐标变换，不自动提交世界修改。

已把固定的旧候选资产放到新估计相机下渲染，保留相同实体。这直接暴露单图遮挡造成的地面空洞、建筑厚度不确定、树冠 box 化等问题。历史参考里人物处于不同时刻，所以该图只检查静态背景，不用来验证历史人物位置。

## 接入边界

当前输出是能用现有 Godot 资产接口渲染的候选 scene.json，不是已经验收可漫游的完整世界。地面只保留可见区域，遮挡处没有自动补齐；人物脚下可能缺少支撑，需要显式补全并再次验收后才能进入动作模拟。生成的静态几何相对路径供 scene 文件渲染使用；正式写入数据库前需解析为部署位置并通过运动检查。

未修改任何已有 world.sqlite、已接受历史、H3 推理配置或环境依赖。没有调用付费 GPT API，没有启动新的 H3 视频生成。VGGT 使用官方代码和镜像站权重；本轮未运行 MapAnything 或 SAM 3D。

本地及服务器通过 4 项资产几何测试和 18 项对齐/验收测试；相机指标另有尺度不变性、方向符号、零基线等自检。服务器没有 matplotlib，误差图改用现有 OpenCV/Pillow 绘制后测试通过，没有额外安装绘图库。

下一步优先改善区域级几何表达：地面延续、建筑分块与去重、置信度随距离变化的局部对齐。新区域应提出新增资产，旧区域保留 ID 并逐区域判断是否有足够证据修改，不能每次整体覆盖。

## 复现入口

在 agentic_world 下执行，VGGT 和分割在服务器使用既有 Python 环境。代码不更改环境依赖。

```bash
python scripts/prepare_geometry_trial.py
python scripts/reconstruct_vggt.py INPUT_RGB --output OUTPUT.npz --gpu 2 --local-files-only
python scripts/probe_perception.py --image INPUT_RGB --output PERCEPTION_DIR --device cuda:0 --text "person. car. building. road. sidewalk. tree. pole. wall."
python scripts/compile_geometry_assets.py --reconstruction outputs/geometry_init/urban/reconstruction/single.npz --perception outputs/geometry_init/urban/perception_target/result.json --output outputs/geometry_init/urban/new_candidate --actor-height-prior 1.7 --plane-threshold 0.005
python -m whitebox.godot.render --scene outputs/geometry_init/urban/new_candidate/scene.json --output outputs/geometry_init/urban/new_candidate/render --width 1036 --height 588
python scripts/evaluate_geometry_alignment.py
python scripts/evaluate_geometry_alignment.py --joint outputs/geometry_init/urban/reconstruction/joint3.npz --anchor-index 0 --output outputs/geometry_init/urban/alignment_three_frames
python scripts/test_geometry_assets.py
python scripts/test_geometry_alignment.py
python scripts/evaluate_geometry_cameras.py --self-test
```

`prepare_geometry_trial.py` 当前是这一个 GTA 样本的实验导出器，需要本机 E 盘源数据。`segment_geometry.py` 是本次 road 补跑结果的合并工具，不能当作通用自动分割器。完整时间、来源、阈值与逐物体残差以结果目录 JSON 为准。
