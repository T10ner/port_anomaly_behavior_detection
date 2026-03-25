# 港口异常行为检测项目

本项目面向**港口泊位及锚地场景的异常行为识别**，当前以 **AIS 轨迹异常候选检测** 为主线，同时开展了 **视频-AIS 多源融合** 的初步预研实验。

目前项目整体定位为：  
**先构建可解释的 AIS 规则法 baseline，再逐步扩展到视频-AIS 关联匹配与融合验证。**

---

## 一、项目目标

当前项目主要围绕以下几个问题展开：

1. 总结港口泊位及锚地常见异常行为；
2. 基于 AIS 轨迹构建可解释的异常行为识别 baseline；
3. 为后续视频-AIS 多源融合方法预留接口并开展初步验证。

当前重点关注的异常行为包括：

- 主航道异常转向
- 主航道低速停留
- 锚地或低速区域的位置异常变化
- 后续拟扩展至异常接近、逆常规流向运动、异常加减速等行为

需要说明的是，当前阶段的输出更适合表述为：

- **异常候选事件**
- **异常候选行为**

而不是最终异常判定结果。

---

## 二、当前已完成内容

### 1. AIS 轨迹异常检测 baseline

已完成的主要模块包括：

- AIS 数据清洗
- 轨迹切分
- 轨迹统计与行为特征构建
- 规则法异常候选检测
- 港口空间区域约束
- 结果可视化与人工核验

当前已实现的候选行为类型包括：

- `abnormal_stop`：低速停留候选
- `sharp_turn`：大角度转向候选
- `drift_like`：低速位置异常候选

同时已引入三类空间区域约束：

- 主航道区域（channel）
- 正常回转/机动区域（turning basin）
- 锚地/等待区域（anchorage）

---

### 2. 视频-AIS 融合预研

在 AIS 主线 baseline 的基础上，已完成视频-AIS 融合方向的初步实验验证，包括：

- FVessel 单个 clip 的数据读取与时间对齐
- AIS 小文件合并与窗口裁剪
- AIS 轨迹概览图生成
- 原始视频上的 YOLO 船舶检测
- 原始视频上的多目标跟踪（tracking）
- 视觉轨迹摘要表构建
- 视频轨迹与 AIS 轨迹的第一版候选匹配

当前融合部分的定位是：

**用于验证“视频-AIS 关联匹配与融合方式”的可行性，  
而不是直接替代 AIS 主线异常检测任务。**

---

## 三、项目结构

```text
port_anomaly_demo/
├─ src/
│  ├─ main.py
│  ├─ anomaly_rules.py
│  ├─ anomaly_rules_v1.py
│  ├─ check_regions.py
│  ├─ regions.py
│  ├─ track_stats.py
│  └─ visualize_events.py
│
├─ config/
│  └─ thresholds.yaml
│
├─ fvessel_fusion/
│  └─ scripts/
│     ├─ inspect_fvessel.py
│     ├─ merge_ais_csvs.py
│     ├─ plot_ais_overview.py
│     ├─ extract_video_frames.py
│     ├─ inspect_gt_files.py
│     ├─ visualize_gt_fusion.py
│     ├─ run_yolo_video.py
│     ├─ run_track_custom_vis.py
│     ├─ build_visual_tracks.py
│     ├─ summarize_matching_candidates.py
│     └─ make_first_match_candidates.py
│
└─ README.md
```

---

## 四、各模块说明

### `src/`
AIS 主线异常检测代码，主要负责：

- AIS 数据清洗与轨迹切分
- 行为统计特征提取
- 异常规则检测
- 结果可视化

### `config/`
配置文件目录，目前主要包含阈值设置。

### `fvessel_fusion/scripts/`
视频-AIS 融合实验脚本目录，主要负责：

- FVessel 数据读取
- AIS 合并与时间对齐
- 视频抽帧与可视化
- YOLO 检测
- 多目标跟踪
- 视觉轨迹构建
- 视频轨迹与 AIS 的候选匹配

---

## 五、运行环境

推荐环境：

- Python 3.10
- Windows 10/11
- NVIDIA GPU（建议 CUDA 可用）
- PyTorch + CUDA
- Ultralytics YOLO


---

## 六、依赖库

常用依赖包括：

- pandas
- numpy
- matplotlib
- opencv-python
- ultralytics
- torch
- torchvision

如需手动安装，可参考：

```bash
pip install pandas numpy matplotlib opencv-python ultralytics
```

如需使用 GPU，请安装带 CUDA 的 PyTorch 版本。

---

## 七、数据说明

### 1. AIS 主线实验数据
当前 AIS baseline 使用港区 AIS 子集开展实验，数据经过区域裁剪与预处理后用于轨迹异常候选识别。

### 2. FVessel 数据
FVessel 数据集主要用于视频-AIS 关联匹配与融合验证。

### 注意
由于数据体积、来源限制及仓库管理考虑，本仓库**不直接提供原始 AIS 数据、FVessel 视频数据及中间输出结果**。  
使用者需要自行准备数据，并放置到本地对应目录后再运行脚本。

---



