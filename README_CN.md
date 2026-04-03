# nous-sim-engine

基于 NavSim PDM（Predictive Driver Model）的 RL 优化闭环评分引擎。独立重实现并扩展了 PDM 评分流水线，设计为无 GPU 依赖的 HTTP 微服务，核心改进是面向 GRPO 训练的连续奖励函数。

## 设计动机

### NavSim PDM 用于 RL 训练的问题

NavSim 原生 PDMScorer 是为 **评估（eval）** 设计的，直接用于 RL 训练存在根本性问题：

| 问题 | NavSim PDM | nous-sim-engine |
|------|-----------|-----------------|
| **奖励稀疏** | 二值 0/1 指标 + 乘法聚合，一次碰撞整体得分归零 | 连续 [0,1] 指标 + 加法聚合，每个维度独立贡献 |
| **无梯度信号** | 差 1cm 擦碰 = 完美驾驶（得分相同），RL 对"差一点对"的轨迹无学习信号 | 近距离给部分惩罚，碰撞严重程度按重叠面积缩放 |
| **批次依赖** | Ego Progress 在 batch 内归一化 (`progress / max_progress`)，奖励非平稳 | 绝对归一化 (`progress / threshold`)，同一轨迹奖励确定性 |
| **接口不匹配** | 需要 11 维状态向量 `(B, T, 11)` 含速度/加速度/转向角 | 直接接受 ego-relative `(x, y)` 路点，匹配 VLM 输出 |
| **环境耦合** | 训练进程须安装 NavSim + nuPlan (~2GB) | 训练端零依赖，HTTP 解耦 |
| **无缓存** | 每次调用反序列化 MetricCache pickle (~1s/scene，含大量嵌套 NavSim 对象) | 三级缓存：L1 LRU 内存 (22ms) → L2 Boost SSD (66ms) → L3 LZMA 回退 (1s) |

## 相对 NavSim PDM 的改进

### 改进 1：连续安全指标

NavSim 4 个安全指标全部是二值 pass/fail。我们为每个指标设计了基于距离的连续替代版本：

#### NC（无过错碰撞）

```
NavSim:  ████████████ 1.0 ████████████ | 0.0 (碰撞)
                                        ↑ 悬崖式跌落

Ours:    ████████ 1.0 ████ 0.8 ██ 0.5 █ 0.2 | 0.0
                  ← 缓冲区 →          ↑ 重叠区
```

| | NavSim | nous-sim-engine |
|---|--------|----------------|
| 碰撞 | → 0 | 按重叠面积缩放: `1 - 2 × (overlap / ego_area)` |
| 近距离 | → 1（无惩罚） | 缓冲区内线性衰减: `1 - nearness × 0.5` |
| 远距离 | → 1 | → 1（TTC 负责远距离检测） |

#### DAC（可行驶区域合规）

| NavSim | nous-sim-engine |
|--------|----------------|
| 任何离开道路 → 0 | 中心离开 → 0，仅车角离开 → 按 `dac_margin` 部分惩罚 |

#### DDC（行驶方向合规）

| NavSim | nous-sim-engine |
|--------|----------------|
| 三档: 1.0 / 0.5 / 0.0 | 阈值间线性插值，平滑过渡 |

#### TLC（交通灯合规）

| NavSim | nous-sim-engine |
|--------|----------------|
| 闯红灯 → 0 | `tlc_margin` 缓冲区内距离衰减，穿越 → 0 |

### 改进 2：连续性能指标

RL 模式下性能指标始终连续（即使在 `discrete` 子模式下）：

| 指标 | NavSim | nous-sim-engine |
|------|--------|----------------|
| **EP 行驶进度** | 批次相对: `progress / max(batch)` | 绝对: `progress / threshold`，无批次依赖 |
| **TTC 碰撞时间** | 二值: 有违规 → 0 | 时间归一化: `first_violation_time / horizon` |
| **LK 车道保持** | 二值: 连续偏差超阈值 → 0 | 均值偏差线性衰减: `1 - (mean_dev - lo) / (hi - lo)` |
| **HC 舒适性** | 二值: 任一阈值超标 → 0 | 最大违规比: `1 - clip(max(|val|/threshold - 1), 0, 1)`，7 个阈值 |

### 改进 3：加法聚合

| | NavSim PDM | nous-sim-engine RL |
|---|-----------|-------------------|
| **公式** | `prod(安全) × weighted_avg(性能)` | `weighted_sum(全部8项) / sum(权重)` |
| **效果** | 一个安全违规归零全部 | 每个指标独立贡献 |
| **RL 信号** | "你失败了"（无方向） | "碰撞很差，但进度不错"（分解反馈） |

### 改进 4：双评分模式

同一服务支持三种模式，评估和训练统一：

| 模式 | 端点 | 安全指标 | 性能指标 | 聚合 | 用途 |
|------|------|---------|---------|------|------|
| **PDM** | `/v1/score` | 二值 | 加权 | 乘法 | eval benchmark（NavSim 兼容） |
| **RL discrete** | `/v1/score/rl` + `discrete` | 二值 | 连续 | 加法 | RL 基线对比 |
| **RL continuous** | `/v1/score/rl` + `continuous` | 连续 | 连续 | 加法 | **RL 训练（推荐）** |

## 架构

```
训练进程 (ms-swift GRPO)                nous-sim-engine (评分服务)
    |                                       |
    |  POST /v1/score/rl                    |
    |  { trajectory: [[x,y], ...],          |
    |    scene_token, log_name }            |
    |-------------------------------------->|
    |                                       |  1. 加载 SceneContext (LRU 缓存)
    |                                       |  2. ego-relative → 全局坐标变换
    |                                       |  3. 自行车运动学模型仿真
    |                                       |  4. 计算 8 个子奖励
    |  { rl_score: 0.82,                    |
    |    sub_rewards: {nc, dac, ...} }      |
    |<--------------------------------------|
```

### 目录结构

```
nous-sim-engine/
├── src/nous_sim_engine/
│   ├── core/                    # 评分引擎核心（无 HTTP 依赖）
│   │   ├── scorer.py            #   PDMScorer + RLScorerConfig: 8 指标, 3 种模式
│   │   ├── simulator.py         #   自行车运动学模型 + LQR 轨迹跟踪
│   │   ├── geometry.py          #   坐标变换, PDMPath 中心线
│   │   ├── observation.py       #   时序占用图 (障碍物 + 红灯)
│   │   ├── occupancy.py         #   STRtree 空间索引, DrivableMap
│   │   ├── comfort.py           #   7 阈值舒适性 (二值 + 连续)
│   │   ├── enums.py             #   StateIndex, CollisionType 等枚举
│   │   └── types.py             #   SceneContext, ScoringResult 数据结构
│   ├── server/                  # FastAPI HTTP 层
│   │   ├── app.py               #   5 个端点 + RLConfigOverrides 动态调参
│   │   └── schemas.py           #   Pydantic 请求/响应模型
│   ├── adapters/navsim/         # NavSim MetricCache 适配器
│   │   └── cache_loader.py      #   LZMA pickle → SceneContext + 三级缓存 (LRU/Boost/LZMA)
│   ├── client.py                # HTTP 客户端 (纯标准库, 零依赖)
│   └── __main__.py              # CLI 入口
├── scripts/
│   ├── navsim_scorer.py         # 与 NavSim 参考实现交叉验证
│   └── validate_rl_scoring.py   # RL vs PDM 指标一致性验证
└── tests/
    ├── conftest.py              # 合成场景 fixtures
    └── test_server_client.py    # 端到端 server + client 测试
```

## 评分流水线

### 输入 → 输出

```
输入: ego-relative (x, y) 路点 (VLM 输出)
  ↓
坐标变换: ego → 全局坐标, 从相邻点计算朝向角
  ↓
运动学仿真: 自行车模型 + LQR 控制器 → 完整 11D 状态序列
  ↓
区域判定: 可行驶区域、对向车道、交叉路口、多车道
  ↓
8 个子指标并行计算
  ↓
聚合: PDM (乘法) 或 RL (加权求和)
  ↓
输出: 总分 + 8 个子奖励
```

### 8 个子指标

| 指标 | 缩写 | PDM | RL 权重 | 连续模式改进 |
|------|------|-----|---------|------------|
| 无过错碰撞 | NC | 乘法 | 5.0 | 重叠面积缩放 + 缓冲区近距离惩罚 |
| 可行驶区域 | DAC | 乘法 | 3.0 | 中心/车角分级惩罚 |
| 行驶方向 | DDC | 乘法 | 2.0 | 阈值间线性插值 |
| 交通灯 | TLC | 乘法 | 3.0 | 距离衰减缓冲区 |
| 行驶进度 | EP | 加权 5.0 | 5.0 | 绝对归一化（无批次依赖） |
| 碰撞时间 | TTC | 加权 5.0 | 5.0 | 首次违规时间归一化 |
| 车道保持 | LK | 加权 2.0 | 2.0 | 均值偏差线性衰减 |
| 舒适性 | HC | 加权 2.0 | 2.0 | 最大违规比（7 阈值） |

## 性能对比

### 三级缓存性能

| 缓存层 | 延迟 | vs NavSim (1s) | 触发条件 |
|--------|------|---------------|---------|
| **L1: LRU 内存** | **22 ms** | **45x** | 同一 scene，同一 worker |
| **L2: Boost SSD** | **66 ms** | **15x** | warmup 完成后首次访问 |
| L3: LZMA 回退 | ~1000 ms | 1x | warmup 未完成时 |

### nous-sim-engine vs NavSim 进程内调用

| 指标 | NavSim 进程内 | nous-sim-engine | 提升 |
|------|-------------|-----------------|------|
| 首次调用（无 boost） | ~1.0s | ~1.0s | 1x |
| **首次调用（有 boost）** | ~1.0s | **66ms** | **15x** |
| **重复调用（LRU 命中）** | **~1.0s**（无缓存） | **22ms** | **45x** |
| 环境依赖 | NavSim + nuPlan ~2GB | 训练端零依赖 | 环境隔离 |
| 并发能力 | 单线程 | 多 worker 并行 | 线性扩展 |

### GRPO 训练加速（核心收益）

GRPO 每个 scene 采样 G 次（group_size 个 response）：

```
                         无 boost cache         有 boost cache
NavSim 进程内:            G × 1.0s               G × 1.0s

nous-sim-engine:
  G=4:   1.0 + 3×0.02 = 1.06s (1x)      0.07 + 3×0.02 = 0.13s  (31x)
  G=8:   1.0 + 7×0.02 = 1.14s (7x)      0.07 + 7×0.02 = 0.21s  (38x)
  G=16:  1.0 + 15×0.02 = 1.30s (12x)    0.07 + 15×0.02 = 0.37s (43x)
```

### Boost Cache Warmup

| 指标 | 值 |
|------|-----|
| 场景数 | 103,288 |
| 单 scene 转换 | ~163 ms |
| 32 线程预热耗时 | **~9 分钟** |
| SSD 占用 | ~148 GB |
| AFS 存储（不变） | 35 GB |

Warmup 在**独立子进程**中运行，不与 server 主线程争抢 GIL。server 在预热期间完全可用。

### 延迟分解（L3 路径）

| 组件 | 耗时 | 说明 |
|------|------|------|
| 磁盘读取 | ~0.5 ms | 540 KB 压缩文件 |
| LZMA 解压 | ~22 ms | 解压到 1.5 MB |
| **Pickle 反序列化** | **~1080 ms** | **瓶颈 — boost cache 直接消除** |
| 评分计算 | **65-75 ms** | 仿真 + 8 指标计算 |
| HTTP 开销 | <5 ms | FastAPI + JSON 序列化 |

### 额外工程收益

| 收益 | 说明 |
|------|------|
| **环境解耦** | 训练进程无需安装 NavSim/nuPlan，避免与 ms-swift、transformers 版本冲突 |
| **水平扩展** | `--workers N` 多进程并行，每个 worker 独立 LRU 缓存 |
| **故障隔离** | scorer crash 不影响训练进程，自动重启即可 |
| **灵活部署** | 可部署在训练节点 localhost，也可部署在专用评分节点 |
| **动态调参** | 权重和阈值支持逐请求覆盖，无需重启服务 |

## 测评数据

在训练机（208 CPU cores, H20 节点）上实测。

### 延迟分布 (30 scenes)

| 指标 | 值 |
|------|-----|
| L2 Boost 首次加载 | **66ms** |
| L1 LRU 命中 | **22ms** |
| L3 LZMA 回退（无 boost） | ~1000ms |

### 评分分布 (continuous 模式)

| 统计量 | 值 |
|--------|-----|
| 均值 | 0.536 |
| 标准差 | 0.100 |
| 最小值 | 0.208 |
| 最大值 | 0.699 |

### Continuous vs Discrete 模式对比

| 统计量 | 值 |
|--------|-----|
| 均值差异 | 0.065 |
| 最大差异 | 0.130 |

Continuous 模式得分更高，因为 near-miss 场景在 discrete 下为 0，而 continuous 给出部分分数 — 这正是 RL 需要的梯度信号。

### MetricCache 兼容性

自动适配两种 NavSim MetricCache 版本：

| 版本 | `log_name` 属性 | 处理方式 |
|------|----------------|---------|
| reference_navsim | 有 | 直接读取 |
| recogdrive | 无 | 从 `file_path` 自动提取 |

## 安装

```bash
# 仅核心库
pip install -e .

# 含 HTTP 服务
pip install -e ".[server]"
```

## 使用方法

### 启动服务

```bash
# 基础模式（仅 LRU 缓存）
conda activate navsim
python -m nous_sim_engine --host 0.0.0.0 --port 8100 --workers 4

# 生产模式（推荐：boost cache 加速）
python -m nous_sim_engine --host 0.0.0.0 --port 8100 --workers 1 \
  --metric-cache-dir /path/to/metric_cache_navtrain \
  --boost-cache-dir /path/to/ssd/boost_cache \
  --warmup-workers 32
```

启动时带 `--boost-cache-dir`，后台子进程自动将 LZMA MetricCache 转换为快速 SceneContext pickle 写入 SSD。server 立即可用 — 未转换的 scene 走 LZMA 回退并 lazy 回写到 boost cache。

### 评分请求

```bash
curl -X POST http://localhost:8100/v1/score/rl \
  -H "Content-Type: application/json" \
  -d '{
    "trajectory": [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], ...],
    "scene_token": "0a678d2136b35b56",
    "log_name": "2021.05.12.19.36.12_veh-35_00005_00204",
    "metric_cache_dir": "/path/to/metric_cache_navtrain",
    "scoring_mode": "continuous"
  }'
```

### Python 客户端

```python
from nous_sim_engine import SimEngineClient

client = SimEngineClient("http://localhost:8100")
score, result = client.score_rl(
    trajectory=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
    scene_token="0a678d2136b35b56",
    log_name="2021.05.12.19.36.12_veh-35_00005_00204",
    metric_cache_dir="/path/to/metric_cache",
    scoring_mode="continuous",
)
```

### 库模式（无 server）

```python
from nous_sim_engine import PDMScorer
from nous_sim_engine.core.scorer import RLScorerConfig

scorer = PDMScorer()

# NavSim 兼容 eval
result = scorer.score(trajectory_xy, scene_context)

# RL 训练奖励
rl_result = scorer.score_for_rl(
    trajectory_xy, scene_context,
    rl_config=RLScorerConfig(safety_mode="continuous"),
)
```

### 动态调参

```json
{
  "scoring_mode": "continuous",
  "config_overrides": {
    "nc_weight": 10.0,
    "ep_weight": 3.0,
    "collision_distance_scale": 3.0,
    "progress_distance_threshold": 10.0
  }
}
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/health` | GET | 健康检查 + LRU 缓存统计 + boost cache 预热进度 |
| `/v1/score` | POST | PDM 评分 (NavSim 兼容, 乘法聚合) |
| `/v1/score/batch` | POST | 批量 PDM 评分 |
| `/v1/score/rl` | POST | RL 奖励评分 (加法聚合, continuous/discrete) |
| `/v1/score/rl/batch` | POST | 批量 RL 评分 |

## 测试

```bash
# 单元测试
pytest tests/ -v

# RL vs PDM 一致性验证
python scripts/validate_rl_scoring.py --cache-dir /path/to/metric_cache

# 与 NavSim 交叉验证
python scripts/navsim_scorer.py --cache-dir /path/to/metric_cache \
  --log-name "2021.05.12..." --token "0a678d21..."
```
