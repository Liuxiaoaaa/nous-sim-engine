# nous-sim-engine

RL-optimized closed-loop scoring engine based on NavSim's PDM (Predictive Driver Model). Reimplements and extends the PDM scoring pipeline as a standalone, GPU-free HTTP microservice, with continuous reward functions designed for GRPO training.

## Motivation

NavSim's original `PDMScorer` is designed for **evaluation**, not **training**:

| Problem | NavSim PDM | nous-sim-engine |
|---------|-----------|-----------------|
| **Sparse rewards** | Binary 0/1 metrics — one collision zeros the entire score | Continuous [0,1] with per-timestep decay, soft safety gate |
| **No gradient signal** | Near-miss scores the same as perfect driving | Overlap severity accumulates across frames, light brush > deep collision |
| **Batch normalization** | EP normalized across batch (non-stationary) | GT-progress normalization (deterministic per trajectory) |
| **Interface mismatch** | Requires 11D state vectors `(B, T, 11)` | Accepts raw `(x, y)` waypoints — matches VLM output directly |
| **Environment coupling** | Requires NavSim + nuPlan in training | Decoupled HTTP service, zero NavSim dependency at training time |
| **No caching** | Deserializes MetricCache pickle every call (~1s) | 3-tier cache: LRU (22ms) → boost SSD (66ms) → LZMA fallback (1s) |

### Architecture

```
VLM (GRPO training)                    nous-sim-engine (scoring service)
    |                                       |
    |  POST /v1/score/rl/batch              |
    |  { trajectories: [[[x,y], ...]],      |
    |    scene_token, log_name,             |
    |    scoring_mode: "continuous" }        |
    |-------------------------------------->|
    |                                       |  1. Load SceneContext (3-tier cache)
    |                                       |  2. ego-relative → global transform
    |                                       |  3. Kinematic bicycle simulation (LQR)
    |                                       |  4. Compute metrics (NC/DAC/EP/TTC/HC)
    |                                       |  5. Soft safety gate aggregation
    |  { results: [{rl_score: 0.82,         |
    |     sub_rewards: {nc, dac, ...}}] }   |
    |<--------------------------------------|
```

## Continuous RL Scoring

### NC (No At-Fault Collision) — Per-timestep Collision Decay

Full at-fault classification reuses PDMS logic (5 collision types). For at-fault collisions, each timestep independently decays the score:

```
severity_t = overlap_area / ego_area
cumulative = ∏(1 - severity_t) across all collision frames
NC = floor + (1 - floor) × cumulative
```

Where `floor = 0.0` (agent) or `0.5` (static obstacle). Encodes both collision depth and duration without artificial parameters.

### DAC (Drivable Area Compliance) — Per-timestep Coverage Product

Route-aware spatial filtering + multiplicative time decay:

```
eligible = route_lane_ids ∪ structural_layers(roadblock, intersection)
coverage(t) = ego_poly ∩ eligible_union / ego_area
DAC = ∏ coverage(t)
```

Uses STRtree spatial index for efficient polygon queries.

### EP (Ego Progress) — GT-Progress Normalization

```
EP = clip(raw_progress / gt_progress, 0, 1)    # if gt_progress > 5m
EP = clip(raw_progress / 5.0, 0, 1)            # fallback
```

### Aggregation — Soft Safety Gate

```
safety = NC × DAC × DDC × TLC
safety_gate = safety ^ α                        # α = 0.5
performance = weighted_avg(EP×5, TTC×5, HC×2)
rl_score = safety_gate × performance
```

Not pure multiplicative (bimodal) or additive (weak penalty), but a balanced soft gate.

### Dual Scoring Modes

| Mode | Endpoint | Safety | Performance | Aggregation | Use Case |
|------|----------|--------|-------------|-------------|----------|
| **PDMS** | `/v1/score/batch` | Binary | Weighted | Multiplicative | NavSim eval |
| **RL continuous** | `/v1/score/rl/batch` | Continuous | Continuous | Soft gate | RL training |
| **RL discrete** | `/v1/score/rl/batch` | Binary | Continuous | Soft gate | RL comparison |

## Project Structure

```
nous-sim-engine/
├── src/nous_sim_engine/
│   ├── core/                    # Scoring engine (no HTTP dependency)
│   │   ├── scorer.py            #   PDMScorer + RLScorerConfig
│   │   ├── simulator.py         #   Kinematic bicycle model + LQR tracker
│   │   ├── geometry.py          #   Coordinate transforms, PDMPath
│   │   ├── observation.py       #   Time-indexed occupancy maps
│   │   ├── occupancy.py         #   STRtree spatial index, DrivableMap
│   │   ├── comfort.py           #   Comfort thresholds
│   │   ├── enums.py             #   StateIndex, CollisionType, SemanticMapLayer
│   │   └── types.py             #   SceneContext, ScoringResult, VehicleParams
│   ├── server/                  # FastAPI HTTP layer
│   │   ├── app.py               #   Endpoints: score, score/batch, score/rl/batch
│   │   ├── schemas.py           #   Pydantic models + RLConfigOverrides
│   │   └── registry.py          #   Dataset directory registry
│   ├── adapters/navsim/         # NavSim MetricCache adapter
│   │   └── cache_loader.py      #   LZMA pickle → SceneContext + 3-tier cache
│   ├── client.py                # HTTP client (stdlib-only)
│   └── __main__.py              # CLI: python -m nous_sim_engine
├── scripts/
│   ├── navsim_scorer.py         # Cross-validate vs NavSim reference
│   └── validate_rl_scoring.py   # RL vs PDM consistency check
└── tests/
    ├── conftest.py              # Synthetic scene fixtures
    ├── test_rl_continuous.py    # 23 tests for continuous scoring
    └── test_server_client.py    # E2E server + client tests
```

## Installation

```bash
pip install -e .            # Core library
pip install -e ".[server]"  # With HTTP server
```

## Usage

### HTTP Server

```bash
# With dataset and boost cache
python -m nous_sim_engine --port 8100 \
  --dataset navtest=/path/to/metric_cache_navtest \
  --boost-cache-dir /path/to/ssd/boost_cache
```

### Score Trajectories (RL)

```bash
curl -X POST http://localhost:8100/v1/score/rl/batch \
  -H "Content-Type: application/json" \
  -d '{
    "trajectories": [[[0.5, 0.0], [1.0, 0.0], ...]],
    "scene_token": "0a678d2136b35b56",
    "log_name": "2021.05.12.19.36.12_veh-35_00005_00204",
    "dataset": "navtest",
    "scoring_mode": "continuous"
  }'
```

### Python Client

```python
from nous_sim_engine import SimEngineClient

client = SimEngineClient("http://localhost:8100")
results = client.score_rl_batch(
    trajectories=[[[0.5, 0.0], [1.0, 0.0], ...]],
    scene_token="0a678d2136b35b56",
    log_name="2021.05.12...",
    dataset="navtest",
    scoring_mode="continuous",
)
```

### Library Usage

```python
from nous_sim_engine import PDMScorer
from nous_sim_engine.core.scorer import RLScorerConfig

scorer = PDMScorer()
rl_result = scorer.score_for_rl(
    trajectory_xy, scene_context,
    rl_config=RLScorerConfig(safety_mode="continuous"),
)
```

## Validation Results (3600 trajectories)

| Metric | Result |
|--------|--------|
| NC false positive (PDMS=1, cont<1) | 1.7% |
| DAC edge-brush (PDMS=1, cont<1) | 3.2% |
| EP Spearman (safe trajectories) | 0.67 |
| Safe vs Collision score gap | 0.64 |

| Category | N | PDMS | Continuous |
|----------|---|------|-----------|
| Safe (NC=1, DAC=1) | 1807 | 0.859 | 0.861 |
| Collision (NC<1) | 633 | 0.000 | 0.217 |
| Offroad (DAC<1) | 1160 | 0.000 | 0.141 |

Continuous scoring provides non-zero gradient signal for collision/offroad trajectories while maintaining separation from safe trajectories.

## Benchmark

| Cache Layer | Latency | vs NavSim |
|-------------|---------|-----------|
| L1: LRU memory | 22 ms | 45x |
| L2: Boost SSD | 66 ms | 15x |
| L3: LZMA fallback | ~1000 ms | 1x |

Scoring throughput: ~30 traj/s (single worker, 32 concurrent clients).

## Testing

```bash
pytest tests/ -v
```
