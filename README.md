---
title: ko2cube
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Ko2cube: Carbon-Aware Kubernetes Job Scheduling Environment

Ko2cube is a carbon-aware cloud job scheduling environment designed for training LLM agents using reinforcement learning. It challenges AI agents to act as Infrastructure Engineers who must optimize the placement of diverse compute workloads across multiple cloud regions.

**Primary Goal**: Minimize carbon emissions while maintaining SLA compliance and managing infrastructure costs.

## Key Features

- **Cloud-Agnostic Design**: Abstracted carbon data providers supporting WattTime, Electricity Map, or static CSV data
- **Realistic Infrastructure**: Simulates AWS EC2 instance types (m5, c5, r5) with CPU/Memory constraints and Spot/On-Demand pricing
- **Dynamic Carbon Data**: 12-week high-fidelity synthetic timeseries with daily solar/wind cycles and weekly demand patterns
- **Curriculum Learning**: Progressive difficulty system that adapts to agent performance
- **Comprehensive Evaluation**: Baseline agents and statistical evaluation framework
- **OpenEnv Compatible**: Follows OpenEnv specification for RL environment servers

## Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With development dependencies (testing)
pip install -e ".[dev]"

# With provider dependencies (real-time carbon APIs)
pip install -e ".[providers]"

# With training dependencies (LLM fine-tuning)
pip install -e ".[train]"

# Everything
pip install -e ".[all]"
```

### Running the Environment Server

```bash
# Start the OpenEnv-compatible server
python -m server.app

# Or use uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=server --cov-report=html
```

### Evaluating Baseline Agents

```bash
# Compare all baseline agents on easy scenario
python eval.py --all-baselines --scenarios easy --episodes 5

# Evaluate a specific agent across difficulties
python eval.py --agent oracle --scenarios easy,medium,hard --episodes 10

# Run curriculum-based evaluation
python eval.py --agent hybrid --curriculum --episodes 30
```

### Training an LLM Agent

```bash
# Lightweight training (without LLM, for testing pipeline)
python train.py --no-llm --episodes 50 --curriculum

# Full training with GRPO (requires GPU and train dependencies)
python train.py --model Qwen/Qwen2.5-3B-Instruct --episodes 500 --curriculum
```

## Architecture

```
ko2cube/
├── server/
│   ├── app.py              # FastAPI server (OpenEnv interface)
│   ├── environment.py      # Core environment logic
│   ├── rewards.py          # Reward calculations and grader
│   ├── baselines.py        # Baseline agents for comparison
│   ├── curriculum.py       # Curriculum learning system
│   ├── adversarial.py      # Adversarial scenario generator
│   ├── metrics.py          # Prometheus metrics
│   ├── providers/          # Carbon data providers
│   │   ├── base.py         # Abstract CarbonProvider interface
│   │   ├── static.py       # CSV-based static provider
│   │   ├── watttime.py     # WattTime API integration
│   │   └── electricity_map.py  # Electricity Map API
│   └── data/
│       ├── scenarios.py    # Scenario definitions
│       ├── infrastructure.json  # Instance types and regions
│       └── cleaned_timeseries_data.csv  # Carbon timeseries
├── models.py               # Pydantic data models
├── eval.py                 # Evaluation framework
├── train.py                # Training script with GRPO
└── tests/                  # Comprehensive test suite
```

## Environment Interface

### Observation Space

The agent receives a `Ko2cubeObservation` containing:

| Field | Type | Description |
|-------|------|-------------|
| `current_step` | int | Current simulation step |
| `job_queue` | List[Job] | Jobs waiting to be scheduled |
| `active_jobs` | List[RunningJob] | Jobs currently running |
| `regions` | Dict[str, RegionInfo] | Region data with carbon and pricing |

**Job Properties:**
- `job_id`: Unique identifier
- `cpu_cores`, `memory_gb`: Resource requirements
- `sla_start`, `sla_end`: Valid scheduling window
- `delay_tolerant`: Whether job can be deferred
- `instance_preference`: "spot" or "on-demand"

**Region Properties:**
- `carbon.current_intensity`: Current gCO2/kWh
- `carbon.forecast`: 24-hour carbon forecast
- `available_instances`: List of available instance types

### Action Space

The agent submits a `Ko2cubeAction` with assignments for each job:

| Decision | Description | Required Fields |
|----------|-------------|-----------------|
| `schedule` | Run job immediately | `region`, `instance_type`, `machine_type` |
| `defer` | Wait for better conditions | `defer_to_step` |
| `drop` | Abandon job (SLA violation) | None |

### Example Action

```python
action = Ko2cubeAction(assignments=[
    JobAssignment(
        job_id="etl_sales_001",
        decision="schedule",
        region="us-west-2",
        instance_type="m5.xlarge",
        machine_type="spot"
    ),
    JobAssignment(
        job_id="ml_training_002",
        decision="defer",
        defer_to_step=5  # Wait for cleaner grid
    ),
])
```

## Task Scenarios

### Task 1: Easy (Single Region)

- **Regions**: us-east-1 only
- **Jobs**: Delay-tolerant batch jobs (ETL, reports)
- **Duration**: 12 steps (half day)
- **Objective**: Learn to defer work from fossil-fuel peaks to solar peaks

### Task 2: Medium (Multi-Region)

- **Regions**: us-east-1, us-west-2, eu-west-1
- **Jobs**: Mix of delay-tolerant and time-sensitive (CI/CD)
- **Duration**: 24 steps (1 day)
- **Objective**: Triage workloads and route to cleanest regions

### Task 3: Hard (Full Complexity)

- **Regions**: us-east-1, us-west-2, eu-west-1
- **Jobs**: All types including always-on API serving
- **Duration**: 48 steps (2 days)
- **Challenges**: Mid-episode burst, always-on constraints, capacity limits

## Reward System

The reward function combines multiple components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Carbon Savings | 0.50 | Percentage below baseline carbon intensity |
| Cost Savings | 0.25 | Percentage below baseline cost |
| SLA Compliance | 0.25 | Jobs completed within SLA window |
| Penalties | N/A | SLA violations, drops, early scheduling |

### Grader Score

The final grader score (0.0-1.0) compares agent performance against:
1. **Baseline**: Regional average carbon/cost over job SLA windows
2. **Theoretical Minimum**: Best possible outcome with perfect foresight

```
grader_score = 0.6 * carbon_score + 0.3 * cost_score + 0.1 * sla_score
```

## Baseline Agents

| Agent | Strategy | Use Case |
|-------|----------|----------|
| `RandomAgent` | Random decisions | Lower bound |
| `GreedyCostAgent` | Cheapest region first | Cost optimization |
| `CarbonAwareGreedyAgent` | Lowest carbon first | Carbon optimization |
| `OracleAgent` | Uses forecast optimally | Upper bound reference |
| `HybridAgent` | Balanced multi-objective | Realistic baseline |

### Baseline Scores

Scores from running `python eval.py --all-baselines --scenarios easy,medium,hard`:

| Agent | Task | Grader Score | Total Reward | SLA Violations |
|-------|------|:------------:|:------------:|:--------------:|
| RandomAgent | easy | 0.50 | 3.91 | 0 |
| RandomAgent | medium | 0.38 | -8.55 | 7 |
| RandomAgent | hard | 0.35 | -3.60 | 45 |
| CarbonAwareGreedy | easy | 0.50 | -5.00 | 0 |
| CarbonAwareGreedy | medium | 0.38 | 19.35 | 39 |
| OracleAgent | easy | 0.50 | -1.12 | 0 |
| OracleAgent | hard | 0.37 | 156.79 | 198 |
| HybridAgent | easy | 0.50 | -1.08 | 0 |
| HybridAgent | medium | 0.38 | 19.35 | 39 |

The hard scenario is genuinely challenging even for oracle-level agents due to always-on constraints, burst capacity, and tight SLA windows across 3 regions.

## Carbon Data Providers

### Static Provider (Default)

Uses pre-generated CSV timeseries data. Best for reproducible training and evaluation.

### WattTime Provider

Real-time carbon intensity from WattTime API.

```bash
export WATTTIME_USERNAME=your_username
export WATTTIME_PASSWORD=your_password
```

### Electricity Map Provider

Global carbon intensity from Electricity Map.

```bash
export ELECTRICITYMAP_API_KEY=your_api_key
```

## Curriculum Learning

The curriculum system progressively increases difficulty:

1. **Level 1**: Easy scenarios with delay-tolerant jobs only
2. **Level 2**: Easy scenarios with some time-sensitive jobs
3. **Level 3**: Medium scenarios with multi-region routing
4. **Level 4**: Medium scenarios with bursts
5. **Level 5**: Hard scenarios with always-on constraints

Advancement requires:
- Mastery threshold: 70% success rate
- Minimum episodes: 10 per level

## Metrics & Observability

Prometheus metrics are exposed at `/metrics`:

- `ko2cube_carbon_savings_total`: Cumulative CO2 saved (gCO2)
- `ko2cube_episode_duration_seconds`: Episode timing histogram
- `ko2cube_jobs_scheduled_total`: Jobs scheduled by type/region
- `ko2cube_sla_violations_total`: SLA violations by type
- `ko2cube_grader_score`: Current grader score gauge

## Training with GRPO

The training script uses Group Relative Policy Optimization:

```bash
python train.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --episodes 500 \
    --curriculum \
    --batch-size 4 \
    --lr 1e-5 \
    --lora-r 16 \
    --output-dir checkpoints
```

**Training Flow:**
1. Generate observation prompts from environment
2. Sample completions from LLM
3. Parse actions and execute in environment
4. Use grader score as reward signal
5. Update policy using GRPO

## Docker Deployment

```bash
# Build image
docker build -t ko2cube-env .

# Run container
docker run -p 8000:8000 ko2cube-env

# With real-time carbon data
docker run -p 8000:8000 \
    -e WATTTIME_USERNAME=user \
    -e WATTTIME_PASSWORD=pass \
    ko2cube-env
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment for new episode |
| `/step` | POST | Execute action and get next observation |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List available tasks/scenarios |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

## Running the Inference Script

The inference script uses the OpenAI client to run an LLM agent through all 3 tasks and emits mandatory `[START]`/`[STEP]`/`[END]` logs.

### Required Environment Variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"   # LLM API endpoint
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # Model identifier
export HF_TOKEN="your-hf-token"                           # HuggingFace API key
export IMAGE_NAME="ko2cube:latest"                        # Docker image name
```

### Running

```bash
# Build the Docker image first
docker build -t ko2cube:latest .

# Run inference (all 3 tasks)
python inference.py
```

### Output Format

```
[START] task=easy env=ko2cube model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=schedule(etl_sales_000->us-east-1) reward=1.50 done=false error=null
[STEP] step=2 action=defer(ml_training_001->step3) reward=0.80 done=false error=null
...
[END] success=true steps=12 score=0.72 rewards=1.50,0.80,...
```

## Pre-Submission Checklist

- [x] HF Space deploys and responds to `reset()`
- [x] `openenv validate` passes
- [x] `docker build && docker run` works
- [x] `inference.py` runs and produces scores for all 3 tasks
- [x] 3 tasks with graders returning scores in [0.0, 1.0]
- [x] Typed Pydantic models for Observation, Action, State
- [x] `step()` returns observation, reward, done
- [x] `reset()` returns initial observation
- [x] `state()` returns current state
- [x] 317+ passing tests
- [x] README with setup, usage, and baseline scores

## Contributing

1. Run tests before submitting PRs: `pytest tests/ -v`
2. Follow existing code style
3. Add tests for new features
4. Update documentation as needed

## License

MIT License - See LICENSE file for details.
