# Ko2cube Environment

Ko2cube is a carbon-aware cloud job scheduling environment designed for OpenEnv. It challenges AI agents to act as Infrastructure Engineers who must optimize the placement of diverse compute workloads across multiple cloud regions.

The goal is to maximize **SLA compliance** while minimizing **Carbon Intensity** and **Infrastructure Cost**.

## Key Features

- **Realistic Infrastructure**: Simulates real AWS EC2 instance types (m5, c5, r5) with CPU/Memory constraints and Spot/On-Demand pricing.
- **Dynamic Carbon Data**: Uses a 12-week high-fidelity synthetic timeseries featuring daily solar/wind cycles and weekly demand patterns.
- **Complexity Progression**: 3 distinct tasks ranging from single-region batch deferral to multi-region triage of high-burst workloads.
- **Programmatic Grader**: A mathematically robust scoring system (0.0–1.0) that compares agent performance against both a local "Market Average" and a theoretical "Global Optimum."


## Action & Observation Space

### Observation Space
The agent receives a `Ko2cubeObservation` containing:
- **`job_queue`**: List of jobs waiting to be scheduled (includes CPU/Mem requirements, SLA window, and `delay_tolerant` flag).
- **`active_jobs`**: List of jobs currently running and their remaining steps.
- **`regions`**: Dictionary mapping global regions (us-east-1, eu-west-1, etc.) to their **current carbon intensity** and a **24-hour forecast**.
- **`available_instances`**: Per-region inventory of instance types with current **Spot Pricing multipliers**.

### Action Space
The agent submits a `Ko2cubeAction` containing a list of `assignments`:
- **`schedule`**: Run the job immediately. Requires specifying a `region`, `instance_type`, and `machine_type` (Spot or On-Demand).
- **`defer`**: Target a specific future step for re-evaluation. Used for "shifting" loads to cleaner hours.
- **`drop`**: Permanently cancel a job (used as a last resort for non-viable workloads).

## Task Scenarios

1.  **Task 1 (Easy)**: Single region, delay-tolerant batch jobs only. The agent must learn to defer work away from nightly fossil-fuel peaks to midday solar peaks.
2.  **Task 2 (Medium)**: Three regions. Introduces non-deferrable CI/CD jobs. The agent must correctly triage "Always-Immediately" vs "Delay-Tolerant" workloads and move heavy loads to the cleanest available region.
3.  **Task 3 (Hard)**: Full burst capacity. Includes "Always-On" API serving workloads that must never be deferred or run on Spot. A massive mid-episode burst tests triage and right-sizing under pressure.

## Reward System

The reward function is multi-component:
- **SLA Compliance**: High points for finishing jobs within their window. Heaving penalties for breaches or dropping "Always-On" tasks.
- **Carbon Efficiency**: Proportional bonuses for beating the SLA-window market average.
- **Cost Efficiency**: Proportional rewards for utilizing Spot instances and avoiding expensive regions.
- **Right-Sizing Penalty**: An explicit penalty for "Over-provisioning" (selecting an instance much larger/more expensive than the job actually needs).

## Setup & Usage

### 1. Local Development
```bash
# Install dependencies
uv sync

# Run the simulator server
python -m server.app
```

### 2. Validation
To verify OpenEnv spec compliance:
```bash
openenv validate
```

### 3. Docker
```bash
docker build -t ko2cube-env .
docker run -p 8000:8000 ko2cube-env
```