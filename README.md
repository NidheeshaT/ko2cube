---
title: Ko2cube - Carbon-Aware Cloud Scheduler
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Ko2cube: Carbon-Aware Cloud Job Scheduler

**Ko2cube** is a reinforcement learning environment where an AI agent acts as a cloud infrastructure manager. Its job is simple to state but difficult to solve: run the right workloads, in the right regions, at the right time - using as little carbon and money as possible, without missing deadlines.

Live environment: [Hugging Face Space](https://huggingface.co/spaces/nagarajpandith/ko2cube)

## The Problem

Cloud computing accounts for roughly 1–2% of global electricity consumption, but data centres do not have to run at full carbon cost all day. The grid is cleaner at certain hours (when solar peaks, for instance) and dirtier at others (evening fossil-fuel demand). A job that runs at noon in a solar-heavy region produces a fraction of the CO2 of the same job running at midnight on coal.

Most schedulers today are unaware of this. They schedule work according to priority and availability alone, leaving significant carbon savings on the table.

Ko2cube asks: **can an AI agent learn to be a smarter scheduler?**

## What the Agent Controls

The agent receives a continuous queue of compute jobs — batch ETL pipelines, CI/CD builds, ML training runs, video transcoding, database backups, and live API serving workloads — each with its own resource requirements, time window, and urgency level.

At every simulation step, the agent decides what to do with each job in the queue:

| Decision   | Meaning                                                            |
| ---------- | ------------------------------------------------------------------ |
| `schedule` | Run the job now on a chosen region and instance type               |
| `defer`    | Wait for a better carbon or cost window, within the job's deadline |
| `drop`     | Cancel the job permanently (SLA violation — costly)                |

The agent can see real-time carbon intensity and a 24-hour forecast for each region, along with current spot pricing and instance inventory.

## Three Scenarios, Escalating Difficulty

| Task                                                     | What the agent must learn                                                                                                                                                                |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Easy** — single region, delay-tolerant jobs            | Shift work to low-carbon hours. Understand the grid cycle.                                                                                                                               |
| **Medium** — three global regions, mixed urgency         | Triage urgent CI/CD jobs immediately. Spatially shift batch loads to whichever region is currently greenest.                                                                             |
| **Hard** — all 8 workload types, mid-episode CI/CD burst | Protect always-on API serving (never defer, never use spot). Manage GPU contention between ML training and video transcoding. Recover queue after a sudden 5-step burst of CI/CD builds. |

## Scoring

The final score is a weighted combination of three independent components, each normalized to the range **(0.01, 0.99)**:

| Component         | Weight | How it's measured                                                                |
| ----------------- | ------ | -------------------------------------------------------------------------------- |
| SLA Compliance    | 50%    | Jobs completed within their deadline vs. total jobs                              |
| Carbon Efficiency | 35%    | Actual carbon emitted vs. theoretical minimum achievable given the grid forecast |
| Cost Efficiency   | 15%    | Actual spend vs. market-average cost for the same workload                       |

Scoring is fully deterministic and programmatic. There is no ground truth label — the environment measures what the agent actually did against what it could have done.

## Reward Design

The per-step reward is dense and interpretable. Every decision the agent makes produces an immediate signal:

- Scheduling a job within its SLA window gives a positive bonus. Scheduling it late gives a penalty equal to the SLA breach constant.
- A smart deferral (carbon is currently above the job's SLA-window average) is rewarded. An unnecessary deferral (carbon is already clean) is lightly penalized.
- Deferring or dropping an always-on job triggers the largest penalty in the system — these workloads must never be interrupted.
- Over-provisioning (picking a larger instance than the job requires) triggers a waste penalty proportional to the dollar difference.
- Potential-based shaping rewards the agent for completing jobs on time, not just completing them.

At episode end, a terminal reward layer looks at overall completion rate, the agent's cumulative carbon versus baseline, and cost versus baseline.

## What the Agent Controls

Ko2cube provides a "Full-Stack" control plane. The agent doesn't just manage a job queue; it actively orchestrates the underlying Kubernetes infrastructure using a built-in **KWOK** (Kubernetes Without Kubelet) adapter.

### 1. Workload Scheduling

At every simulation step, the agent makes decisions for jobs in the queue:

- `schedule`: Deploy the workload immediately.
- `defer`: Wait for a cleaner/cheaper carbon window.
- `drop`: Cancel the job permanently (SLA penalty).

### 2. Infrastructure Orchestration (New)

The agent has direct control over the cluster topology across all regions:

- `resources_to_create`: Dynamically provision new **K8s Nodes** or **Pods** to handle bursts.
- `resources_to_delete`: Tear down idle or inefficient infrastructure to save on "Always-On" carbon and costs.

The agent receives a unified observation including a 24-hour carbon forecast, spot pricing, and the current **K8s Cluster State** (Nodes/Pods) for every regional data center.

---

## What Makes It Technically Interesting

**Grid-aware baselines.** The reward does not compare agents to a fixed number. It computes a personal baseline per job — the average carbon intensity and average spot price across the job's entire allowed SLA window, across all available regions. This means an agent that happens to get a "lucky" clean grid still has to beat its own relevant baseline to earn carbon rewards.

**Eight distinct workload archetypes.** Each job type is defined with realistic resource profiles: ETL pipelines (4 vCPU, 16 GB, 45 min runs), ML training (GPU-heavy, 4-hour duration), CI/CD builds (8 vCPU, 12-minute deadline, cannot wait), API serving (always-on, on-demand only), and more. The hard scenario runs all eight simultaneously.

**Expiry penalty.** When a job's last eligible step passes and the agent said nothing about it, an immediate SLA breach penalty fires at that step rather than being buried in the terminal reward. This gives the LLM agent a dense, well-timed training signal.

---

## Observation Space

```
Ko2cubeObservation
├── current_step          int          — simulation hour index
├── job_queue             List[Job]    — jobs awaiting a decision
│   ├── job_id, eta_minutes, cpu_cores, memory_gb
│   ├── sla_start, sla_end           — the allowed scheduling window
│   ├── delay_tolerant               — whether deferral is safe
│   └── baseline_carbon_intensity    — personal SLA-window carbon average
├── active_jobs           List[RunningJob]
│   └── job_id, region, steps_remaining, machine_type
└── regions               Dict[str, RegionInfo]
    ├── carbon.current_intensity     — gCO2/kWh right now
    ├── carbon.forecast              — 24-step ahead forecast
    └── available_instances          — instance types with spot/on-demand prices
```

## Project Structure

```
ko2cube/
├── server/
│   ├── app.py              FastAPI server and Gradio web interface
│   ├── environment.py      Core simulation engine and KWOK lifecycle hooks
│   ├── rewards.py          Per-step and terminal reward computation
│   ├── kwok/               KWOK adapter — node/pod create, delete, and validation
│   └── data/
│       ├── scenarios.py              Three episode definitions
│       ├── infrastructure.json       AWS instance catalogue (m5, c5, r5 families)
│       └── cleaned_timeseries_data.csv  Hourly carbon intensity by region
├── models.py               Pydantic models for Action, Observation, and State
├── inference.py            Baseline LLM agent using async OpenAI-compatible API
├── client.py               OpenEnv client wrapper
├── openenv.yaml            Task registry and environment metadata
└── Dockerfile              Container spec for deployment
```

## Getting Started

**Run with Docker:**

```bash
docker build -t ko2cube .
docker run -p 8000:8000 ko2cube
```

**Run locally:**

```bash
./start.sh
source venv/bin/activate
python -m ko2cube.server.app
```

**Run the baseline agent:**

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen3.5-9B:together"
export HF_TOKEN="your_token_here"

python inference.py
```

## Future Scope

Our roadmap includes:

- **Real-time Infrastructure Sync**: Beyond synthetic data, we plan to integrate live APIs from **WattTime** (real-time grid carbon intensity) and live **Spot Instance** pricing feeds for production-grade decision making.
- **Data Gravity & Egress Optimization**: Incorporating the energy and financial cost of inter-region data movement (egress), ensuring that "green" scheduling isn't offset by high network carbon footprints.
- **Karpenter-Native Intelligence**: Building upon and extending **Karpenter** - the industry-leading Kubernetes autoscaler - to inject carbon-awareness into its native cost-optimization engine.
- **Reasoning-Based Evaluation**: Implementing **Reasoning Text Based Rewards**, where the environment evaluates the "Logic" behind an agent's decision using LLM-as-a-Grader to ensure decisions are justifiable, not just lucky.
- **Multi-Cloud Global Arbitrage**: Expanding the infrastructure substrate to include GCP and Azure regions, allowing the agent to arbitrage carbon and cost across the entire global cloud market.

## CI/CD

This repository includes two GitHub Actions workflows:

- **Validate OpenEnv** — runs on every push. Builds the Docker image and runs the environment unit tests.
- **Sync to Hub** — pushes the latest version of the environment to the Hugging Face Space automatically.
