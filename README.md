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

# 🌍 Ko2cube — Carbon-Aware Cloud Scheduler

An **OpenEnv-compliant** reinforcement learning environment that simulates a cloud infrastructure engineering desk. AI agents act as Infrastructure Schedulers who must move compute workloads across global regions to minimize carbon footprint and cost without breaching service level agreements (SLAs).

Built for the **OpenEnv Round 1**.

🔗 **Live Environment:** [Hugging Face Space](https://huggingface.co/spaces/NidheeshaT/ko2cube)

---

## ⚡ How It Works

The agent manages a continuous **job queue** and global **cloud infrastructure**. For every job in the queue, the agent must make a scheduling decision based on real-time grid carbon intensity and spot instance availability.

### The Challenge
Agents must balance three competing objectives:

| Objective | Description | Weight |
|-----------|-------------|--------|
| **SLA Compliance** | Complete jobs before their deadline. | 50% |
| **Carbon Intensity** | Shift loads to regions/hours with high renewable energy mix. | 35% |
| **Cost Efficiency** | Utilize Spot instances and cheap regions effectively. | 15% |

---

## 📋 Tasks (3 Core Scenarios)

| Task | ID | Difficulty | Workload Type | Key Optimization |
|------|----|------------|---------------|------------------|
| **Task 1** | `easy` | Easy | Single Region, Delay-Tolerant | **Time Shifting:** Defer non-urgent work to solar peaks. |
| **Task 2** | `medium` | Medium | Multi-Region, Mixed Workload | **Spatial Shifting:** Move work to the cleanest global grid. |
| **Task 3** | `hard` | Hard | Burst Capacity & Always-On | **Triage:** Prioritize critical APIs under infrastructure pressure. |

---

## 📊 Observation Space

The agent receives a `Ko2cubeObservation` containing the full system state:

| Field | Type | Description |
|-------|------|-------------|
| `current_step` | integer | Current simulation hour (0-168) |
| `job_queue` | List[Job] | Waiting jobs with CPU/Mem requirements and SLA window |
| `active_jobs` | List[RunningJob] | Jobs currently executing in different regions |
| `regions` | Dict[str, RegionInfo] | Carbon intensity (Current + 24h Forecast) and Instance inventory |

---

## 🛠️ Action Space

The agent submits a list of `JobAssignment` decisions:

| Action | Parameters | Description |
|--------|------------|-------------|
| `schedule`| `region`, `instance_type`, `machine_type` | Start the job immediately on a specific instance. |
| `defer` | `defer_to_step` | Wait for a cleaner/cheaper window in the future. |
| `drop`  | — | Cancel the job permanently (last resort). |

---

## 🏆 Scoring & Rewards

Scoring is deterministic and strictly clamped to **(0.01, 0.99)** per OpenEnv validator requirements.

*   **SLA Score**: Percentage of jobs completed within their deadline.
*   **Carbon Score**: Normalized improvement over a "Market Average" baseline using SLA-window integration.
*   **Cost Score**: Savings achieved relative to On-Demand list prices.

---

## 🏗️ Project Structure

```bash
ko2cube/
├── server/
│   ├── app.py            # FastAPI server with Gradio UI
│   ├── environment.py    # Core Ko2cubeEnvironment simulation logic
│   ├── rewards.py        # Clamped Grader and Reward computation
│   └── data/             # Synthetic Carbon & Infrastructure datasets
├── models.py             # Pydantic models for Action/Observation/State
├── inference.py          # Baseline LLM Agent with robust connection logic
├── openenv.yaml          # Environment metadata & task definitions
├── Dockerfile            # Containerized deployment spec
└── README.md             # This document
```

---

## 🚀 Getting Started

### 1. Run with Docker (Recommended)
```bash
docker build -t ko2cube-env .
docker run -p 8000:8000 ko2cube-env
```

### 2. Run Local Development
```bash
# Install dependencies
pip install -e .

# Start server
PYTHONPATH="." python -m server.app
```

---

## 🤖 Running Inference

The baseline agent uses **Structured CoT Prompting** to make scheduling decisions.

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token"

python inference.py
```

---

## ⚙️ CI/CD & Automation

This repository includes pre-configured **GitHub Actions** located in `.github/workflows/`:

**Validate OpenEnv**: Automatically builds your Docker image and runs environment unit tests on every push.