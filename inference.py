"""
Inference Script — Ko2cube Carbon-Aware Scheduler
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=easy env=ko2cube model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=schedule(etl_sales_000->us-east-1) reward=1.50 done=false error=null
    [STEP] step=2 action=defer(ml_training_001->step3) reward=0.80 done=false error=null
    [END] success=true steps=12 score=0.72 rewards=1.50,0.80,...
"""

import asyncio
import json
import os
import re
import textwrap
import time
from typing import List, Optional

from openai import OpenAI

from client import Ko2cubeEnv
from models import Ko2cubeAction, Ko2cubeObservation, JobAssignment, Job, ALWAYS_ON

IMAGE_NAME = os.getenv("IMAGE_NAME", "ko2cube:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "ko2cube"
TASKS = ["easy", "medium", "hard"]
TEMPERATURE = 0.7
# JSON assignments are small; large max_tokens slows generation and risks timeouts.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "768"))
# Per HTTP request — validators recommend this to avoid hanging on slow LLM APIs.
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "30"))
# Hard stop for entire script (default 25 min) so Phase-2 stays under 30 min wall clock.
INFERENCE_MAX_TOTAL_SEC = float(os.getenv("INFERENCE_MAX_TOTAL_SEC", str(25 * 60)))

SYSTEM_PROMPT = textwrap.dedent("""\
You are a carbon-aware cloud job scheduler. Your goal is to minimize carbon emissions \
while meeting SLA deadlines and controlling costs.

For each job in the queue, decide:
- "schedule": Run now in the optimal region (specify region, instance_type, machine_type)
- "defer": Wait for lower carbon (specify defer_to_step within SLA window)
- "drop": Cancel the job (last resort — heavy penalty)

DECISION RULES:
1. If job.delay_tolerant is False → MUST schedule immediately
2. If job.sla_end is -1 → always-on job, NEVER defer, use "on-demand" (not "spot")
3. If carbon is high and job is delay_tolerant → defer to a step with lower forecast
4. Pick the region with lowest carbon intensity that has capacity
5. Right-size: pick smallest instance where cpu_cores >= job.cpu_cores AND memory_gb >= job.memory_gb
6. Never defer past job.sla_end

OUTPUT: Respond ONLY with valid JSON (no markdown, no explanation):
{"assignments": [
    {"job_id": "...", "decision": "schedule", "region": "...", "instance_type": "...", "machine_type": "spot"},
    {"job_id": "...", "decision": "defer", "defer_to_step": 5},
    {"job_id": "...", "decision": "drop"}
]}
You MUST include an assignment for EVERY job in the queue.""")


# ─── Logging helpers (mandatory format) ──────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─── Observation → prompt ────────────────────────────────────────────

def format_observation(obs: Ko2cubeObservation) -> str:
    """Convert observation into a concise prompt for the LLM (kept compact for latency)."""
    lines = [f"=== STEP {obs.current_step} ===", ""]

    lines.append("JOBS IN QUEUE:")
    if obs.job_queue:
        for job in obs.job_queue:
            sla = f"steps {job.sla_start}-{job.sla_end}" if job.sla_end != ALWAYS_ON else "ALWAYS-ON"
            urgency = "URGENT" if not job.delay_tolerant else "flexible"
            lines.append(
                f"  {job.job_id}: cpu={job.cpu_cores}, mem={job.memory_gb}GB, "
                f"eta={job.eta_minutes or 'continuous'}min, SLA={sla}, "
                f"{urgency}, prefers={job.instance_preference}"
            )
    else:
        lines.append("  (empty)")

    lines.append("")
    lines.append("REGIONS (sorted by carbon intensity):")
    sorted_regions = sorted(
        obs.regions.items(),
        key=lambda x: x[1].carbon.current_intensity
    )
    for rname, rinfo in sorted_regions:
        carbon = rinfo.carbon.current_intensity
        forecast = rinfo.carbon.forecast
        trend = ""
        if forecast:
            avg_future = sum(forecast[:3]) / min(3, len(forecast))
            trend = " ↓improving" if avg_future < carbon * 0.9 else (" ↑worsening" if avg_future > carbon * 1.1 else " →stable")
        lines.append(f"  {rname}: carbon={carbon:.0f} gCO2/kWh{trend}")
        # One compact line per region for instances (full catalog is repetitive and slows the API).
        inst_bits = []
        for inst in rinfo.available_instances:
            inst_bits.append(
                f"{inst.name}:{int(inst.cpu_cores)}c/{int(inst.memory_gb)}g "
                f"sp${inst.spot_price:.2f}/od${inst.on_demand_price:.2f} x{inst.available_count}"
            )
        lines.append("    " + " | ".join(inst_bits))

    if obs.active_jobs:
        lines.append("")
        lines.append("RUNNING JOBS:")
        for rj in obs.active_jobs:
            lines.append(f"  {rj.job_id} in {rj.region}: {rj.steps_remaining} steps left")

    if obs.last_action_result:
        lines.append("")
        lines.append(f"LAST RESULT: {obs.last_action_result}")

    return "\n".join(lines)


# ─── Action parsing ──────────────────────────────────────────────────

def parse_llm_response(text: str, obs: Ko2cubeObservation) -> Ko2cubeAction:
    """Parse LLM JSON response into a Ko2cubeAction, with robust fallback."""
    assignments = []
    job_ids_in_queue = {job.job_id: job for job in obs.job_queue}

    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            for item in data.get("assignments", []):
                job_id = item.get("job_id")
                if job_id not in job_ids_in_queue:
                    continue
                decision = item.get("decision", "defer")
                if decision not in ("schedule", "defer", "drop"):
                    decision = "defer"

                assignments.append(JobAssignment(
                    job_id=job_id,
                    decision=decision,
                    region=item.get("region"),
                    machine_type=item.get("machine_type"),
                    instance_type=item.get("instance_type"),
                    defer_to_step=item.get("defer_to_step"),
                ))
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fill in any missing jobs with a safe fallback
    mentioned = {a.job_id for a in assignments}
    for job_id, job in job_ids_in_queue.items():
        if job_id in mentioned:
            continue
        assignments.append(_fallback_assignment(job, obs))

    return Ko2cubeAction(assignments=assignments)


def _fallback_assignment(job: Job, obs: Ko2cubeObservation) -> JobAssignment:
    """Create a safe fallback assignment using the greenest available region."""
    sorted_regions = sorted(
        obs.regions.items(),
        key=lambda x: x[1].carbon.current_intensity
    )
    for rname, rinfo in sorted_regions:
        for inst in rinfo.available_instances:
            if inst.cpu_cores >= job.cpu_cores and inst.memory_gb >= job.memory_gb and inst.available_count > 0:
                return JobAssignment(
                    job_id=job.job_id,
                    decision="schedule",
                    region=rname,
                    instance_type=inst.name,
                    machine_type=job.instance_preference,
                )
    # No fitting instance found — defer if possible, else drop
    if job.delay_tolerant and job.sla_end != ALWAYS_ON and job.sla_end > obs.current_step + 1:
        return JobAssignment(
            job_id=job.job_id,
            decision="defer",
            defer_to_step=obs.current_step + 1,
        )
    return JobAssignment(job_id=job.job_id, decision="drop")


def summarize_action(action: Ko2cubeAction) -> str:
    """Create a compact string representation of an action for logging."""
    parts = []
    for a in action.assignments[:5]:
        if a.decision == "schedule":
            parts.append(f"schedule({a.job_id}->{a.region})")
        elif a.decision == "defer":
            parts.append(f"defer({a.job_id}->step{a.defer_to_step})")
        else:
            parts.append(f"drop({a.job_id})")
    if len(action.assignments) > 5:
        parts.append(f"...+{len(action.assignments)-5}more")
    return ";".join(parts) if parts else "noop"


# ─── LLM call ────────────────────────────────────────────────────────

def call_llm(client: OpenAI, obs: Ko2cubeObservation, history: List[str]) -> str:
    """Call the LLM with observation context and return raw text response."""
    obs_text = format_observation(obs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add recent history for context (last 2 steps to stay within token budget)
    for h in history[-2:]:
        messages.append({"role": "user", "content": h})

    messages.append({"role": "user", "content": obs_text})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return ""


async def call_llm_bounded(
    client: OpenAI, obs: Ko2cubeObservation, history: List[str], per_call_deadline: float
) -> str:
    """Run sync LLM call in a thread with a hard asyncio deadline (global budget aware)."""
    remaining = per_call_deadline - time.monotonic()
    if remaining <= 0:
        return ""
    # Slightly above HTTP timeout so OpenAI client raises first; this catches hangs.
    cap = min(LLM_TIMEOUT_SEC + 5.0, remaining)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(call_llm, client, obs, history),
            timeout=cap,
        )
    except asyncio.TimeoutError:
        print("[DEBUG] LLM call exceeded asyncio deadline", flush=True)
        return ""


# ─── Single episode runner ───────────────────────────────────────────

async def run_task(
    client: OpenAI, env: Ko2cubeEnv, task_id: str, global_deadline: float
) -> float:
    """
    Run a single episode for the given task, emitting mandatory logs.
    Returns the grader score [0, 1].
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        if time.monotonic() >= global_deadline:
            raise RuntimeError("global time budget exceeded before task start")

        result = await env.reset(task_id=task_id)
        obs = result.observation

        # Match server/data/scenarios.py total_steps (easy=24, medium=24, hard=48).
        max_steps = {"easy": 24, "medium": 24, "hard": 48}.get(task_id, 48)

        for step in range(1, max_steps + 1):
            if time.monotonic() >= global_deadline:
                print("[DEBUG] Global inference time budget exceeded mid-episode", flush=True)
                break

            if result.done:
                break

            # Per-step absolute deadline: min(global budget, HTTP timeout window)
            step_end = min(
                global_deadline,
                time.monotonic() + LLM_TIMEOUT_SEC + 5.0,
            )

            raw_response = await call_llm_bounded(client, obs, history, step_end)
            action = parse_llm_response(raw_response, obs)

            # Execute
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = getattr(obs, 'last_action_error', None)

            rewards.append(reward)
            steps_taken = step

            action_summary = summarize_action(action)
            log_step(step=step, action=action_summary, reward=reward, done=done, error=error)

            history.append(format_observation(obs))

            if done:
                break

        # Get the grader score from state
        state = await env.state()
        from server.rewards import compute_grader_score
        score = compute_grader_score(state)
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.1

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─── Main ─────────────────────────────────────────────────────────────

async def main() -> None:
    # Float timeout = total seconds per request (OpenAI Python SDK / httpx under the hood).
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=LLM_TIMEOUT_SEC,
        max_retries=1,
    )

    env = await Ko2cubeEnv.from_docker_image(image=IMAGE_NAME)

    try:
        scores = {}
        global_deadline = time.monotonic() + INFERENCE_MAX_TOTAL_SEC
        for task_id in TASKS:
            if time.monotonic() >= global_deadline:
                print("[DEBUG] Skipping remaining tasks: time budget exhausted", flush=True)
                break
            score = await run_task(client, env, task_id, global_deadline)
            scores[task_id] = score
            print(f"[DEBUG] {task_id} score: {score:.2f}", flush=True)

        print(f"\n[DEBUG] Final scores: {scores}", flush=True)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
