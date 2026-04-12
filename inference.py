"""
Inference Script Example
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
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
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
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import re
import traceback
import asyncio
import os
import textwrap
import sys
from typing import List, Optional

from openai import AsyncOpenAI
from ko2cube.client import Ko2cubeEnv
from ko2cube.models import Ko2cubeAction, Ko2cubeObservation

from dotenv import load_dotenv
load_dotenv()


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B:together")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "ko2cube")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "ko2cube")
MAX_STEPS = 4
TEMPERATURE = 0.7
MAX_TOKENS = 1500
SUCCESS_SCORE_THRESHOLD = 0.1
MAX_TOTAL_REWARD = 60.0 # Ceiling for normalization in [END] line
TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Carbon-Aware Cloud Scheduler (Ko2cube).
    Your goal is to maximize SLA compliance while minimizing carbon footprints and cost.
    
    You will be provided with:
    1. A Current Job Queue (jobs waiting to be scheduled).
    2. Regional Data (carbon intensity forecasts and instance pricing/availability).
    
    Capabilities:
    - Job Assignments: Decide for EVERY job in the queue (schedule, defer, or drop).
    - Infrastructure Management: You can explicitly CREATE or DELETE Kubernetes resources (Nodes/Pods) in any region to manage capacity.
    
    Strategy:
    - Prioritize jobs near their SLA deadline.
    - Place compute-heavy jobs in regions with low solar/wind carbon intensity.
    - Use 'spot' instances for savings, but prefer 'on-demand' for critical always-on jobs.
    - Provision nodes in a region before scheduling jobs there if capacity is tight.
    
    REFERENCE EXAMPLES for resources_to_create and resources_to_delete items:
    - Create Node: {"kind": "Node", "metadata": {"name": "n1", "labels": {"node.kubernetes.io/instance-type": "m5.large"}}}
    - Create Pod:  {"kind": "Pod", "metadata": {"name": "p1"}, "spec": {"containers": [{"name": "c1", "image": "nginx", "resources": {"requests": {"cpu": "1", "memory": "1Gi"}}}]}}
    - Delete Node: {"kind": "Node", "name": "n1"}
    - Delete Pod:  {"kind": "Pod", "name": "p1"}
    
    RESPONSE FORMAT:
    You must respond with a raw JSON object matching this structure:
    {
      "assignments": [
        {"job_id": "job_1", "decision": "schedule", "region": "us-east-1", "instance_type": "m5.large", "machine_type": "spot"},
        {"job_id": "job_2", "decision": "defer", "defer_to_step": 5}
      ],
      "resources_to_create": {
        "us-east-1": [
          {
            "kind": "Node",
            "metadata": {"name": "node-1", "labels": {"node.kubernetes.io/instance-type": "m5.xlarge"}}
          }
        ]
      },
      "resources_to_delete": {
        "us-east-1": []
      }
    }
    
    Respond ONLY with the raw JSON object. No markdown, no reasoning text.
    """
).strip()


def format_start(task: str, env: str, model: str) -> str:
    return f"[START] task={task} env={env} model={model}"


def format_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> str:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Compact action string for the strictly required log format
    action_log = action.replace("\n", "").replace(" ", "")[:250]
    return f"[STEP] step={step} action={action_log} reward={reward:.2f} done={done_val} error={error_val}"


def format_end(success: bool, steps: int, score: float, rewards: List[float]) -> str:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    return f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}"


def build_user_prompt(step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    """Formats the current environment state into a structured dashboard for the LLM."""
    queue = obs.get("job_queue", [])
    regions = obs.get("regions", {})
    
    job_table = "| ID | CPU | RAM | SLA End | Tolerant | Pref |\n|---|---|---|---|---|---|\n"
    for j in queue:
        sla_end = "ALWAYS" if j['sla_end'] == -1 else j['sla_end']
        job_table += f"| {j['job_id']} | {j['cpu_cores']} | {j['memory_gb']}G | {sla_end} | {j['delay_tolerant']} | {j['instance_preference']} |\n"

    region_summary = ""
    for rname, rinfo in regions.items():
        carbon = rinfo['carbon']['current_intensity']
        cheapest_spot = min([i['spot_price'] for i in rinfo['available_instances']])
        region_summary += f"- {rname}: Carbon={carbon:.1f} gCO2/kWh, Min Spot=${cheapest_spot:.3f}\n"

    history_block = "\n".join(history[-3:]) if history else "None"

    return textwrap.dedent(
        f"""
        ## Ko2cube Dashboard - Step {step}
        
        ### Current Job Queue
        {job_table}
        
        ### Regional Status (Carbon & Price)
        {region_summary}
        
        ### History (Last 3 Steps)
        {history_block}
        
        Last Reward: {last_reward:.2f}
        
        **Action Required:** Assign decisions for all queued jobs. Respond in JSON.
        """
    ).strip()


def parse_llm_response(text: str) -> str:
    """Robustly extracts JSON from LLM response text."""
    text = text.strip()
    # Remove markdown blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # Try to find JSON braces if there's surrounding text
    matches = re.findall(r'(\{.*\})', text, re.DOTALL)
    if matches:
        # Take the most likely assignment JSON (usually the last or largest)
        text = max(matches, key=len)
        
    return text.strip()


async def get_model_message(client: AsyncOpenAI, step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    for i in range(3):
        try:
            completion = await client.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1, # Conservative for scheduling
                max_tokens=MAX_TOKENS,
                reasoning_effort="none",
                response_format=Ko2cubeAction,
                timeout=30
            )
            
            # Validation
            action_obj = completion.choices[0].message.parsed
            if not action_obj:
                raise ValueError("LLM failed to return a valid Ko2cubeAction object.")
            return action_obj
        except Exception as e:
            print(f"  [DEBUG] LLM parsing/API retry {i+1}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            
    return Ko2cubeAction(assignments=[])


async def _connect_env(image_name: str) -> Ko2cubeEnv:
    """Create and connect to the Ko2cube environment."""
    if not image_name or image_name == "":
        env = Ko2cubeEnv(base_url="http://localhost:8000")
        await env.connect()
    else:
        env = await Ko2cubeEnv.from_docker_image(image=image_name)
    return env


async def run_episode(client: AsyncOpenAI, base_url: str, task_id: str) -> List[str]:
    env = Ko2cubeEnv(base_url=base_url)
    episode_logs: List[str] = []
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    episode_logs.append(format_start(task=task_id, env=BENCHMARK, model=MODEL_NAME))

    try:
        # Connect to the shared environment server
        await env.connect()
        result = await env.reset(task_id=task_id)

        obs_dict = result.observation.model_dump()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_obj = await get_model_message(client, step, obs_dict, last_reward, history)

            try:
                result = await env.step(action_obj)
            except Exception as step_err:
                print(f"  [ERROR] env.step failed: {step_err}", file=sys.stderr)
                raise

            obs_obj = result.observation
            obs_dict = obs_obj.model_dump()
            reward = result.reward or 0.0

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            # Extract any error message from last_action_result if it implies failure
            step_error = None
            if "failed" in obs_obj.last_action_result.lower() or "error" in obs_obj.last_action_result.lower():
                step_error = obs_obj.last_action_result

            episode_logs.append(format_step(step=step, action=action_obj.model_dump_json(), reward=reward, done=result.done, error=step_error))
            history.append(f"Step {step}: {len(action_obj.assignments)} actions -> reward {reward:+.2f}")

            if result.done:
                break

        # Calculate final metrics
        total_reward = sum(rewards)
        # Normalize score and clamp strictly within (0.01, 0.99) per validator requirements
        score = max(0.01, min(0.99, total_reward / MAX_TOTAL_REWARD))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"  [ERROR] Episode failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        if env:
            try:
                await env.close()
            except Exception:
                pass
        episode_logs.append(format_end(success=success, steps=steps_taken, score=score, rewards=rewards))

    return episode_logs


async def main() -> None:
    """Main entry point running tasks in parallel using a shared container."""
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=30.0)

    print(f"🚀 Starting Ko2cube Inference Baseline (Parallel Shared)", flush=True, file=sys.stderr)
    print(f"📡 API: {API_BASE_URL} | Model: {MODEL_NAME}", flush=True, file=sys.stderr)

    shared_env = None
    try:
        shared_env = await _connect_env(LOCAL_IMAGE_NAME)
        # Retrieve the URL from the shared container (convert ws://... back to http://...)
        base_url = shared_env._ws_url.replace("ws://", "http://").replace("/ws", "")
        print(f"🌍 Shared environment live at {base_url}", flush=True, file=sys.stderr)

        # task_results = await asyncio.gather(*(run_episode(client, base_url, task) for task in TASKS))
        # Run tasks sequentially against the shared server
        task_results = []
        for task in TASKS:
            result = await run_episode(client, base_url, task)
            task_results.append(result)
        
        # Finally, print logs in order
        for episode_logs in task_results:
            for log_line in episode_logs:
                print(log_line, flush=True)
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Shutting down...", flush=True, file=sys.stderr)
    finally:
        if shared_env:
            print(f"🛑 Cleaning up shared environment and killing container...", flush=True, file=sys.stderr)
            try:
                await shared_env.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())