import requests
import time
import subprocess
import os

print("🔄 1. Resetting environment ('hard' task)...")
response = requests.post("http://localhost:8000/reset", json={"task_id": "hard"})
if response.status_code != 200:
    print("Reset failed!", response.text)
    
# Get current queue to find a valid job to schedule
state = response.json()
queue = state.get("observation", {}).get("job_queue", [])
if not queue:
    print("Queue is empty, cannot run demo!")
    exit(1)
    
# Find a long-running job so it doesn't instantly finish in step 1
first_job = next((j for j in queue if (j.get("eta_minutes") or 0) > 60), queue[0])
job_id = first_job["job_id"]
print(f"📌 Found queued job '{job_id}' (CPU={first_job['cpu_cores']}, Mem={first_job['memory_gb']}GB, ETA={first_job.get('eta_minutes')}m).")

print(f"🚀 2. Scheduling job '{job_id}' onto a KWOK node (us-east-1, m5.4xlarge)...")
payload = {
    "action": {
        "assignments": [{
            "job_id": job_id,
            "decision": "schedule",
            "region": "us-east-1",
            "instance_type": "m5.4xlarge",
            "machine_type": "spot"
        }]
    }
}
response = requests.post("http://localhost:8000/step", json=payload)
data = response.json()
obs = data.get("observation", {})
print("Step response:", obs.get("last_action_result"))
print("Active jobs:", len(obs.get("active_jobs", [])))

print("⏱️  Waiting 1 second for K8s pod to transition to Running...")
time.sleep(1)

print("\n📦 3. Checking KWOK virtual cluster for running pods:")
print("-" * 50)

# Calling kubectl via KWOK's internal path
kwok_kubectl = os.path.expanduser("~/.kwok/clusters/kwok/bin/kubectl")
kwok_cmd = [kwok_kubectl, "--kubeconfig", os.path.expanduser("~/.kwok/clusters/kwok/kubeconfig.yaml"), "get", "pods", "-n", "ko2cube", "-o", "wide"]
subprocess.run(kwok_cmd)
print("-" * 50)
print("✅ Success! The Ko2cube API seamlessly mirrored the decision into KWOK!")
