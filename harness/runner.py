"""Harness: start the Comms Agent (sourcecode serve) and submit tasks to it.

Starts the agent as a subprocess, waits for it to become ready, submits a
handful of prompts to POST /v1/agent/runs, and records pass/fail the same
way the original AgentSwitch harness did.

Required environment:
  S17_CONTROL_TOKEN   must match the token the agent is configured with
Optional environment:
  S17_PORT            defaults to 8113
  S17_HOST            defaults to 127.0.0.1
  INSTANCE            label used in result ids, defaults to "local"

The agent picks its own chat transport at startup: OpenAI directly if
OPENAI_API_KEY is set in .env, otherwise the GLC gateway (GLC_BASE_URL), which
must then already be running separately. .env in the repo root carries the
rest of the agent's configuration either way.
"""
import json
import os
import subprocess
import sys
import time
from dotenv import load_dotenv
import requests
load_dotenv()

HOST = os.environ.get("S17_HOST", "127.0.0.1")
PORT = os.environ.get("S17_PORT", "8113")
BASE = f"http://{HOST}:{PORT}"
CONTROL_TOKEN = os.environ["S17_CONTROL_TOKEN"]
AUTH = {"Authorization": "Bearer " + CONTROL_TOKEN}
INSTANCE = os.environ.get("INSTANCE", "local")
STARTUP_TIMEOUT_S = 60

# Prompts submitted to the running agent, one per task.
TASK_PROMPTS = [
    ("list_campaigns", "List all subscribers with status unsubscribed."),
    ("list_audiences", "List all campaigns"),
    ("list_subscribers", "List all subscribers"),
]

tasks = []


def record(task_id, title, passed, evidence):
    tasks.append({"id": f"{INSTANCE}:{task_id}", "title": title,
                  "passed": bool(passed), "score": 1.0 if passed else 0.0,
                  "evidence": str(evidence)[:500]})
    print(f"[{'PASS' if passed else 'FAIL'}] {task_id}: {str(evidence)[:200]}", flush=True)


def start_agent():
    """Launch `sourcecode serve` as a subprocess and return the Popen handle."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "sourcecode.cli", "serve", "--host", HOST, "--port", PORT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return proc


def wait_until_ready(proc, timeout_s=STARTUP_TIMEOUT_S):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"agent process exited early (code={proc.returncode})")
        try:
            r = requests.get(f"{BASE}/healthz", timeout=2)
            if r.ok:
                body = r.json()
                print(f"[info] gateway_provider={body.get('gateway_provider')} "
                      f"gateway_base_url={body.get('gateway_base_url')}", flush=True)
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def submit_run(prompt):
    body = {"tenant_id": "demo", "project_id": "campaigns", "user_id": "harness", "prompt": prompt}
    r = requests.post(f"{BASE}/v1/agent/runs", headers=AUTH, json=body, timeout=120)
    r.raise_for_status()
    return r.json()


agent_proc = start_agent()
try:
    # 1. The agent process comes up and reports healthy.
    ready = False
    try:
        ready = wait_until_ready(agent_proc)
        record("agent_startup", "Comms Agent starts and reports healthy", ready, f"ready={ready}")
    except Exception as exc:  # noqa: BLE001
        record("agent_startup", "Comms Agent starts and reports healthy", False, repr(exc))

    # 2. Submit each task prompt as a run and check it completes.
    if ready:
        for task_id, prompt in TASK_PROMPTS:
            try:
                result = submit_run(prompt)
                status = result.get("status")
                answer = result.get("answer") or {}
                ok = status == "completed"
                record(task_id, f"Run completes for prompt: {prompt!r}", ok,
                       f"status={status} answer={str(answer)[:200]!r}")
            except Exception as exc:  # noqa: BLE001
                record(task_id, f"Run completes for prompt: {prompt!r}", False, repr(exc))
    else:
        for task_id, prompt in TASK_PROMPTS:
            record(task_id, f"Run completes for prompt: {prompt!r}", False, "skipped: agent not ready")
finally:
    agent_proc.terminate()
    try:
        agent_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        agent_proc.kill()

with open("results.json", "w") as fh:
    json.dump({"tasks": tasks,
               "summary": f"{INSTANCE}: {sum(t['passed'] for t in tasks)}/{len(tasks)} checks passed"}, fh, indent=2)
