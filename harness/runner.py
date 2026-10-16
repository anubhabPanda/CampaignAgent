"""Harness: start the Comms Agent (sourcecode serve) and submit tasks to it.

Starts the agent as a subprocess, waits for it to become ready, submits a
handful of prompts to POST /v1/agent/runs, and records pass/fail the same
way the original AgentSwitch harness did.

No environment variable is required to already exist for the runner itself:
the control-plane token is generated here and handed only to the agent
subprocess this run starts, so there is nothing to configure beforehand.

Optional environment (read from the real process environment; no .env file
is required or assumed - the official run has none):
  S17_PORT              defaults to 8113
  S17_HOST              defaults to 127.0.0.1
  INSTANCE              label used in result ids, defaults to "local"
  OPENAI_API_KEY        passed through to the agent subprocess if set
  OPENAI_BASE_URL       passed through to the agent subprocess if set
  OPENAI_MODEL          passed through to the agent subprocess if set
  AGENTSWITCH_BASE_URL  passed through to the agent subprocess if set
  AGENTSWITCH_TOKEN     passed through to the agent subprocess if set

The agent picks its own chat transport at startup: OpenAI directly if
OPENAI_API_KEY is set, otherwise the GLC gateway (GLC_BASE_URL), which must
then already be running separately. A local .env, if present, is loaded too,
but nothing here depends on one existing.
"""
import json
import os
import secrets
import subprocess
import sys
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HOST = os.environ.get("S17_HOST", "127.0.0.1")
PORT = os.environ.get("S17_PORT", "8113")
BASE = f"http://{HOST}:{PORT}"
# Generated here, not read from outside: the runner owns the agent process it
# starts, so it can mint the control-plane credential that process needs
# instead of requiring the caller to have configured one in advance.
CONTROL_TOKEN = secrets.token_urlsafe(32)
AUTH = {"Authorization": "Bearer " + CONTROL_TOKEN}
INSTANCE = os.environ.get("INSTANCE", "local")
STARTUP_TIMEOUT_S = 60

# Variables the agent subprocess needs for its LLM/AgentSwitch transport.
# Passed through explicitly (on top of the inherited environment) so the
# agent's configuration doesn't silently depend on a .env file existing.
PASSTHROUGH_ENV_VARS = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
    "AGENTSWITCH_BASE_URL", "AGENTSWITCH_TOKEN", "GLC_BASE_URL",
)

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
    """Launch `sourcecode serve` as a subprocess and return the Popen handle.

    The child gets the control token this run minted plus whatever LLM/
    AgentSwitch variables are present in the real environment, so the agent
    never needs a pre-existing S17_CONTROL_TOKEN or a .env file to start.
    """
    env = os.environ.copy()
    env["S17_CONTROL_TOKEN"] = CONTROL_TOKEN
    for name in PASSTHROUGH_ENV_VARS:
        value = os.environ.get(name)
        if value:
            env[name] = value
    proc = subprocess.Popen(
        [sys.executable, "-m", "sourcecode.cli", "serve", "--host", HOST, "--port", PORT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
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


# Every exit path below writes results.json, including a crash before the
# agent process could even be started: "no results" must never be the outcome
# of a run, only "a task failed with this evidence" can be.
agent_proc = None
try:
    agent_proc = start_agent()
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
except Exception as exc:  # noqa: BLE001
    # Anything unexpected (e.g. start_agent() itself failing to spawn) still
    # produces a result file describing the failure rather than none at all.
    record("agent_startup", "Comms Agent starts and reports healthy", False, repr(exc))
    for task_id, prompt in TASK_PROMPTS:
        if not any(t["id"] == f"{INSTANCE}:{task_id}" for t in tasks):
            record(task_id, f"Run completes for prompt: {prompt!r}", False, "skipped: agent startup crashed")
finally:
    if agent_proc is not None:
        agent_proc.terminate()
        try:
            agent_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            agent_proc.kill()
    with open("results.json", "w") as fh:
        json.dump({"tasks": tasks,
                   "summary": f"{INSTANCE}: {sum(t['passed'] for t in tasks)}/{len(tasks)} checks passed"},
                   fh, indent=2)
