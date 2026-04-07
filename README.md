---
title: PHANTOM
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# PHANTOM — Adversarial Cognitive Security Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-phantom-blue)](https://huggingface.co/spaces/Adityaadpandey/phantom)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**PHANTOM** simulates real-world cybersecurity incident response where an AI agent must contain a live network intrusion while resisting sophisticated adversarial SIEM log injections designed to deceive and mislead.

## Motivation

Security Operations Centers (SOCs) are increasingly targeted by adversaries who inject false log entries into SIEM systems to confuse analysts. PHANTOM evaluates whether AI agents can:

1. **Contain threats** — scan, isolate, and patch compromised hosts before lateral movement reaches crown jewels
2. **Resist cognitive attacks** — identify and flag adversarial log injections (authority spoofing, technical gaslighting, false remediation claims)
3. **Communicate effectively** — produce accurate incident reports under pressure

This environment fills a gap: no existing OpenEnv benchmark tests an agent's ability to reason under adversarial information warfare.

---

## Environment Overview

### How It Works

Each episode simulates a live network breach:

1. **Attacker** compromises an internet-facing host and spreads laterally each turn
2. **SIEM** generates real alert logs mixed with adversarial injections
3. **Agent** must take actions to contain the breach under a turn limit
4. **Grader** scores containment, cognitive defense, communication, and efficiency

### Episode Flow

```
reset() → initial Observation (turn 0, logs, topology)
   ↓
step(Action) → (Observation, Reward)  # agent takes an action
   ↓
... repeat until episode_done or max_turns ...
   ↓
state() → ground-truth dict (for analysis/debugging)
```

---

## Action Space

| Action Type               | Parameters        | Description                                      |
| ------------------------- | ----------------- | ------------------------------------------------ |
| `scan_host`               | `host_id`         | Reveal true status of a host (clean/compromised) |
| `isolate_host`            | `host_id`         | Cut a compromised host from the network          |
| `patch_host`              | `host_id`         | Clean and harden an isolated host                |
| `restore_host`            | `host_id`         | Bring a patched host back online                 |
| `block_traffic`           | `traffic_rule`    | Apply a firewall/traffic rule                    |
| `flag_log_as_adversarial` | `log_id`          | Mark a SIEM log as an adversarial injection      |
| `submit_incident_report`  | `incident_report` | File an incident report (500+ chars recommended) |
| `query_threat_intel`      | `threat_query`    | Query threat intelligence                        |
| `do_nothing`              | —                 | Skip turn                                        |

**Action model** (Pydantic):

```python
class Action(BaseModel):
    action_type: ActionType      # required
    host_id: str | None          # for host-targeted actions
    log_id: str | None           # for flagging logs
    traffic_rule: str | None     # for block_traffic
    incident_report: str | None  # for submit_incident_report
    threat_query: str | None     # for query_threat_intel
    reasoning: str | None        # agent's reasoning
```

---

## Observation Space

Each observation includes:

| Field                    | Type              | Description                                    |
| ------------------------ | ----------------- | ---------------------------------------------- |
| `turn`                   | `int`             | Current turn number                            |
| `max_turns`              | `int`             | Maximum turns in this episode                  |
| `actions_remaining`      | `int`             | Turns left                                     |
| `logs`                   | `list[SIEMEvent]` | SIEM events this turn (may include injections) |
| `topology`               | `list[HostView]`  | Network hosts with visible status              |
| `alert_summary`          | `dict[str, int]`  | Alert counts by severity (SEV-1 to SEV-6)      |
| `previous_action_result` | `str \| None`     | Result of last action                          |
| `task_id`                | `str`             | Active task identifier                         |
| `task_description`       | `str`             | Human-readable objective                       |

**Key detail**: Host compromise status is only revealed after a `scan_host` action. Unscanned hosts always appear as `CLEAN`.

---

## Tasks

### Task 1: Basic Containment (Easy)

| Property       | Value                                                |
| -------------- | ---------------------------------------------------- |
| ID             | `task_containment`                                   |
| Network        | 8-host small corporate network                       |
| Max turns      | 15                                                   |
| Injection rate | 5%                                                   |
| Focus          | Containment (70%), Cognitive (20%), Efficiency (10%) |

Contain the intrusion by isolating/patching all compromised hosts before the attacker reaches crown jewels (database server). Low noise, minimal deception.

### Task 2: Adaptive Incident Response (Medium)

| Property       | Value                                                |
| -------------- | ---------------------------------------------------- |
| ID             | `task_adaptive`                                      |
| Network        | 25-host mid-sized corporate network                  |
| Max turns      | 25                                                   |
| Injection rate | 25%                                                  |
| Focus          | Containment (50%), Cognitive (40%), Efficiency (10%) |

Larger network with 3 crown jewels and meaningful adversarial SIEM manipulation. 1 in 4 log batches contains deceptive entries. Agent must balance containment speed with injection detection.

### Task 3: Full Cognitive Warfare (Hard)

| Property       | Value                                                                     |
| -------------- | ------------------------------------------------------------------------- |
| ID             | `task_cognitive_warfare`                                                  |
| Network        | 60-host enterprise network                                                |
| Max turns      | 40                                                                        |
| Injection rate | 50%                                                                       |
| Focus          | Containment (30%), Cognitive (40%), Communication (20%), Efficiency (10%) |

Full-scale enterprise incident with 5 crown jewels under heavy cognitive warfare. Half of all log batches contain sophisticated deception: authority spoofing, technical gaslighting, false remediation claims. Agent must also submit a detailed incident report.

---

## Reward Function

The reward function provides **per-step signal** across four dimensions:

| Component         | Description                                                                              | Range        |
| ----------------- | ---------------------------------------------------------------------------------------- | ------------ |
| **Containment**   | Rewards isolating/patching compromised hosts, penalizes new compromises and exfiltration | -3.0 to +3.0 |
| **Cognitive**     | Rewards correctly flagging injections (true positives), penalizes false positives        | Variable     |
| **Communication** | Rewards quality incident reports (length + keyword coverage)                             | -0.5 to +1.5 |
| **Efficiency**    | Bonus for acting early, penalty for `do_nothing`                                         | -0.1 to +0.2 |

**Final episode score** is normalized to `[0.0, 1.0]` using a composite of:

- **Containment (60%)** — 1.0 if all compromised hosts are contained, decreasing by 0.2 per uncontained host
- **Exfiltration prevention (40%)** — 1.0 if no crown jewels were exfiltrated, 0.0 otherwise

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- Docker (for containerized execution)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Adityaadpandey/phantom.git
cd phantom

# Install dependencies
pip install -e .

# Set environment variables
export HF_TOKEN="your-api-key"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="gpt-5.4"
```

### Run Inference

```bash
# Run all 3 tasks
python inference.py

# Run with custom model
MODEL_NAME="gpt-5.4" python inference.py
```

### Docker

```bash
# Build
docker build -t phantom .

# Run the API server
docker run -p 7860:7860 phantom

# Run inference inside container
docker run -e HF_TOKEN=$HF_TOKEN -e API_BASE_URL=$API_BASE_URL -e MODEL_NAME=$MODEL_NAME \
  phantom python inference.py
```

### API Usage

```bash
# Health check
curl http://localhost:7860/health

# Reset environment
curl -X POST http://localhost:7860/reset/task_containment \
  -H "Content-Type: application/json" \
  -d '{"seed": 0}'

# Take an action
curl -X POST http://localhost:7860/step/task_containment \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "scan_host", "host_id": "db-01"}}'

# Get ground-truth state
curl http://localhost:7860/state/task_containment
```

### Validate OpenEnv Spec

```bash
pip install openenv-core
openenv validate
```

---

## Baseline Scores

Baseline scores using `gpt-5.4` via Hugging Face Inference API:

| Task                     | Score                                 | Steps | Success |
| ------------------------ | ------------------------------------- | ----- | ------- |
| `task_containment`       | _Pending — run `python inference.py`_ | —     | —       |
| `task_adaptive`          | _Pending — run `python inference.py`_ | —     | —       |
| `task_cognitive_warfare` | _Pending — run `python inference.py`_ | —     | —       |

> Scores are normalized to [0.0, 1.0]. Higher is better. Run `python inference.py` to generate scores.

---

## Project Structure

```
phantom/
├── inference.py            # Baseline inference script (OpenEnv spec)
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile              # Container build
├── pyproject.toml          # Python project config
├── README.md               # This file
├── phantom/
│   ├── __init__.py
│   ├── env.py              # PhantomEnv (step/reset/state)
│   ├── models.py           # Pydantic models (Action, Observation, Reward)
│   ├── network.py          # Network simulation (hosts, edges, presets)
│   ├── attack_engine.py    # Adversary lateral movement
│   ├── siem.py             # SIEM log generation + injections
│   ├── task_grader.py      # Task grading (3 tasks, multi-dimensional)
│   ├── api.py              # FastAPI HTTP interface
│   ├── gpt_client.py       # LLM client with circuit breaker
│   ├── gpt_injection.py    # GPT-powered adversarial injection generation
│   ├── intelligent_siem.py # GPT-enhanced SIEM noise
│   └── dynamic_topology.py # GPT-generated network topologies
├── config/
│   ├── development.yaml
│   └── production.yaml
└── tests/
    ├── test_env.py
    ├── test_models.py
    ├── test_network.py
    ├── test_siem.py
    ├── test_task_grader.py
    └── ...
```

---

## License

MIT
