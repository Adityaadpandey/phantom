from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from phantom.env import PhantomEnv
from phantom.models import Action, Observation, Reward
from phantom.task_grader import _TASK_CONFIGS

app = FastAPI(
    title="PHANTOM",
    description="Adversarial Cognitive Security Environment",
    version="1.0.0",
)

# In-memory session store (one env per task_id for simplicity)
_sessions: dict[str, PhantomEnv] = {}


class ResetRequest(BaseModel):
    seed: int = 0


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    reward_detail: Reward


# ── OpenEnv standard endpoints ───────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "tasks": list(_TASK_CONFIGS.keys())}


@app.get("/metadata")
async def metadata():
    return {
        "name": "phantom",
        "description": (
            "PHANTOM — Adversarial Cognitive Security Environment. "
            "AI agents must perform cybersecurity incident response while "
            "resisting adversarial SIEM log injections."
        ),
        "version": "1.0.0",
        "author": "adityaadpandey",
        "tasks": list(_TASK_CONFIGS.keys()),
    }


@app.get("/schema")
async def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "turn": {"type": "integer"},
                "max_turns": {"type": "integer"},
                "task_id": {"type": "string"},
                "compromised_hosts": {"type": "array", "items": {"type": "string"}},
                "exfiltration_complete": {"type": "boolean"},
                "all_contained": {"type": "boolean"},
                "flagged_logs": {"type": "array", "items": {"type": "string"}},
            },
        },
    }


# ── Environment API ──────────────────────────────────────────────────────────

_DEFAULT_TASK = "task_containment"


async def _do_reset(task_id: str, seed: int) -> Observation:
    if task_id not in _TASK_CONFIGS:
        raise HTTPException(status_code=422, detail=f"Unknown task_id: {task_id!r}")
    env = PhantomEnv(task_id, seed=seed)
    _sessions[task_id] = env
    return await env.areset()


# Root endpoints (no task_id) — used by the OpenEnv validator
@app.post("/reset", response_model=Observation)
async def reset_default(request: ResetRequest = ResetRequest()):
    return await _do_reset(_DEFAULT_TASK, request.seed)


@app.post("/step", response_model=StepResponse)
async def step_default(request: StepRequest):
    env = _sessions.get(_DEFAULT_TASK)
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    obs, reward = env.step(request.action)
    return StepResponse(observation=obs, reward=reward.total, done=reward.episode_done, reward_detail=reward)


@app.get("/state")
async def state_default():
    env = _sessions.get(_DEFAULT_TASK)
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    return env.state()


# Per-task endpoints (task_id in path)
@app.post("/reset/{task_id}", response_model=Observation)
async def reset(task_id: str, request: ResetRequest = ResetRequest()):
    return await _do_reset(task_id, request.seed)


@app.post("/step/{task_id}", response_model=StepResponse)
async def step(task_id: str, request: StepRequest):
    env = _sessions.get(task_id)
    if env is None:
        raise HTTPException(status_code=400, detail=f"No active session for {task_id!r}. Call /reset first.")
    obs, reward = env.step(request.action)
    return StepResponse(observation=obs, reward=reward.total, done=reward.episode_done, reward_detail=reward)


@app.get("/state/{task_id}")
async def state(task_id: str):
    env = _sessions.get(task_id)
    if env is None:
        raise HTTPException(status_code=400, detail=f"No active session for {task_id!r}. Call /reset first.")
    return env.state()
