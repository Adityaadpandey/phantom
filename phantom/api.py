from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from phantom.env import PhantomEnv
from phantom.models import Action, Observation
from phantom.session_store import SessionStore
from phantom.task_grader import _TASK_CONFIGS

app = FastAPI(
    title="PHANTOM",
    description="Adversarial Cognitive Security Environment",
    version="1.0.0",
)

_sessions = SessionStore()

# Per-task replay history for counterfactual analysis:
#   task_id -> {"seed": int, "actions": list[Action]}
_episode_history: dict[str, dict] = {}


class ResetRequest(BaseModel):
    seed: int = 0


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool


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
    await _sessions.set(task_id, env)
    _episode_history[task_id] = {"seed": seed, "actions": []}
    return await env.areset()


def _record_action(task_id: str, seed: int, action: Action) -> None:
    hist = _episode_history.get(task_id)
    if hist is None or hist.get("seed") != seed:
        _episode_history[task_id] = {"seed": seed, "actions": [action]}
    else:
        hist["actions"].append(action)


def _round2(v: float) -> float:
    return round(float(v), 2)


def _require_active_session(task_id: str) -> PhantomEnv:
    env = _sessions.get(task_id)
    if env is None:
        raise HTTPException(status_code=400, detail=f"No active session for {task_id!r}. Call /reset first.")
    return env


# Root endpoints (no task_id) — used by the OpenEnv validator
@app.post("/reset", response_model=Observation)
async def reset_default(request: ResetRequest = ResetRequest()):
    return await _do_reset(_DEFAULT_TASK, request.seed)


@app.post("/step", response_model=StepResponse)
async def step_default(request: StepRequest):
    env = _require_active_session(_DEFAULT_TASK)
    obs, reward = env.step(request.action)
    _record_action(env.task_id, env.seed, request.action)
    return StepResponse(observation=obs, reward=reward.total, done=reward.episode_done)


@app.get("/state")
async def state_default():
    env = _require_active_session(_DEFAULT_TASK)
    return env.state()


# Per-task endpoints (task_id in path)
@app.post("/reset/{task_id}", response_model=Observation)
async def reset(task_id: str, request: ResetRequest = ResetRequest()):
    return await _do_reset(task_id, request.seed)


@app.post("/step/{task_id}", response_model=StepResponse)
async def step(task_id: str, request: StepRequest):
    env = _require_active_session(task_id)
    obs, reward = env.step(request.action)
    _record_action(env.task_id, env.seed, request.action)
    return StepResponse(observation=obs, reward=reward.total, done=reward.episode_done)


@app.get("/state/{task_id}")
async def state(task_id: str):
    env = _require_active_session(task_id)
    return env.state()


def _replay(task_id: str, seed: int, actions: list[Action], zero_injections: bool) -> dict:
    env = PhantomEnv(task_id, seed=seed)
    if zero_injections:
        env._injection_rate = 0
    env.reset()
    if zero_injections:
        env._siem.injection_rate = 0
    containment_total = 0.0
    cognitive_total = 0.0
    communication_total = 0.0
    last_total = 0.0
    steps = 0
    for a in actions:
        _, reward = env.step(a)
        containment_total += max(reward.containment_score, 0)
        cognitive_total += max(reward.cognitive_score, 0)
        communication_total += reward.communication_score
        last_total = reward.total
        steps += 1
        if reward.episode_done:
            break
    n = max(steps, 1)
    return {
        "containment": _round2(min(containment_total / n, 1)),
        "cognitive": _round2(min(cognitive_total / n, 1)),
        "overall": _round2(last_total),
    }


@app.get("/counterfactual/{task_id}")
async def counterfactual(task_id: str):
    if task_id not in _TASK_CONFIGS:
        raise HTTPException(status_code=422, detail=f"Unknown task_id: {task_id!r}")
    hist = _episode_history.get(task_id)
    if not hist or not hist.get("actions"):
        raise HTTPException(status_code=404, detail=f"No episode history for {task_id!r}. Run an episode first.")

    seed = hist["seed"]
    actions = hist["actions"]

    with_inj = _replay(task_id, seed, actions, zero_injections=False)
    without_inj = _replay(task_id, seed, actions, zero_injections=True)
    impact = _round2(with_inj["overall"] - without_inj["overall"])
    pct = int(round(abs(impact) * 100))
    direction = "reduced" if impact < 0 else "increased"
    return {
        "task_id": task_id,
        "seed": seed,
        "with_injections": with_inj,
        "without_injections": without_inj,
        "cognitive_warfare_impact": impact,
        "interpretation": f"Adversarial injections {direction} overall performance by {pct} percentage points.",
    }
