from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from phantom.env import PhantomEnv
from phantom.models import Action, Observation, Reward
from phantom.task_grader import _TASK_CONFIGS

app = FastAPI(title="PHANTOM", description="Adversarial Cognitive Security Environment")

# In-memory session store (one env per task_id for simplicity)
_sessions: dict[str, PhantomEnv] = {}


class ResetRequest(BaseModel):
    seed: int = 0


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward


@app.get("/health")
async def health():
    return {"status": "ok", "tasks": list(_TASK_CONFIGS.keys())}


@app.post("/reset/{task_id}", response_model=Observation)
async def reset(task_id: str, request: ResetRequest):
    if task_id not in _TASK_CONFIGS:
        raise HTTPException(status_code=422, detail=f"Unknown task_id: {task_id!r}")
    env = PhantomEnv(task_id, seed=request.seed)
    _sessions[task_id] = env
    obs = env.reset()
    return obs


@app.post("/step/{task_id}", response_model=StepResponse)
async def step(task_id: str, request: StepRequest):
    env = _sessions.get(task_id)
    if env is None:
        raise HTTPException(status_code=400, detail=f"No active session for {task_id!r}. Call /reset first.")
    obs, reward = env.step(request.action)
    return StepResponse(observation=obs, reward=reward)


@app.get("/state/{task_id}")
async def state(task_id: str):
    env = _sessions.get(task_id)
    if env is None:
        raise HTTPException(status_code=400, detail=f"No active session for {task_id!r}. Call /reset first.")
    return env.state()
