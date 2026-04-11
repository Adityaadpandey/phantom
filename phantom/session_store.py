from __future__ import annotations

from phantom.env import PhantomEnv


class SessionStore:
    """Small in-memory store for per-task PhantomEnv sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, PhantomEnv] = {}

    def get(self, task_id: str) -> PhantomEnv | None:
        return self._sessions.get(task_id)

    async def set(self, task_id: str, env: PhantomEnv) -> None:
        previous = self._sessions.get(task_id)
        self._sessions[task_id] = env
        if previous is not None and previous is not env:
            await previous.close()

