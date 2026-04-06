from __future__ import annotations
import asyncio
import os
from typing import Callable
from openai import AsyncOpenAI


class CircuitBreakerOpen(Exception):
    pass


class CircuitBreaker:
    def __init__(self, threshold: int = 5):
        self._threshold = threshold
        self._failures = 0
        self._open = False

    def is_open(self) -> bool:
        return self._open

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._open = True

    def record_success(self) -> None:
        self._failures = 0
        self._open = False


class GPTClient:
    def __init__(self, primary_model: str, fallback_model: str):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        # Use a sentinel key when none is set so the client can be constructed
        # without a real API key in test environments (tests mock _openai.chat).
        api_key = os.environ.get("OPENAI_API_KEY") or "sk-phantom-no-key"
        self._openai = AsyncOpenAI(api_key=api_key)
        self._circuit_breaker = CircuitBreaker()

    async def generate(
        self,
        prompt: str,
        fallback_fn: Callable[[], str],
        system: str = "You are a helpful assistant.",
        timeout: float = 30.0,
    ) -> str:
        if self._circuit_breaker.is_open():
            return fallback_fn()

        for model in (self.primary_model, self.fallback_model):
            try:
                resp = await asyncio.wait_for(
                    self._openai.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    ),
                    timeout=timeout,
                )
                self._circuit_breaker.record_success()
                return resp.choices[0].message.content
            except Exception:
                self._circuit_breaker.record_failure()
                if self._circuit_breaker.is_open():
                    break

        return fallback_fn()
