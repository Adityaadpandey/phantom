from __future__ import annotations
import random
from phantom.siem import SIEMBus
from phantom.gpt_client import GPTClient
from phantom.network import NetworkState
from phantom.models import SIEMEvent

_CONTEXTUAL_SYSTEM = """You are a corporate SIEM generating realistic background noise events.
Generate a short, realistic SIEM log message (one sentence) appropriate for a normal business day.
Vary the type: authentication, network, system health, backup, patch management.
Return ONLY the message text, no JSON, no explanation."""


class IntelligentSIEMBus(SIEMBus):
    """SIEMBus enhanced with GPT-generated contextual background noise."""

    def __init__(
        self,
        network: NetworkState,
        rng: random.Random,
        injection_rate: float,
        gpt_client: GPTClient,
    ):
        super().__init__(network, rng, injection_rate)
        self._gpt = gpt_client

    async def emit_async(self, turn: int, newly_compromised: list[str]) -> list[SIEMEvent]:
        """Async emit with GPT-generated contextual noise."""
        events = self._emit_attack_logs(turn, newly_compromised)

        # Try to get GPT-generated noise
        try:
            host = self.rng.choice(list(self.network.hosts.values()))
            msg = await self._gpt.generate(
                prompt=f"Generate a background SIEM event for host {host.hostname} ({host.subnet} subnet) at turn {turn}.",
                system=_CONTEXTUAL_SYSTEM,
                fallback_fn=lambda: "",
                timeout=10.0,
            )
            if msg.strip():
                self._log_counter += 1
                events.append(SIEMEvent(
                    log_id=f"LOG-{self._log_counter:06d}",
                    timestamp=self._timestamp(turn),
                    severity="SEV-6",
                    source=f"{host.subnet}-monitor",
                    message=msg.strip(),
                    is_injection=False,
                ))
        except Exception:
            events.extend(self._emit_noise(turn))

        # Injections (use base GPT-less method)
        if self.rng.random() < self.injection_rate:
            events.extend(self._emit_injections(turn))

        return events
