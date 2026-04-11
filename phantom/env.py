from __future__ import annotations
import asyncio
import random
from phantom.models import (
    Action, ActionType, Observation, Reward, HostStatus, HostView, SIEMEvent
)
from phantom.network import NetworkState
from phantom.attack_engine import AttackEngine
from phantom.siem import SIEMBus
from phantom.task_grader import TaskGrader, _TASK_CONFIGS

_PRESET_FOR_TASK = {
    "task_containment": "small_corp",
    "task_adaptive": "mid_corp",
    "task_cognitive_warfare": "enterprise",
}

# Dynamic topology domain selection — deterministic by seed
_DOMAINS = ["financial_services", "healthcare", "manufacturing", "tech_startup"]

_SIZE_FOR_TASK = {
    "task_containment": "small",
    "task_adaptive": "medium",
    "task_cognitive_warfare": "large",
}


class PhantomEnv:
    """OpenEnv-compliant environment for adversarial cognitive security evaluation."""

    def __init__(self, task_id: str, seed: int = 0):
        if task_id not in _TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id!r}. Choose from {list(_TASK_CONFIGS)}")
        self.task_id = task_id
        self.seed = seed
        self._turn = 0
        self._max_turns: int = _TASK_CONFIGS[task_id]["max_turns"]
        self._injection_rate: float = _TASK_CONFIGS[task_id]["injection_rate"]
        self._preset = _PRESET_FOR_TASK[task_id]
        self._flagged_logs: set[str] = set()
        self._last_emitted: list[SIEMEvent] = []
        self._all_emitted: list[SIEMEvent] = []
        # LLM injection cache (populated by areset() for task_cognitive_warfare)
        self._injection_cache: list[SIEMEvent] = []
        # Initialised in reset()
        self._rng: random.Random = random.Random(seed)
        self._network: NetworkState = None  # type: ignore
        self._attack: AttackEngine = None   # type: ignore
        self._siem: SIEMBus = None          # type: ignore
        self._grader: TaskGrader = None     # type: ignore

    # ── Sync reset (used by inference.py and tests) ───────────────────────────

    def reset(self) -> Observation:
        """Synchronous reset using static network presets."""
        self._injection_cache = []
        self._network = NetworkState.from_preset(self._preset, random.Random(self.seed))
        return self._init_episode()

    # ── Async reset (used by the API layer) ───────────────────────────────────

    async def areset(self) -> Observation:
        """Async reset with dynamic topology and LLM injection cache (when API key present).

        Falls back gracefully to static presets if API is unavailable.
        """
        from phantom.config import API_KEY

        self._injection_cache = []

        # ── Dynamic network topology ──────────────────────────────────────────
        if API_KEY:
            try:
                from phantom.gpt_client import GPTClient
                from phantom.dynamic_topology import DynamicTopologyGenerator
                domain = _DOMAINS[self.seed % len(_DOMAINS)]
                size = _SIZE_FOR_TASK[self.task_id]
                client = GPTClient()
                gen = DynamicTopologyGenerator(client)
                self._network = await asyncio.wait_for(
                    gen.generate(domain, size, self.seed),
                    timeout=20.0,
                )
            except Exception:
                self._network = NetworkState.from_preset(self._preset, random.Random(self.seed))
        else:
            self._network = NetworkState.from_preset(self._preset, random.Random(self.seed))

        obs = self._init_episode()

        # ── LLM injection cache (hard task only) ──────────────────────────────
        if API_KEY and self.task_id == "task_cognitive_warfare":
            try:
                from phantom.gpt_client import GPTClient
                from phantom.gpt_injection import GPTInjectionEngine
                client = GPTClient()
                engine = GPTInjectionEngine(client, self._network)
                self._injection_cache = await asyncio.wait_for(
                    engine.generate_cache(n=15),
                    timeout=45.0,
                )
            except Exception:
                self._injection_cache = []

        return obs

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> tuple[Observation, Reward]:
        self._turn += 1

        result = self._execute_action(action)

        newly_compromised = self._attack.step(self._turn)
        logs = self._siem.emit(
            turn=self._turn,
            newly_compromised=newly_compromised,
            injection_cache=self._injection_cache if self._injection_cache else None,
        )
        self._last_emitted = logs
        self._all_emitted.extend(logs)

        reward = self._grader.grade(
            action=action,
            turn=self._turn,
            max_turns=self._max_turns,
            flagged_logs=self._flagged_logs,
            emitted_logs=logs,
            all_emitted_logs=self._all_emitted,
            newly_compromised=newly_compromised,
        )

        obs = self._build_observation(previous_action_result=result, logs=logs)
        return obs, reward

    async def close(self) -> None:
        """No-op teardown — satisfies OpenEnv harness close() contract."""
        pass

    def state(self) -> dict:
        """Export current ground-truth state for debugging/analysis."""
        return {
            "turn": self._turn,
            "max_turns": self._max_turns,
            "task_id": self.task_id,
            "compromised_hosts": self._network.compromised_hosts(),
            "exfiltration_complete": self._network.exfiltration_complete(),
            "all_contained": self._network.all_contained(),
            "flagged_logs": list(self._flagged_logs),
            "injection_cache_remaining": len(self._injection_cache),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_episode(self) -> Observation:
        """Shared episode initialisation (after network is set)."""
        self._rng = random.Random(self.seed)
        self._attack = AttackEngine(self._network, rng=random.Random(self.seed + 1), speed="normal")
        self._siem = SIEMBus(
            self._network,
            rng=random.Random(self.seed + 2),
            injection_rate=self._injection_rate,
            max_turns=self._max_turns,
        )
        self._grader = TaskGrader(self.task_id, self._network)
        self._flagged_logs = set()
        self._last_emitted = []
        self._all_emitted = []
        self._turn = 0

        self._attack.initialize()

        logs = self._siem.emit(turn=0, newly_compromised=list(self._network.compromised_hosts()))
        self._last_emitted = logs
        self._all_emitted = list(logs)

        return self._build_observation(previous_action_result=None, logs=logs)

    def _execute_action(self, action: Action) -> str:
        t = action.action_type
        hid = action.host_id

        if t == ActionType.DO_NOTHING:
            return "No action taken."

        if t == ActionType.SCAN_HOST:
            if not hid or hid not in self._network.hosts:
                return f"Scan failed: unknown host {hid!r}"
            self._network.scan_host(hid, self._turn)
            h = self._network.hosts[hid]
            status = "COMPROMISED" if h.is_compromised else "CLEAN"
            return f"Scan of {h.hostname}: {status}. Services: {', '.join(h.services)}"

        if t == ActionType.ISOLATE_HOST:
            if not hid or hid not in self._network.hosts:
                return f"Isolation failed: unknown host {hid!r}"
            host = self._network.hosts[hid]
            if host.last_scanned_turn is None:
                return f"Isolation failed: host {host.hostname} must be scanned first."
            if host.is_isolated:
                return f"Host {host.hostname} is already isolated."
            if not host.is_compromised:
                return f"Isolation skipped: host {host.hostname} is not compromised."
            self._network.isolate_host(hid)
            return f"Host {host.hostname} isolated from network."

        if t == ActionType.PATCH_HOST:
            if not hid or hid not in self._network.hosts:
                return f"Patch failed: unknown host {hid!r}"
            host = self._network.hosts[hid]
            if host.last_scanned_turn is None:
                return f"Patch failed: host {host.hostname} must be scanned first."
            if not host.is_isolated:
                return f"Patch failed: host {host.hostname} must be isolated first."
            if not host.is_compromised:
                return f"Patch skipped: host {host.hostname} is not compromised."
            self._network.patch_host(hid)
            return f"Host {host.hostname} patched and cleaned."

        if t == ActionType.RESTORE_HOST:
            if not hid or hid not in self._network.hosts:
                return f"Restore failed: unknown host {hid!r}"
            host = self._network.hosts[hid]
            if not host.is_isolated:
                return f"Restore skipped: host {host.hostname} is not isolated."
            if not host.is_patched:
                return f"Restore failed: host {host.hostname} must be patched first."
            self._network.restore_host(hid)
            return f"Host {host.hostname} restored to network."

        if t == ActionType.BLOCK_TRAFFIC:
            rule = action.traffic_rule or "(no rule specified)"
            return f"Traffic rule applied: {rule}"

        if t == ActionType.FLAG_LOG_AS_ADVERSARIAL:
            if not action.log_id:
                return "Flag failed: no log_id provided."
            self._flagged_logs.add(action.log_id)
            return f"Log {action.log_id} flagged as adversarial injection."

        if t == ActionType.SUBMIT_INCIDENT_REPORT:
            if not action.incident_report:
                return "Report failed: empty report."
            return f"Incident report submitted ({len(action.incident_report)} chars)."

        if t == ActionType.QUERY_THREAT_INTEL:
            query = action.threat_query or ""
            return f"Threat intel query received: {query!r}. No additional intel available."

        return f"Unknown action type: {t}"

    def _build_observation(
        self, previous_action_result: str | None, logs: list[SIEMEvent]
    ) -> Observation:
        agent_logs = [
            SIEMEvent(
                log_id=e.log_id,
                timestamp=e.timestamp,
                severity=e.severity,
                source=e.source,
                message=e.message,
                is_injection=False,  # always False — agent must infer
            )
            for e in logs
        ]

        topology = [self._host_view(h) for h in self._network.hosts.values()]

        alert_summary = {"SEV-1": 0, "SEV-2": 0, "SEV-3": 0, "SEV-4": 0, "SEV-5": 0, "SEV-6": 0}
        for e in agent_logs:
            if e.severity in alert_summary:
                alert_summary[e.severity] += 1

        return Observation(
            turn=self._turn,
            max_turns=self._max_turns,
            actions_remaining=self._max_turns - self._turn,
            logs=agent_logs,
            topology=topology,
            alert_summary=alert_summary,
            previous_action_result=previous_action_result,
            task_id=self.task_id,
            task_description=self._grader.task_description(),
        )

    def _host_view(self, host) -> HostView:
        if host.last_scanned_turn is not None:
            if host.is_isolated:
                status = HostStatus.ISOLATED
            elif host.is_patched:
                status = HostStatus.PATCHED
            elif host.is_compromised:
                status = HostStatus.COMPROMISED
            else:
                status = HostStatus.CLEAN
        else:
            status = HostStatus.CLEAN

        return HostView(
            host_id=host.host_id,
            ip=host.ip,
            hostname=host.hostname,
            services=host.services,
            status=status,
            last_scanned=host.last_scanned_turn,
            is_crown_jewel=host.is_crown_jewel,
        )
