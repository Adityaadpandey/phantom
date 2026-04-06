from __future__ import annotations
import random
from phantom.network import NetworkState

# MITRE ATT&CK-inspired phases
_PHASES = [
    "initial_access",
    "execution",
    "persistence",
    "privilege_escalation",
    "lateral_movement",
    "collection",
    "exfiltration",
]


class AttackEngine:
    """Simulates adversary lateral movement through the network."""

    def __init__(self, network: NetworkState, rng: random.Random, speed: str = "normal"):
        self.network = network
        self.rng = rng
        # speed controls how aggressively the attacker spreads
        self._spread_probability = {"slow": 0.2, "normal": 0.4, "fast": 0.7}.get(speed, 0.4)
        self._phase_index = 0

    def initialize(self) -> str:
        """Compromise the initial entry point (internet-facing host)."""
        dmz_hosts = [
            hid for hid, h in self.network.hosts.items()
            if h.subnet in ("dmz", "edge") and not h.is_isolated
        ]
        if not dmz_hosts:
            dmz_hosts = list(self.network.hosts.keys())
        entry = self.rng.choice(dmz_hosts)
        self.network.compromise_host(entry, turn=0)
        return entry

    def step(self, turn: int) -> list[str]:
        """Advance attack one turn. Returns list of newly compromised host IDs."""
        self._advance_phase(turn)
        newly_compromised: list[str] = []

        for hid in list(self.network.compromised_hosts()):
            for neighbor in self.network.get_neighbors(hid):
                neighbor_host = self.network.hosts[neighbor]
                if (not neighbor_host.is_compromised
                        and not neighbor_host.is_isolated
                        and not neighbor_host.is_patched):
                    if self.rng.random() < self._spread_probability:
                        self.network.compromise_host(neighbor, turn)
                        newly_compromised.append(neighbor)

        return newly_compromised

    def _advance_phase(self, turn: int) -> None:
        """Progress through MITRE ATT&CK phases based on turn number."""
        phase_turn = turn // max(1, 5)  # advance phase every ~5 turns
        self._phase_index = min(phase_turn, len(_PHASES) - 1)

    @property
    def current_phase(self) -> str:
        return _PHASES[self._phase_index]
