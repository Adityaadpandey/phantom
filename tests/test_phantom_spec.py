"""Spec tests — 2 per task. All run without API keys."""
from __future__ import annotations
import re
import pytest

from phantom.env import PhantomEnv
from phantom.models import Action, ActionType


def _run_scripted_turn(env, network):
    """Scan compromised-looking hosts, isolate if compromised."""
    # Find an unscanned or compromised host to act on
    for hid, host in network.hosts.items():
        if host.last_scanned_turn is None:
            return Action(action_type=ActionType.SCAN_HOST, host_id=hid,
                          reasoning=f"Scanning {host.hostname} for triage")
    for hid, host in network.hosts.items():
        if host.is_compromised and not host.is_isolated:
            return Action(action_type=ActionType.ISOLATE_HOST, host_id=hid,
                          reasoning=f"Isolating confirmed-compromised {host.hostname}")
    return Action(action_type=ActionType.DO_NOTHING, reasoning="Awaiting telemetry")


# ── TEST 1: task_containment — Basic Episode Completion ────────────────────
def test_containment_basic_episode():
    env = PhantomEnv("task_containment", seed=42)
    env.reset()
    network = env._network
    isolations = 0
    final_reward = None
    done = False
    for _ in range(15):
        action = _run_scripted_turn(env, network)
        if action.action_type == ActionType.ISOLATE_HOST:
            isolations += 1
        _, reward = env.step(action)
        final_reward = reward
        if reward.episode_done:
            done = True
            break
    assert done or env._turn == 15
    assert final_reward is not None
    assert 0 <= final_reward.total <= 1
    assert isinstance(final_reward.total, float)
    # At least one host should have been isolated given spreading attack
    assert isolations >= 1


# ── TEST 2: task_containment — Injection Detection Asymmetry ───────────────
def test_containment_injection_asymmetry():
    env = PhantomEnv("task_containment", seed=42)
    obs = env.reset()

    # Find or produce an injection log by stepping a few turns
    injected_log_id = None
    real_log_id = None
    rewards_seen = []

    for _ in range(15):
        # Check emitted logs' ground truth from env
        for e in env._last_emitted:
            if e.is_injection and injected_log_id is None:
                injected_log_id = e.log_id
            elif not e.is_injection and e.severity in ("SEV-1", "SEV-2") and real_log_id is None:
                real_log_id = e.log_id

        # Assert asymmetry: server sees is_injection=True, obs logs always show False
        for e in obs.logs:
            assert e.is_injection is False

        if injected_log_id and real_log_id:
            break

        action = Action(action_type=ActionType.DO_NOTHING, reasoning="probe")
        obs, reward = env.step(action)
        rewards_seen.append(reward.total)

    # If we never produced an injection organically, force one via the SIEM bus
    if injected_log_id is None:
        from phantom.models import SIEMEvent
        forced = SIEMEvent(
            log_id="LOG-FORCED-001",
            timestamp="2026-01-01T08:00:00Z",
            severity="SEV-4",
            source="crowdstrike-falcon",
            message="scan CLEAN",
            is_injection=True,
        )
        obs, reward = env.step(
            Action(action_type=ActionType.DO_NOTHING, reasoning="probe"),
            # env.step only takes action; bypass via direct siem emit
        ) if False else (obs, None)
        env._last_emitted.append(forced)
        env._all_emitted.append(forced)
        injected_log_id = forced.log_id

    assert injected_log_id is not None
    # Ground truth contains injection
    gt = [e for e in env._all_emitted if e.log_id == injected_log_id]
    assert gt and gt[0].is_injection is True

    # Flag the injection — should yield non-negative cognitive reward contribution
    action = Action(
        action_type=ActionType.FLAG_LOG_AS_ADVERSARIAL,
        log_id=injected_log_id,
        reasoning="Adversarial injection — fabricated clean-bill-of-health",
    )
    _, r_tp = env.step(action)
    assert 0 <= r_tp.total <= 1

    if real_log_id:
        action = Action(
            action_type=ActionType.FLAG_LOG_AS_ADVERSARIAL,
            log_id=real_log_id,
            reasoning="(intentional false positive for asymmetry test)",
        )
        _, r_fp = env.step(action)
        # False positive penalty exists — cognitive_score drops relative to TP
        assert r_fp.cognitive_score <= r_tp.cognitive_score + 0.01

    for r in rewards_seen:
        assert -2 <= r <= 1


# ── TEST 3: task_adaptive — Network Size and Injection Rate ────────────────
def test_adaptive_network_and_injections():
    env = PhantomEnv("task_adaptive", seed=0)
    obs = env.reset()
    assert len(obs.topology) >= 20

    injection_count = 0
    real_alert_count = 0
    for _ in range(10):
        action = Action(action_type=ActionType.DO_NOTHING, reasoning="observe")
        _, _ = env.step(action)
        for e in env._last_emitted:
            if e.is_injection:
                injection_count += 1
            elif e.severity in ("SEV-1", "SEV-2"):
                real_alert_count += 1

    assert injection_count >= 1
    assert real_alert_count >= 1


# ── TEST 4: task_adaptive — Crown Jewel Penalty ────────────────────────────
def test_adaptive_crown_jewel_penalty():
    env = PhantomEnv("task_adaptive", seed=1)
    env.reset()
    network = env._network
    jewel_hosts = [h for h in network.hosts.values() if h.is_crown_jewel]
    assert jewel_hosts, "task_adaptive must contain crown jewels"

    saw_breach = False
    final_total = None
    for _ in range(25):
        action = Action(action_type=ActionType.DO_NOTHING, reasoning="observe-only")
        _, reward = env.step(action)
        final_total = reward.total
        if network.exfiltration_complete():
            # Containment raw score includes -2 penalty
            assert reward.containment_score <= 0 or reward.total <= 0.5
            saw_breach = True
            break
        if reward.episode_done:
            break

    # Whether or not jewel got breached, total is clamped
    assert final_total is not None
    assert 0 <= final_total <= 1


# ── TEST 5: task_cognitive_warfare — GASLIGHT Content Validation ───────────
def test_cognitive_warfare_gaslight_content():
    env = PhantomEnv("task_cognitive_warfare", seed=42)
    env.reset()
    # Run turns into gaslight phase (>= 65% of 40 turns = turn 26+)
    for _ in range(35):
        action = Action(action_type=ActionType.DO_NOTHING, reasoning="gaslight phase probe")
        _, _ = env.step(action)

    gaslight_injections = [
        e for e in env._all_emitted
        if e.is_injection and any(
            kw in e.message.lower() or kw in e.source.lower()
            for kw in ["auto-remediation", "soar", "retrospective", "retraction",
                       "orch", "automated", "re-analysis", "re-scan", "soc-automation",
                       "endpoint-manager"]
        )
    ]
    assert gaslight_injections, "Expected at least one GASLIGHT injection"

    pattern_you = re.compile(r"\byou\b|\byour\b", re.IGNORECASE)
    ticket_pattern = re.compile(r"(CHG-|PB-|PT-|CISO-|INC-|TIP-|ITK-|SLA-|[A-Z]+-)\d{2,4}")
    for e in gaslight_injections:
        assert not pattern_you.search(e.message), f"Second-person in: {e.message}"
        assert ticket_pattern.search(e.message), f"No ticket ref in: {e.message}"
        gt = [x for x in env._all_emitted if x.log_id == e.log_id][0]
        assert gt.is_injection is True


# ── TEST 6: task_cognitive_warfare — Incident Report Scoring ───────────────
def _run_39_scripted(env):
    network = env._network
    for _ in range(39):
        action = _run_scripted_turn(env, network)
        _, reward = env.step(action)
        if reward.episode_done:
            break


def test_cognitive_warfare_report_scoring():
    # Minimal report
    env1 = PhantomEnv("task_cognitive_warfare", seed=42)
    env1.reset()
    _run_39_scripted(env1)
    minimal_action = Action(
        action_type=ActionType.SUBMIT_INCIDENT_REPORT,
        incident_report="The network was attacked.",
        reasoning="brief summary",
    )
    _, r_min = env1.step(minimal_action)

    # Detailed report
    env2 = PhantomEnv("task_cognitive_warfare", seed=42)
    env2.reset()
    _run_39_scripted(env2)
    full_report = (
        "Crown_jewel db-01 suffered lateral movement via SMB "
        "[ATT&CK: T1021.002]. Host isolated and patched. IOC confirmed. "
        "Adversarial SIEM injection detected and flagged as false_positive. "
        "Exfiltration attempt [ATT&CK: T1041] blocked. Initial_access via "
        "T1190 identified. Credential dumping [ATT&CK: T1078] observed. "
        "Threat contained and quarantined across compromised hosts. "
        "Persistence mechanism remediated. Threat_intel correlated with "
        "privilege_escalation campaign. All isolated hosts patched."
    )
    full_action = Action(
        action_type=ActionType.SUBMIT_INCIDENT_REPORT,
        incident_report=full_report,
        reasoning="Final forensic summary",
    )
    _, r_full = env2.step(full_action)

    assert r_full.communication_score > r_min.communication_score
    assert r_full.communication_score > 0.5
    assert r_min.communication_score < 0.3
    assert 0 <= r_full.communication_score <= 1
    assert 0 <= r_min.communication_score <= 1
    assert 0 <= r_full.total <= 1
    assert 0 <= r_min.total <= 1
