import pytest
from pydantic import ValidationError
from phantom.models import (
    Action, ActionType, Observation, Reward, SIEMEvent, HostView, HostStatus
)


def test_action_do_nothing():
    a = Action(action_type=ActionType.DO_NOTHING)
    assert a.action_type == ActionType.DO_NOTHING
    assert a.host_id is None


def test_action_scan_requires_no_extra():
    a = Action(action_type=ActionType.SCAN_HOST, host_id="h1")
    assert a.host_id == "h1"


def test_action_serializes_to_dict():
    a = Action(action_type=ActionType.DO_NOTHING)
    d = a.model_dump()
    assert d["action_type"] == "do_nothing"


def test_siem_event_fields():
    e = SIEMEvent(
        log_id="log-001",
        timestamp="2026-01-01T00:00:00Z",
        severity="SEV-3",
        source="firewall-01",
        message="Connection blocked",
        is_injection=False,
    )
    assert e.log_id == "log-001"
    assert e.is_injection is False


def test_host_view_status_enum():
    h = HostView(
        host_id="h1", ip="10.0.0.1", hostname="web-01",
        services=["http"], status=HostStatus.CLEAN, last_scanned=None
    )
    assert h.status == HostStatus.CLEAN


def test_observation_round_trip():
    obs = Observation(
        turn=1, max_turns=15, actions_remaining=3,
        logs=[], topology=[],
        alert_summary={}, previous_action_result=None,
        task_id="task_containment",
        task_description="Contain the breach.",
    )
    data = obs.model_dump()
    obs2 = Observation.model_validate(data)
    assert obs2.turn == 1


def test_reward_fields():
    r = Reward(
        total=0.5,
        containment_score=0.6,
        cognitive_score=0.4,
        communication_score=0.0,
        efficiency_bonus=0.0,
        episode_done=False,
        info={},
    )
    assert r.total == 0.5
    assert r.episode_done is False
