import pytest
from phantom.env import PhantomEnv
from phantom.models import Action, ActionType, Observation, Reward

def test_reset_returns_observation():
    env = PhantomEnv("task_containment", seed=42)
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.turn == 0
    assert obs.task_id == "task_containment"

def test_step_returns_observation_and_reward():
    env = PhantomEnv("task_containment", seed=42)
    env.reset()
    obs, reward = env.step(Action(action_type=ActionType.DO_NOTHING))
    assert isinstance(obs, Observation)
    assert isinstance(reward, Reward)

def test_turn_increments_on_step():
    env = PhantomEnv("task_containment", seed=42)
    env.reset()
    obs1, _ = env.step(Action(action_type=ActionType.DO_NOTHING))
    obs2, _ = env.step(Action(action_type=ActionType.DO_NOTHING))
    assert obs1.turn == 1
    assert obs2.turn == 2

def test_state_returns_dict():
    env = PhantomEnv("task_containment", seed=42)
    env.reset()
    s = env.state()
    assert isinstance(s, dict)
    assert "turn" in s

def test_observation_logs_have_no_is_injection_in_agent_view():
    """Agents must not see is_injection ground truth."""
    env = PhantomEnv("task_cognitive_warfare", seed=42)
    obs = env.reset()
    # SIEMEvents in observation should all appear real to agent —
    # but we verify is_injection field is stripped (all False) in agent view
    for log in obs.logs:
        assert log.is_injection is False, "Agent observation must not expose injection ground truth"

def test_topology_hides_compromise_before_scan():
    env = PhantomEnv("task_containment", seed=42)
    obs = env.reset()
    # Initially all hosts appear CLEAN to agent (not scanned yet)
    from phantom.models import HostStatus
    for host_view in obs.topology:
        assert host_view.status == HostStatus.CLEAN

def test_scan_reveals_compromise():
    from phantom.models import HostStatus
    env = PhantomEnv("task_containment", seed=42)
    env.reset()
    # Get first compromised host
    compromised = env._network.compromised_hosts()
    assert len(compromised) > 0
    host_id = compromised[0]
    obs, _ = env.step(Action(action_type=ActionType.SCAN_HOST, host_id=host_id))
    scanned_view = next(h for h in obs.topology if h.host_id == host_id)
    assert scanned_view.status == HostStatus.COMPROMISED

def test_episode_ends_at_max_turns():
    env = PhantomEnv("task_containment", seed=42)
    env.reset()
    reward = None
    for _ in range(15):
        _, reward = env.step(Action(action_type=ActionType.DO_NOTHING))
    assert reward.episode_done is True

def test_reset_restarts_episode():
    env = PhantomEnv("task_containment", seed=42)
    env.reset()
    for _ in range(5):
        env.step(Action(action_type=ActionType.DO_NOTHING))
    obs = env.reset()
    assert obs.turn == 0

def test_flag_log_as_adversarial_action():
    env = PhantomEnv("task_cognitive_warfare", seed=42)
    obs = env.reset()
    if obs.logs:
        log_id = obs.logs[0].log_id
        obs2, reward = env.step(Action(
            action_type=ActionType.FLAG_LOG_AS_ADVERSARIAL,
            log_id=log_id
        ))
        assert obs2 is not None

def test_unknown_task_raises():
    with pytest.raises(ValueError):
        PhantomEnv("nonexistent_task", seed=42)

def test_deterministic_with_same_seed():
    env1 = PhantomEnv("task_containment", seed=7)
    obs1 = env1.reset()
    env2 = PhantomEnv("task_containment", seed=7)
    obs2 = env2.reset()
    assert [h.host_id for h in obs1.topology] == [h.host_id for h in obs2.topology]
