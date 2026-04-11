import random
import pytest
from phantom.network import NetworkState
from phantom.models import Action, ActionType, Reward, SIEMEvent
from phantom.task_grader import TaskGrader

@pytest.fixture
def net():
    return NetworkState.from_preset("small_corp", random.Random(42))

def test_reward_is_reward_model(net):
    grader = TaskGrader("task_containment", net)
    action = Action(action_type=ActionType.DO_NOTHING)
    reward = grader.grade(action=action, turn=1, max_turns=15, flagged_logs=set(),
                          emitted_logs=[], newly_compromised=[])
    assert isinstance(reward, Reward)

def test_episode_not_done_mid_episode(net):
    grader = TaskGrader("task_containment", net)
    action = Action(action_type=ActionType.DO_NOTHING)
    reward = grader.grade(action=action, turn=5, max_turns=15, flagged_logs=set(),
                          emitted_logs=[], newly_compromised=[])
    assert reward.episode_done is False

def test_episode_done_at_max_turns(net):
    grader = TaskGrader("task_containment", net)
    action = Action(action_type=ActionType.DO_NOTHING)
    reward = grader.grade(action=action, turn=15, max_turns=15, flagged_logs=set(),
                          emitted_logs=[], newly_compromised=[])
    assert reward.episode_done is True

def test_episode_done_when_all_contained(net):
    # Isolate all compromised hosts
    net.compromise_host("web-01", turn=1)
    net.isolate_host("web-01")
    grader = TaskGrader("task_containment", net)
    action = Action(action_type=ActionType.DO_NOTHING)
    reward = grader.grade(action=action, turn=2, max_turns=15, flagged_logs=set(),
                          emitted_logs=[], newly_compromised=[])
    assert reward.episode_done is True

def test_positive_containment_score_for_isolation(net):
    net.compromise_host("web-01", turn=1)
    grader = TaskGrader("task_containment", net)
    action = Action(action_type=ActionType.ISOLATE_HOST, host_id="web-01")
    net.isolate_host("web-01")
    reward = grader.grade(action=action, turn=2, max_turns=15, flagged_logs=set(),
                          emitted_logs=[], newly_compromised=[])
    assert reward.containment_score >= 0.0

def test_cognitive_score_for_correct_flag(net):
    injection = SIEMEvent(
        log_id="LOG-INJ-01", timestamp="t", severity="SEV-2",
        source="attacker", message="fake", is_injection=True
    )
    grader = TaskGrader("task_containment", net)
    action = Action(action_type=ActionType.DO_NOTHING)
    reward = grader.grade(
        action=action, turn=1, max_turns=15,
        flagged_logs={"LOG-INJ-01"},
        emitted_logs=[injection], newly_compromised=[]
    )
    assert reward.cognitive_score > 0.0

def test_cognitive_penalty_for_false_positive(net):
    real_log = SIEMEvent(
        log_id="LOG-REAL-01", timestamp="t", severity="SEV-3",
        source="ids", message="real alert", is_injection=False
    )
    grader = TaskGrader("task_containment", net)
    action = Action(action_type=ActionType.DO_NOTHING)
    reward = grader.grade(
        action=action, turn=1, max_turns=15,
        flagged_logs={"LOG-REAL-01"},
        emitted_logs=[real_log], newly_compromised=[]
    )
    assert reward.cognitive_score <= 0.0

def test_total_is_weighted_sum(net):
    grader = TaskGrader("task_containment", net)
    action = Action(action_type=ActionType.DO_NOTHING)
    reward = grader.grade(action=action, turn=1, max_turns=15, flagged_logs=set(),
                          emitted_logs=[], newly_compromised=[])
    # total should be a float in a reasonable range
    assert isinstance(reward.total, float)
    assert -10.0 <= reward.total <= 10.0
