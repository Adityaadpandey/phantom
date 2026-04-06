import random
import pytest
from phantom.network import NetworkState
from phantom.attack_engine import AttackEngine

@pytest.fixture
def net():
    return NetworkState.from_preset("small_corp", random.Random(42))

def test_initial_entry_point_is_compromised(net):
    engine = AttackEngine(net, rng=random.Random(42), speed="normal")
    engine.initialize()
    assert net.any_compromised()

def test_step_returns_list_of_strings(net):
    engine = AttackEngine(net, rng=random.Random(42), speed="normal")
    engine.initialize()
    result = engine.step(turn=1)
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)

def test_attack_spreads_over_turns(net):
    engine = AttackEngine(net, rng=random.Random(42), speed="fast")
    engine.initialize()
    compromised_at_start = len(net.compromised_hosts())
    for t in range(1, 6):
        engine.step(turn=t)
    assert len(net.compromised_hosts()) >= compromised_at_start

def test_isolated_host_not_spread_to(net):
    engine = AttackEngine(net, rng=random.Random(42), speed="fast")
    engine.initialize()
    # Isolate every host
    for hid in list(net.hosts.keys()):
        net.isolate_host(hid)
    initial = set(net.compromised_hosts())
    for t in range(1, 10):
        engine.step(turn=t)
    assert set(net.compromised_hosts()) == initial

def test_deterministic_with_same_seed():
    net1 = NetworkState.from_preset("small_corp", random.Random(7))
    engine1 = AttackEngine(net1, rng=random.Random(7), speed="normal")
    engine1.initialize()
    for t in range(1, 5):
        engine1.step(t)

    net2 = NetworkState.from_preset("small_corp", random.Random(7))
    engine2 = AttackEngine(net2, rng=random.Random(7), speed="normal")
    engine2.initialize()
    for t in range(1, 5):
        engine2.step(t)

    assert net1.compromised_hosts() == net2.compromised_hosts()

def test_exfiltration_flag_set_when_crown_jewel_reached(net):
    engine = AttackEngine(net, rng=random.Random(42), speed="fast")
    engine.initialize()
    # Force-compromise the crown jewel
    cj = next(hid for hid, h in net.hosts.items() if h.is_crown_jewel)
    net.compromise_host(cj, turn=1)
    assert net.exfiltration_complete()
