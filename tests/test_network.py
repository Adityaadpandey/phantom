import random
import pytest
from phantom.network import NetworkState, Host

def test_from_preset_small_corp():
    rng = random.Random(42)
    net = NetworkState.from_preset("small_corp", rng)
    assert len(net.hosts) >= 5
    assert len(net.hosts) <= 20
    assert len(net.edges) >= 1

def test_from_preset_mid_corp():
    rng = random.Random(42)
    net = NetworkState.from_preset("mid_corp", rng)
    assert len(net.hosts) >= 20

def test_from_preset_enterprise():
    rng = random.Random(42)
    net = NetworkState.from_preset("enterprise", rng)
    assert len(net.hosts) >= 50

def test_unknown_preset_raises():
    rng = random.Random(42)
    with pytest.raises(ValueError, match="Unknown preset"):
        NetworkState.from_preset("nonexistent", rng)

def test_has_at_least_one_crown_jewel():
    rng = random.Random(42)
    net = NetworkState.from_preset("small_corp", rng)
    crown_jewels = [h for h in net.hosts.values() if h.is_crown_jewel]
    assert len(crown_jewels) >= 1

def test_host_ids_unique():
    rng = random.Random(42)
    net = NetworkState.from_preset("small_corp", rng)
    ids = list(net.hosts.keys())
    assert len(ids) == len(set(ids))

def test_all_contained_initially_false():
    rng = random.Random(42)
    net = NetworkState.from_preset("small_corp", rng)
    # No hosts compromised initially
    assert net.any_compromised() is False

def test_compromise_host():
    rng = random.Random(42)
    net = NetworkState.from_preset("small_corp", rng)
    first_host = next(iter(net.hosts.values()))
    net.compromise_host(first_host.host_id, turn=1)
    assert net.hosts[first_host.host_id].is_compromised is True
    assert net.any_compromised() is True

def test_isolate_host():
    rng = random.Random(42)
    net = NetworkState.from_preset("small_corp", rng)
    first_id = next(iter(net.hosts))
    net.compromise_host(first_id, turn=1)
    net.isolate_host(first_id)
    assert net.hosts[first_id].is_isolated is True

def test_all_contained_when_all_compromised_isolated():
    rng = random.Random(42)
    net = NetworkState.from_preset("small_corp", rng)
    for host_id in net.hosts:
        net.compromise_host(host_id, turn=1)
        net.isolate_host(host_id)
    assert net.all_contained() is True

def test_deterministic_with_same_seed():
    net1 = NetworkState.from_preset("small_corp", random.Random(99))
    net2 = NetworkState.from_preset("small_corp", random.Random(99))
    assert list(net1.hosts.keys()) == list(net2.hosts.keys())

def test_get_neighbors():
    rng = random.Random(42)
    net = NetworkState.from_preset("small_corp", rng)
    first_id = next(iter(net.hosts))
    neighbors = net.get_neighbors(first_id)
    assert isinstance(neighbors, list)
