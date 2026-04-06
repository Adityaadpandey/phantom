import random
import pytest

@pytest.fixture
def rng():
    return random.Random(42)

@pytest.fixture
def small_network(rng):
    from phantom.network import NetworkState
    return NetworkState.from_preset("small_corp", rng)
