import random
import pytest
from phantom.network import NetworkState

@pytest.fixture
def rng():
    return random.Random(42)

@pytest.fixture
def small_network(rng):
    return NetworkState.from_preset("small_corp", rng)
