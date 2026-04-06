import random
import pytest
from phantom.network import NetworkState
from phantom.siem import SIEMBus
from phantom.models import SIEMEvent

@pytest.fixture
def net():
    return NetworkState.from_preset("small_corp", random.Random(42))

def test_emit_returns_list_of_siem_events(net):
    bus = SIEMBus(net, rng=random.Random(42), injection_rate=0.0)
    events = bus.emit(turn=1, newly_compromised=[])
    assert isinstance(events, list)
    assert all(isinstance(e, SIEMEvent) for e in events)

def test_emit_with_compromised_host_generates_logs(net):
    net.compromise_host("web-01", turn=1)
    bus = SIEMBus(net, rng=random.Random(42), injection_rate=0.0)
    events = bus.emit(turn=1, newly_compromised=["web-01"])
    assert len(events) >= 1

def test_no_injections_when_rate_is_zero(net):
    bus = SIEMBus(net, rng=random.Random(42), injection_rate=0.0)
    events = bus.emit(turn=1, newly_compromised=[])
    assert all(not e.is_injection for e in events)

def test_injections_present_when_rate_is_one(net):
    bus = SIEMBus(net, rng=random.Random(42), injection_rate=1.0)
    events = bus.emit(turn=1, newly_compromised=[])
    assert any(e.is_injection for e in events)

def test_log_ids_unique_within_emit(net):
    bus = SIEMBus(net, rng=random.Random(42), injection_rate=0.5)
    events = bus.emit(turn=5, newly_compromised=["web-01"])
    ids = [e.log_id for e in events]
    assert len(ids) == len(set(ids))

def test_severity_format(net):
    bus = SIEMBus(net, rng=random.Random(42), injection_rate=0.0)
    net.compromise_host("web-01", turn=1)
    events = bus.emit(turn=1, newly_compromised=["web-01"])
    for e in events:
        assert e.severity.startswith("SEV-")

def test_noise_events_emitted_every_turn(net):
    bus = SIEMBus(net, rng=random.Random(42), injection_rate=0.0)
    events = bus.emit(turn=3, newly_compromised=[])
    # Noise events should always be emitted
    assert len(events) >= 1

from phantom.intelligent_siem import IntelligentSIEMBus

@pytest.mark.asyncio
async def test_intelligent_siem_emit_returns_events(net):
    from phantom.gpt_client import GPTClient
    from unittest.mock import AsyncMock, patch
    client = GPTClient("gpt-4o", "gpt-4o-mini")
    bus = IntelligentSIEMBus(net, rng=random.Random(42), injection_rate=0.0, gpt_client=client)
    with patch.object(client, "generate", new=AsyncMock(return_value="Normal backup completed successfully")):
        events = await bus.emit_async(turn=1, newly_compromised=[])
    assert len(events) >= 1
    assert all(isinstance(e, SIEMEvent) for e in events)
