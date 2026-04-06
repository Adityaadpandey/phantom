import random
import pytest
from unittest.mock import AsyncMock, patch
from phantom.dynamic_topology import DynamicTopologyGenerator
from phantom.network import NetworkState

MOCK_TOPOLOGY_JSON = '''
{
  "hosts": [
    {"host_id": "fw-01", "ip": "10.0.0.1", "hostname": "firewall-01",
     "services": ["firewall"], "os": "Linux", "subnet": "dmz", "is_crown_jewel": false},
    {"host_id": "web-01", "ip": "10.0.1.10", "hostname": "web-server-01",
     "services": ["http", "https"], "os": "Linux", "subnet": "dmz", "is_crown_jewel": false},
    {"host_id": "db-01", "ip": "10.1.0.10", "hostname": "db-01",
     "services": ["postgresql"], "os": "Linux", "subnet": "internal", "is_crown_jewel": true}
  ],
  "edges": [["fw-01", "web-01"], ["web-01", "db-01"]]
}
'''

@pytest.fixture
def generator():
    from phantom.gpt_client import GPTClient
    client = GPTClient(primary_model="gpt-4o", fallback_model="gpt-4o-mini")
    return DynamicTopologyGenerator(client)

@pytest.mark.asyncio
async def test_generate_returns_network_state(generator):
    with patch.object(generator._client, "generate", new=AsyncMock(return_value=MOCK_TOPOLOGY_JSON)):
        net = await generator.generate(template="tech_startup", size="small", seed=42)
    assert isinstance(net, NetworkState)
    assert len(net.hosts) == 3

@pytest.mark.asyncio
async def test_generate_falls_back_to_static_on_invalid_json(generator):
    with patch.object(generator._client, "generate", new=AsyncMock(return_value="not json")):
        net = await generator.generate(template="tech_startup", size="small", seed=42)
    assert isinstance(net, NetworkState)
    assert len(net.hosts) >= 5  # fallback preset

@pytest.mark.asyncio
async def test_generated_network_has_crown_jewel(generator):
    with patch.object(generator._client, "generate", new=AsyncMock(return_value=MOCK_TOPOLOGY_JSON)):
        net = await generator.generate(template="tech_startup", size="small", seed=42)
    crown_jewels = [h for h in net.hosts.values() if h.is_crown_jewel]
    assert len(crown_jewels) >= 1

@pytest.mark.asyncio
async def test_generate_validates_required_fields(generator):
    bad_json = '{"hosts": [{"host_id": "x"}], "edges": []}'
    with patch.object(generator._client, "generate", new=AsyncMock(return_value=bad_json)):
        net = await generator.generate(template="tech_startup", size="small", seed=42)
    # Should fall back to static preset
    assert isinstance(net, NetworkState)
