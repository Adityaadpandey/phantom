import pytest
from httpx import AsyncClient, ASGITransport
from phantom.api import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_reset_returns_observation():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/reset/task_containment", json={"seed": 42})
    assert resp.status_code == 200
    data = resp.json()
    assert data["turn"] == 0
    assert data["task_id"] == "task_containment"
    assert "topology" in data
    assert "logs" in data

@pytest.mark.asyncio
async def test_step_returns_observation_and_reward():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/reset/task_containment", json={"seed": 42})
        resp = await client.post("/step/task_containment", json={
            "action": {"action_type": "do_nothing"}
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert "reward" in data
    assert data["observation"]["turn"] == 1

@pytest.mark.asyncio
async def test_reset_unknown_task_returns_422():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/reset/nonexistent_task", json={"seed": 42})
    assert resp.status_code == 422

@pytest.mark.asyncio
async def test_step_before_reset_returns_400():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/step/task_adaptive", json={
            "action": {"action_type": "do_nothing"}
        })
    assert resp.status_code == 400

@pytest.mark.asyncio
async def test_state_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/reset/task_containment", json={"seed": 42})
        resp = await client.get("/state/task_containment")
    assert resp.status_code == 200
    assert "turn" in resp.json()
