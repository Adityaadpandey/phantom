"""
PHANTOM Environment — OpenEnv client.

Root-level client required by the OpenEnv spec.
Connects to the PHANTOM FastAPI server via WebSocket using openenv.core.EnvClient.

Usage:
    # Connect to a running server
    with PhantomEnvClient(base_url="http://localhost:7860") as client:
        result = client.reset()
        result = client.step(PhantomAction(action_type="scan_host", host_id="db-01")

    # Auto-start via Docker
    client = PhantomEnvClient.from_docker_image("phantom:latest")
"""

from __future__ import annotations
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import PhantomAction, PhantomObservation


class PhantomEnvClient(
    EnvClient[PhantomAction, PhantomObservation, State]
):
    """
    Client for the PHANTOM Adversarial Cognitive Security Environment.

    Maintains a persistent WebSocket connection to the PHANTOM server,
    enabling efficient multi-step incident-response interactions.
    """

    def _step_payload(self, action: PhantomAction) -> Dict:
        """Convert PhantomAction to JSON payload for the step endpoint."""
        payload: Dict = {"action_type": action.action_type}
        if action.host_id is not None:
            payload["host_id"] = action.host_id
        if action.log_id is not None:
            payload["log_id"] = action.log_id
        if action.traffic_rule is not None:
            payload["traffic_rule"] = action.traffic_rule
        if action.incident_report is not None:
            payload["incident_report"] = action.incident_report
        if action.threat_query is not None:
            payload["threat_query"] = action.threat_query
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning
        return {"action": payload}

    def _parse_result(self, payload: Dict) -> StepResult[PhantomObservation]:
        """Parse server step response into StepResult[PhantomObservation].

        The server returns `reward` as a float and `done` as a bool at the
        top level of the step response (see StepResponse in phantom/api.py).
        For reset responses, the payload IS the observation dict directly.
        """
        if "observation" in payload:
            obs_data = payload["observation"]
            reward = payload.get("reward")
            done = payload.get("done", False)
        else:
            obs_data = payload
            reward = None
            done = False

        # Guard against a dict-shaped reward (future schema change)
        if isinstance(reward, dict):
            reward = reward.get("total")
            done = payload.get("reward", {}).get("episode_done", done) if isinstance(payload.get("reward"), dict) else done

        reward_float = float(reward) if reward is not None else 0.0
        reward_float = max(0.0, min(1.0, reward_float))

        observation = PhantomObservation(
            turn=obs_data.get("turn", 0),
            max_turns=obs_data.get("max_turns", 15),
            actions_remaining=obs_data.get("actions_remaining", 0),
            logs=obs_data.get("logs", []),
            topology=obs_data.get("topology", []),
            alert_summary=obs_data.get("alert_summary", {}),
            previous_action_result=obs_data.get("previous_action_result"),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            done=bool(done),
            reward=reward_float,
        )

        return StepResult(
            observation=observation,
            reward=reward_float,
            done=bool(done),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server state response into State object."""
        return State(
            episode_id=payload.get("task_id"),
            step_count=payload.get("turn", 0),
        )
