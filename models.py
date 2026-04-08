"""
PHANTOM Environment — OpenEnv data models.

Root-level models required by the OpenEnv spec.
These extend the openenv base types and wrap PHANTOM's internal Pydantic models.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class PhantomAction(Action):
    """Action for the PHANTOM environment."""

    action_type: str = Field(..., description=(
        "One of: scan_host, isolate_host, patch_host, restore_host, "
        "block_traffic, flag_log_as_adversarial, submit_incident_report, "
        "query_threat_intel, do_nothing"
    ))
    host_id: Optional[str] = Field(default=None, description="Target host ID (for host actions)")
    log_id: Optional[str] = Field(default=None, description="Target log ID (for flag action)")
    traffic_rule: Optional[str] = Field(default=None, description="Firewall rule string")
    incident_report: Optional[str] = Field(default=None, description="Incident report text (500+ chars)")
    threat_query: Optional[str] = Field(default=None, description="Threat intel query string")
    reasoning: Optional[str] = Field(default=None, description="Agent reasoning for this action")


class PhantomObservation(Observation):
    """Observation from the PHANTOM environment."""

    turn: int = Field(default=0, description="Current turn number")
    max_turns: int = Field(default=15, description="Maximum turns for this episode")
    actions_remaining: int = Field(default=15, description="Turns remaining")
    logs: List[Dict[str, Any]] = Field(default_factory=list, description="SIEM events this turn")
    topology: List[Dict[str, Any]] = Field(default_factory=list, description="Network host views")
    alert_summary: Dict[str, int] = Field(default_factory=dict, description="Alert counts by severity")
    previous_action_result: Optional[str] = Field(default=None, description="Result of last action")
    task_id: str = Field(default="", description="Active task identifier")
    task_description: str = Field(default="", description="Human-readable task objective")
