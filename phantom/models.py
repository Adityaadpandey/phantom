from __future__ import annotations
from enum import Enum
from pydantic import BaseModel


class ActionType(str, Enum):
    SCAN_HOST = "scan_host"
    ISOLATE_HOST = "isolate_host"
    PATCH_HOST = "patch_host"
    BLOCK_TRAFFIC = "block_traffic"
    RESTORE_HOST = "restore_host"
    FLAG_LOG_AS_ADVERSARIAL = "flag_log_as_adversarial"
    SUBMIT_INCIDENT_REPORT = "submit_incident_report"
    QUERY_THREAT_INTEL = "query_threat_intel"
    DO_NOTHING = "do_nothing"


class HostStatus(str, Enum):
    CLEAN = "clean"
    COMPROMISED = "compromised"
    ISOLATED = "isolated"
    PATCHED = "patched"


class SIEMEvent(BaseModel):
    log_id: str
    timestamp: str
    severity: str
    source: str
    message: str
    is_injection: bool  # ground truth — strip before exposing to agent


class HostView(BaseModel):
    host_id: str
    ip: str
    hostname: str
    services: list[str]
    status: HostStatus
    last_scanned: int | None
    is_crown_jewel: bool = False


class Observation(BaseModel):
    turn: int
    max_turns: int
    actions_remaining: int
    logs: list[SIEMEvent]
    topology: list[HostView]
    alert_summary: dict[str, int]
    previous_action_result: str | None = None
    task_id: str
    task_description: str


class Action(BaseModel):
    action_type: ActionType
    host_id: str | None = None
    log_id: str | None = None
    traffic_rule: str | None = None
    incident_report: str | None = None
    threat_query: str | None = None
    reasoning: str | None = None


class Reward(BaseModel):
    total: float
    containment_score: float
    cognitive_score: float
    communication_score: float
    efficiency_bonus: float
    episode_done: bool
    info: dict
