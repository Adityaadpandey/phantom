from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    host_id: str | None = None
    log_id: str | None = None
    traffic_rule: str | None = None
    incident_report: str | None = None
    threat_query: str | None = None
    reasoning: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> "Action":
        def _has(value: str | None) -> bool:
            return bool(value and value.strip())

        required_field_by_action = {
            ActionType.SCAN_HOST: "host_id",
            ActionType.ISOLATE_HOST: "host_id",
            ActionType.PATCH_HOST: "host_id",
            ActionType.RESTORE_HOST: "host_id",
            ActionType.FLAG_LOG_AS_ADVERSARIAL: "log_id",
            ActionType.BLOCK_TRAFFIC: "traffic_rule",
            ActionType.SUBMIT_INCIDENT_REPORT: "incident_report",
            ActionType.QUERY_THREAT_INTEL: "threat_query",
        }
        required = required_field_by_action.get(self.action_type)
        if required is not None and not _has(getattr(self, required)):
            raise ValueError(f"{required} is required for action_type={self.action_type.value}")

        if self.action_type == ActionType.DO_NOTHING:
            disallowed = [
                "host_id",
                "log_id",
                "traffic_rule",
                "incident_report",
                "threat_query",
            ]
            used = [name for name in disallowed if _has(getattr(self, name))]
            if used:
                raise ValueError(f"do_nothing must not include action fields: {', '.join(used)}")

        return self


class Reward(BaseModel):
    # Weighted final score normalised to [0, 1] for submission compatibility.
    total: float = Field(ge=0.0, le=1.0)
    # Component scores are intentionally not constrained to [0, 1]:
    # penalties can push containment/cognitive below zero.
    containment_score: float
    cognitive_score: float
    # Communication/efficiency are bounded and non-negative.
    communication_score: float = Field(ge=0.0, le=1.0)
    efficiency_bonus: float = Field(ge=0.0, le=1.0)
    episode_done: bool
    info: dict
