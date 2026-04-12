from phantom.env import PhantomEnv
from phantom.models import Action, ActionType, Observation, Reward

# OpenEnv-spec-compliant aliases — the root spec requires these names to be
# importable directly from the `phantom` package namespace.
PhantomAction = Action
PhantomObservation = Observation

__all__ = [
    "PhantomEnv",
    "PhantomAction",
    "PhantomObservation",
    "Action",
    "ActionType",
    "Observation",
    "Reward",
]
