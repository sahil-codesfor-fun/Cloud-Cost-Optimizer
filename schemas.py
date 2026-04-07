from pydantic import BaseModel
from enum import Enum

class ActionType(str, Enum):
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    NO_OP = "NO_OP"

class Action(BaseModel):
    action_type: ActionType
    instance_count: int

class Observation(BaseModel):
    current_traffic: int
    active_instances: int
    cpu_utilization: float
    latency_ms: float

class Reward(BaseModel):
    value: float
    dropped_requests: int
    cost_incurred: float

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict