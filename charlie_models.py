from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# 🧩 Modelo de datos (memoria y ejecución) según Target.md

class UIAtom(BaseModel):
    name: str  # Ej: "buscador principal", "filtro ubicación", "botón aplicar"
    selector: Optional[str] = None
    last_seen_at: Optional[datetime] = None

class SiteProfile(BaseModel):
    domain: str
    known_atoms: Dict[str, UIAtom] = Field(default_factory=dict)
    login_required: bool = False
    common_popups: List[str] = Field(default_factory=list)

class Action(BaseModel):
    action_type: str  # "click", "fill", "type", "navigate", etc.
    target: str       # selector, app name, etc.
    value: Optional[str] = None
    description: str = ""

class FailurePattern(BaseModel):
    trigger_action: Action
    error_msg: str
    context_map_signature: str

class RecoveryPattern(BaseModel):
    failure_id: str
    successful_action: Action

class Trace(BaseModel):
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)
    success: bool = False

class Session(BaseModel):
    session_id: str
    task_description: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    traces: List[Trace] = Field(default_factory=list)
    final_result: str = ""

class Skill(BaseModel):
    name: str # Ej: "job_search_portal_x_v2"
    description: str
    trigger_conditions: Dict[str, Any]  # Ej: {"domain": "linkedin.com", "task_family": "job_search"}
    steps: List[str]  # Acciones abstractas o concretas
    success_rate: float = 1.0

class Task(BaseModel):
    objective: str
    environment: str = "browser"
    parameters: Dict[str, Any] = Field(default_factory=dict)
