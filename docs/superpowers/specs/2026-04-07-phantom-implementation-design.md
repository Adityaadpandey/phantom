# PHANTOM Implementation Design Document

**Date:** 2026-04-07  
**Project:** PHANTOM - Adversarial Cognitive Security Environment  
**Implementation Approach:** Phased Core-First Development  

## Overview

This document describes the implementation plan for the enhanced PHANTOM environment following a phased core-first approach. The system will be built incrementally, ensuring OpenEnv compliance at each phase while progressively adding sophisticated GPT-powered features.

## Architecture Design

### Layered System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PhantomEnv (OpenEnv Interface)          │
├─────────────────────────────────────────────────────────────┤
│  Observation │  Action   │  Reward   │  State Management   │
├─────────────────────────────────────────────────────────────┤
│           Core Game Logic & Grading System                  │
├─────────────────────────────────────────────────────────────┤
│ NetworkState │ AttackEngine │ SIEMBus │ TaskGrader        │
├─────────────────────────────────────────────────────────────┤
│              Enhanced GPT Integration Layer                  │
├─────────────────────────────────────────────────────────────┤
│ DynamicTopology │ GPTInjection │ IntelligentSIEM          │
└─────────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **OpenEnv Compliance First**: Every phase maintains strict adherence to OpenEnv specification
2. **Graceful Degradation**: GPT features enhance but never replace core functionality
3. **Incremental Testing**: Each component is independently testable
4. **Production Ready**: Built with error handling, monitoring, and deployment in mind

## Implementation Phases

### Phase 1: OpenEnv Infrastructure (Week 1-2)

**Deliverables:**
- Complete Pydantic models (`phantom/models.py`)
- Basic PhantomEnv class with reset/step/state interface
- Static network topology presets (small_corp, mid_corp, enterprise)
- Unit test framework and basic test coverage
- OpenEnv validation compliance

**Key Components:**
```python
# phantom/models.py
class Observation(BaseModel):
    turn: int
    max_turns: int
    actions_remaining: int
    logs: list[SIEMEvent]
    topology: list[HostView]
    alert_summary: dict[str, int]
    previous_action_result: str | None
    task_id: str
    task_description: str

class Action(BaseModel):
    action_type: ActionType
    host_id: str | None = None
    # ... all action parameters
    reasoning: str | None = None

class Reward(BaseModel):
    total: float
    # ... all component signals
    episode_done: bool
    info: dict
```

**Success Criteria:** 
- `openenv validate` passes
- Basic episode loop works with DO_NOTHING actions
- Deterministic behavior with seed control

### Phase 2: Core Game Logic (Week 2-3)

**Deliverables:**
- NetworkState with host management and topology generation
- AttackEngine with lateral movement simulation
- SIEMBus with real log generation (no injections yet)
- TaskGrader with per-step reward calculation
- All three tasks (containment, adaptive, cognitive_warfare) functional

**Key Components:**
```python
# phantom/network.py
class NetworkState:
    hosts: dict[str, Host]
    edges: list[tuple[str, str]]
    
    @classmethod
    def from_preset(cls, preset: str, rng: random.Random) -> "NetworkState"
    
    def all_contained(self) -> bool
    def exfiltration_complete(self) -> bool

# phantom/attack_engine.py  
class AttackEngine:
    def step(self, turn: int) -> list[str]  # Returns newly compromised hosts
    def _advance_phase(self, turn: int)     # MITRE ATT&CK progression

# phantom/siem.py
class SIEMBus:
    def emit(self, turn: int) -> list[SIEMEvent]
    def _emit_real(self, turn: int) -> list[SIEMEvent]
    def _emit_noise(self, turn: int) -> list[SIEMEvent]
```

**Success Criteria:**
- Full episode completion with all action types
- Realistic network intrusion progression
- Proper reward calculation for all scenarios
- Basic baseline agent achieves expected scores

### Phase 3: GPT Integration Layer (Week 3-4)

**Deliverables:**
- DynamicTopologyGenerator with GPT-5.4 integration
- GPTInjectionEngine for sophisticated adversarial content
- Enhanced SIEMBus with contextual background noise
- Robust error handling and fallback systems
- LLM-powered incident report grading

**Key Components:**
```python
# phantom/dynamic_topology.py
class DynamicTopologyGenerator:
    def generate(self, template: str, size: str, seed: int) -> NetworkState:
        # Uses GPT-5.4 to create realistic network layouts
        # Fallback to enhanced static presets on failure

# phantom/gpt_injection.py  
class GPTInjectionEngine:
    def generate_injections(self, context: dict, n: int) -> list[dict]:
        # Context-aware adversarial content generation
        # Analyzes network state and agent actions for contradictions
        # Fallback to sophisticated static templates

# phantom/intelligent_siem.py
class IntelligentSIEMBus(SIEMBus):
    def _emit_contextual_noise(self, turn: int) -> list[SIEMEvent]:
        # Domain-specific realistic background events
        # Time-of-day and business context awareness
```

**Success Criteria:**
- GPT-generated networks pass topology validation  
- Injection content achieves >8/10 realism score
- System maintains full functionality with API failures
- Performance targets met (<2s reset, <500ms step)

### Phase 4: Production Deployment (Week 4-5)

**Deliverables:**
- FastAPI wrapper with async endpoints
- Docker containerization with multi-stage builds
- HF Spaces deployment pipeline
- Comprehensive monitoring and observability
- Performance optimization and resource management

**Key Components:**
```python
# phantom/api.py
@app.post("/reset/{task_id}", response_model=Observation)
async def reset(task_id: str, config: dict = None):
    # Async episode management with proper error handling

@app.post("/step/{task_id}")  
async def step(task_id: str, action: Action):
    # Rate limiting and request validation

# Dockerfile
FROM python:3.11-slim
# Multi-stage build for production optimization
# Health checks and proper signal handling
```

**Success Criteria:**
- Successful HF Spaces deployment
- 99.9% uptime with proper monitoring
- Concurrent user support (10+ simultaneous episodes)
- API latency <500ms for all endpoints

## Data Flow & Integration

### Core Episode Loop
1. **Agent Action** → PhantomEnv validates parameters and routes to handler
2. **Action Execution** → Network state updates, results captured  
3. **World Advancement** → AttackEngine spreads intrusion, SIEMBus generates logs
4. **Observation Building** → Current state packaged for agent (no ground truth exposed)
5. **Reward Calculation** → TaskGrader evaluates both containment and cognitive performance

### GPT Integration Points
- **Dynamic Networks**: `NetworkState.from_dynamic()` → `DynamicTopologyGenerator.generate()`
- **Adversarial Content**: `SIEMBus._emit_injections()` → `GPTInjectionEngine.generate_injections()`
- **Contextual Noise**: `IntelligentSIEMBus._emit_noise()` → GPT-powered realistic events
- **Report Grading**: `TaskGrader._grade_report_llm()` → GPT evaluation against ground truth

### Error Handling Strategy
```python
class GPTClient:
    def __init__(self):
        self.primary_model = "gpt-5.4"
        self.fallback_model = "gpt-5"
        self.circuit_breaker = CircuitBreaker()
    
    async def generate_with_fallback(self, prompt: str, fallback_fn):
        try:
            return await self._call_gpt(self.primary_model, prompt)
        except Exception:
            if self.circuit_breaker.should_attempt():
                return await self._call_gpt(self.fallback_model, prompt)
            return fallback_fn()  # Static content as last resort
```

## Component Details

### Enhanced Network Topology System

**DynamicTopologyGenerator Features:**
- **Domain Templates**: Financial services, healthcare, manufacturing, tech startup
- **Realistic Structure**: Logical subnets (DMZ, corporate, management, guest)
- **Smart Services**: Role-appropriate service assignment (web servers → HTTP/HTTPS)
- **Authentic Naming**: Domain-specific hostname conventions
- **Scalable Complexity**: 10-200 hosts maintaining logical relationships

**Implementation Strategy:**
```python
def _generate_topology_prompt(self, template: str, size: str) -> str:
    return f"""
    Generate a realistic {template} network topology with {size} characteristics.
    Requirements:
    - Logical subnet segmentation (DMZ, internal, management)  
    - Appropriate services per host role
    - Realistic hostname conventions
    - Crown jewel placement in secure subnets
    - Valid IP addressing scheme
    
    Return JSON with hosts, connections, and services.
    """
```

### Sophisticated Injection System

**GPTInjectionEngine Capabilities:**
- **Authority Spoofing**: Fake management directives with proper authentication details
- **Technical Gaslighting**: Contradictory scan results that appear legitimate  
- **False Remediation**: Fake "all clear" messages masking ongoing compromise
- **Contextual Misdirection**: Injections that reference real network topology

**Context-Aware Generation:**
1. Analyze current network compromise status
2. Identify agent's recent actions and findings
3. Generate plausible contradictory evidence
4. Format with realistic SIEM conventions and correlation IDs

### Multi-Modal Grading System

**Scoring Components:**
- **Network Containment**: Real-time compromise spread tracking
- **Cognitive Integrity**: Injection detection accuracy vs false positive rate  
- **Communication Quality**: GPT-powered incident report evaluation
- **Operational Efficiency**: Action economy and response timing

## Testing Strategy

### Unit Testing (pytest + hypothesis)
- Component isolation with mocked dependencies
- Property-based testing for edge cases
- Deterministic behavior validation with seeds
- Pydantic model serialization verification

### Integration Testing  
- End-to-end episode execution with real GPT calls
- OpenEnv specification compliance validation
- Performance benchmarking (reset/step latency)
- Concurrent episode handling stress tests

### Quality Assurance
- GPT content quality validation (technical accuracy scoring)
- Network topology realism assessment
- Agent confusion rate measurement (successful injection rate)
- Production deployment verification on HF Spaces

## Configuration Management

```yaml
# config/development.yaml
environment:
  mode: "development"
  debug: true
  deterministic: true

llm:
  primary_model: "gpt-5.4"
  fallback_model: "gpt-5"
  timeout: 30
  max_retries: 3

tasks:
  task_containment:
    topology_mode: "static"  # Use presets during development
    injection_rate: 0.05
    
# config/production.yaml  
environment:
  mode: "production"
  debug: false
  deterministic: false

llm:
  rate_limit: 100  # requests per minute
  circuit_breaker_threshold: 5
  
tasks:
  task_containment:
    topology_mode: "dynamic"  # GPT-generated in production
    injection_rate: 0.05
```

## Success Metrics

### Functional Requirements
- ✅ OpenEnv specification compliance (passes validation)
- ✅ All three tasks functional with expected difficulty progression  
- ✅ GPT integration with graceful degradation
- ✅ Production deployment capability (Docker + HF Spaces)

### Quality Targets
- 📈 Injection realism score: >8/10 (human evaluation)
- 📈 Network authenticity: Passes corporate topology validation
- 📈 Performance: <2s episode reset, <500ms action processing
- 📈 Reliability: 99.9% uptime with proper error handling
- 📈 Agent challenge: Baseline GPT-4 score <0.6 on cognitive_warfare task

### Research Value
- 🔬 Deterministic reproducibility for academic studies
- 🔬 Comprehensive ground truth logging for analysis
- 🔬 Configurable difficulty scaling for systematic evaluation
- 🔬 Multi-dimensional scoring (network + cognitive + communication)

This phased approach ensures we build a robust, production-ready system while maintaining the innovative research value of the original PHANTOM concept. Each phase delivers a working system with incremental enhancements, enabling continuous testing and validation throughout development.