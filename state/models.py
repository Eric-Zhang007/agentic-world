from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WorldObject:
    object_id: str
    category: str
    position: list[float] | None = None
    rotation: list[float] | None = None
    scale: list[float] | None = None
    bbox_3d: list[float] | None = None
    appearance_embedding: list[float] | None = None
    attributes: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    confidence: float | None = None
    provenance: str = "UNKNOWN"

    def to_dict(self):
        return asdict(self)


@dataclass
class ObservationObject:
    obs_id: str
    track_id: str | None
    category: str | None
    bbox_xyxy: list[float]
    confidence: float
    frame_id: int
    timestamp: float
    source: str = "GENERATED"
    mask: Any = None
    visual_embedding: list[float] | None = None
    depth: float | None = None
    world_position: list[float] | None = None

    def to_dict(self):
        return asdict(self)
