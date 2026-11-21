"""Pydantic v2 schemas and validation utilities for Homeweave.

Conventions enforced across all stages:
- Units: metres (m). Right-handed axes, +y up, floor at y = 0.
- Angles: degrees clockwise from +z when viewed from above.
- IDs: prefixed with `src-` (current home), `new-` (generated home), or `lst<N>-` (listing N).
- Provenance: every object carries a `provenance` record tracing back to photo, video timestamp, or plan.
- Unknowns: unknown measurements are `None` paired with a non-empty `null_reason` string; never guessed defaults.
- Model responses: validated against these schemas; on failure, retried once, then flagged.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ID_PREFIX_RE = re.compile(r"^(src|new|lst\d+)-[a-zA-Z0-9_-]+$")
LAYOUT_ID_RE = re.compile(r"^(src|new|lst\d+|layout)-[a-zA-Z0-9_-]+$")

RoomType = Literal[
    "living",
    "kitchen",
    "dining",
    "great_room",
    "primary_bedroom",
    "bedroom",
    "bathroom",
    "primary_bathroom",
    "office",
    "den",
    "hallway",
    "entry",
    "laundry",
    "closet",
    "outdoor",
    "other",
]

GroupKind = Literal["conversation", "sleep", "dining", "work", "storage", "other"]


def _check_prefixed_id(value: str, pattern: re.Pattern[str] = ID_PREFIX_RE) -> str:
    if not pattern.match(value):
        raise ValueError(
            f"ID '{value}' must start with 'src-', 'new-', or 'lst<N>-' followed by slug characters."
        )
    return value


class Provenance(BaseModel):
    """Traces every entity or measurement back to raw media."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        description="Primary media source, e.g. 'plan', 'photo:0', 'video', 'arrange_engine', or 'lst1-kitchen'."
    )
    photo_indices: list[int] = Field(
        default_factory=list,
        description="Zero-based listing photo indices supporting this object.",
    )
    seen_at: list[str] = Field(
        default_factory=list,
        description="Video timestamps (MM:SS) or frame ranges where the object appears.",
    )
    scale_ref: Optional[str] = Field(
        default=None,
        description="Scale reference used, e.g. 'door src-door-01 = 0.81 m x 2.03 m' or 'plan_labels'.",
    )
    from_listing: Optional[str] = Field(
        default=None,
        description="Source listing ID when woven into a generated home (e.g. 'lst1').",
    )
    from_room: Optional[str] = Field(
        default=None,
        description="Source room ID when woven or mapped (e.g. 'lst1-kitchen').",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional provenance detail.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, data: object) -> object:
        if isinstance(data, str) and data.strip():
            return {"source": data.strip()}
        return data


class DimensionsM(BaseModel):
    """Axis-aligned bounding box dimensions in metres (w along local X, d along local Z, h along +Y)."""

    model_config = ConfigDict(extra="forbid")

    w: float = Field(gt=0.0, description="Width in metres.")
    d: float = Field(gt=0.0, description="Depth in metres.")
    h: float = Field(gt=0.0, description="Height in metres (+y up from floor y = 0).")


class Pose2D(BaseModel):
    """Floor-plane rigid pose in metres; rot_deg is clockwise from +z viewed from above."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(description="Centroid X coordinate in metres.")
    z: float = Field(description="Centroid Z coordinate in metres.")
    rot_deg: float = Field(
        default=0.0,
        description="Rotation in degrees clockwise from +z when viewed from above.",
    )


class Door(BaseModel):
    """Door opening along a room polygon wall segment."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Prefixed door ID, e.g. 'lst1-door-01' or 'src-door-01'.")
    wall_index: int = Field(ge=0, description="Zero-based index of the room polygon edge.")
    offset_m: float = Field(ge=0.0, description="Distance in metres from wall start vertex to door start.")
    width_m: float = Field(gt=0.0, description="Door opening width in metres (0.81 m standard fallback).")
    swing: Optional[
        Literal["inward_left", "inward_right", "outward_left", "outward_right", "sliding", "open"]
    ] = Field(default=None, description="Door swing direction, or null if unknown.")
    null_reason: Optional[str] = Field(
        default=None, description="Required reason if swing is null."
    )
    provenance: Provenance

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        return _check_prefixed_id(v)

    @model_validator(mode="after")
    def _require_null_reason(self) -> Door:
        if self.swing is None and not self.null_reason:
            raise ValueError(f"Door '{self.id}' has swing=None without a null_reason.")
        return self


class Window(BaseModel):
    """Window along a room polygon wall segment."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Prefixed window ID, e.g. 'lst1-win-01'.")
    wall_index: int = Field(ge=0, description="Zero-based index of the room polygon edge.")
    offset_m: float = Field(ge=0.0, description="Distance in metres from wall start vertex to window start.")
    width_m: float = Field(gt=0.0, description="Window width in metres.")
    sill_height_m: Optional[float] = Field(default=0.9, ge=0.0, description="Sill height above floor in metres.")
    height_m: Optional[float] = Field(default=1.2, gt=0.0, description="Window vertical height in metres.")
    provenance: Provenance

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        return _check_prefixed_id(v)


class Fixture(BaseModel):
    """Built-in architectural fixture (island, counter, fireplace, closet, tv_wall, bath_tub, etc.)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Prefixed fixture ID, e.g. 'lst1-island-01'.")
    room_id: Optional[str] = Field(default=None, description="Containing room ID.")
    type: str = Field(description="Fixture kind: counter, island, fireplace, closet, tv_wall, sink, stove, etc.")
    footprint: Optional[list[list[float]]] = Field(
        default=None,
        description="Ordered [x, z] polygon corners in metres, or null if occluded.",
    )
    null_reason: Optional[str] = Field(
        default=None,
        description="Required reason if footprint is null.",
    )
    provenance: Provenance

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        return _check_prefixed_id(v)

    @field_validator("footprint")
    @classmethod
    def _valid_footprint(cls, v: Optional[list[list[float]]]) -> Optional[list[list[float]]]:
        if v is None:
            return None
        if len(v) < 3:
            raise ValueError("Fixture footprint polygon must have at least 3 [x, z] vertices.")
        for pt in v:
            if len(pt) != 2:
                raise ValueError(f"Polygon vertex {pt} must be [x, z] of length 2.")
        return v

    @model_validator(mode="after")
    def _require_null_reason(self) -> Fixture:
        if self.footprint is None and not self.null_reason:
            raise ValueError(f"Fixture '{self.id}' has footprint=None without a null_reason.")
        return self


class Room(BaseModel):
    """Single room or open-plan functional zone on floor y = 0."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Prefixed room ID, e.g. 'lst1-kitchen', 'src-living', 'new-living'.")
    type: RoomType = Field(description="Canonical room category.")
    polygon: Optional[list[list[float]]] = Field(
        default=None,
        description="Ordered [x, z] corners in metres, or null if unreadable.",
    )
    ceiling_height_m: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Ceiling height in metres (+y), or null if unknown.",
    )
    doors: list[Door] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    fixtures: list[Fixture] = Field(default_factory=list)
    null_reason: Optional[str] = Field(
        default=None,
        description="Required reason when polygon or ceiling_height_m is null.",
    )
    provenance: Provenance

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        return _check_prefixed_id(v)

    @field_validator("polygon")
    @classmethod
    def _valid_polygon(cls, v: Optional[list[list[float]]]) -> Optional[list[list[float]]]:
        if v is None:
            return None
        if len(v) < 3:
            raise ValueError("Room polygon must have at least 3 [x, z] corners.")
        for pt in v:
            if len(pt) != 2:
                raise ValueError(f"Room polygon corner {pt} must be [x, z] of length 2.")
        return v

    @model_validator(mode="after")
    def _require_null_reason(self) -> Room:
        if (self.polygon is None or self.ceiling_height_m is None) and not self.null_reason:
            raise ValueError(
                f"Room '{self.id}' has null polygon or ceiling_height_m without a null_reason."
            )
        return self


class ScaleCheck(BaseModel):
    """Scale calibration record and plan-vs-photo conflict log."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["plan_labels", "door_width", "known_object", "unscaled"] = Field(
        description="How metric scale was established."
    )
    reference: Optional[str] = Field(
        default=None,
        description="Specific door ID or dimension label used for calibration.",
    )
    conflicts: list[str] = Field(
        default_factory=list,
        description="Room IDs where photo scale and plan labels disagree by > 10%.",
    )
    null_reason: Optional[str] = Field(
        default=None,
        description="Reason if method == 'unscaled' or reference is null.",
    )
    provenance: Provenance


class StagingItem(BaseModel):
    """Placeholder furniture visible in listing photos (replaceable staging prior)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Prefixed staging item ID, e.g. 'lst1-stage-sofa-01'.")
    room_id: str = Field(description="ID of the room containing this staging item.")
    category: str = Field(description="Furniture category: sofa, coffee_table, dining_table, bed, etc.")
    dims_m: Optional[DimensionsM] = Field(default=None)
    pose: Optional[Pose2D] = Field(default=None)
    group_kind: Optional[GroupKind] = Field(default=None)
    replaceable: bool = Field(default=True)
    null_reason: Optional[str] = Field(default=None)
    provenance: Provenance

    @field_validator("id", "room_id")
    @classmethod
    def _valid_ids(cls, v: str) -> str:
        return _check_prefixed_id(v)

    @model_validator(mode="after")
    def _require_null_reason(self) -> StagingItem:
        if (self.dims_m is None or self.pose is None) and not self.null_reason:
            raise ValueError(f"StagingItem '{self.id}' has null dims_m/pose without null_reason.")
        return self


class FeatureJudgeScore(BaseModel):
    """Decisions API (`gemini-decision-1`) calibrated scores for one listing feature."""

    model_config = ConfigDict(extra="forbid")

    direct_natural_light_p: float = Field(ge=0.0, le=1.0)
    open_kitchen_living_p: float = Field(ge=0.0, le=1.0)
    counter_seating_two_p: float = Field(ge=0.0, le=1.0)
    buyer_preference_score: float = Field(ge=1.0, le=5.0)
    usable_floor_space_score: float = Field(ge=1.0, le=5.0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(description="Sum computed in Python, never by the model.")


class Feature(BaseModel):
    """Desirable home feature extracted from a listing and scored against the buyer's rubric."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Prefixed feature ID, e.g. 'lst1-feat-01'.")
    home_id: Optional[str] = Field(default=None, description="Source listing ID, e.g. 'lst1-home'.")
    room_id: str = Field(description="Room ID this feature belongs to.")
    label: str = Field(description="Short human-readable description of the feature.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Photo indices or plan references supporting this feature.",
    )
    judge_score: Optional[FeatureJudgeScore] = Field(default=None)
    provenance: Provenance

    @field_validator("id", "room_id")
    @classmethod
    def _valid_ids(cls, v: str) -> str:
        return _check_prefixed_id(v)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v: object) -> list[str]:
        if isinstance(v, (int, str)):
            return [str(v)]
        if isinstance(v, list):
            return [str(x) for x in v]
        return []


class WeaveConflict(BaseModel):
    """Feature or room conflict logged numerically during the Weave stage rather than forced."""

    model_config = ConfigDict(extra="forbid")

    feature_id: Optional[str] = Field(default=None)
    room_id: Optional[str] = Field(default=None)
    reason: str = Field(
        description="Numeric explanation, e.g. 'lst2 kitchen island needs 4.2 m width; slot is 3.6 m'."
    )
    provenance: Provenance


class Home(BaseModel):
    """Top-level representation of a listing, current home, or woven generated home."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Prefixed home ID, e.g. 'lst1-home', 'src-home', 'new-home'.")
    source: Literal["listing", "current", "generated"]
    rooms: list[Room]
    fixtures: list[Fixture] = Field(default_factory=list)
    staging: list[StagingItem] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    conflicts: list[WeaveConflict] = Field(default_factory=list)
    scale_check: ScaleCheck
    provenance: Provenance

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        return _check_prefixed_id(v)


class FurnitureItem(BaseModel):
    """Movable piece of furniture owned by the user in the current home (`src-`)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Prefixed item ID, e.g. 'src-sofa-01'.")
    room_id: str = Field(description="Current room ID, e.g. 'src-living'.")
    category: str = Field(
        description="Category slug: sofa, sectional, armchair, coffee_table, tv_stand, rug, bed, nightstand, dining_table, chair, desk, bookshelf, etc."
    )
    dims_m: Optional[DimensionsM] = Field(
        default=None,
        description="Dimensions in metres, or null if no scale reference is visible.",
    )
    pose: Optional[Pose2D] = Field(
        default=None,
        description="Pose in the source room, or null if occluded.",
    )
    front_dir: Optional[str] = Field(
        default=None,
        description="Semantic facing description (e.g. '+z', 'toward_tv', 'toward_center'), or null.",
    )
    against_wall: bool = Field(
        default=False,
        description="Whether the item's back/side sits flush against a room wall in the source home.",
    )
    nearest_fixture: Optional[Literal["window", "door", "tv_wall", "fireplace", "closet", "island", "other"]] = Field(
        default=None,
        description="Nearest architectural fixture in the source room, if any.",
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Functional group ID, e.g. 'src-grp-conversation-01'.",
    )
    must_keep: bool = Field(default=True, description="Whether the user prioritizes keeping this item.")
    style: Optional[str] = Field(default=None, description="Optional visual style tag.")
    confidence: float = Field(ge=0.0, le=1.0, description="Perception confidence in [0, 1].")
    null_reason: Optional[str] = Field(
        default=None,
        description="Required reason when dims_m, pose, or front_dir is null.",
    )
    provenance: Provenance

    @field_validator("id", "room_id")
    @classmethod
    def _valid_ids(cls, v: str) -> str:
        return _check_prefixed_id(v)

    @model_validator(mode="after")
    def _require_null_reason(self) -> FurnitureItem:
        if (self.dims_m is None or self.pose is None) and not self.null_reason:
            raise ValueError(
                f"FurnitureItem '{self.id}' has null dims_m or pose without a null_reason."
            )
        return self


class FunctionalGroup(BaseModel):
    """Cluster of items used together (conversation, sleep, dining, work, storage, other)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Prefixed group ID, e.g. 'src-grp-conversation-01'.")
    room_id: Optional[str] = Field(default=None)
    kind: GroupKind
    item_ids: list[str] = Field(min_length=1)
    provenance: Provenance

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        return _check_prefixed_id(v)


class UncertainItem(BaseModel):
    """Ambiguous sighting or low-confidence (< 0.6) item surfaced for user confirmation."""

    model_config = ConfigDict(extra="forbid")

    item_id: Optional[str] = Field(default=None)
    confidence: float = Field(ge=0.0, le=1.0)
    question: str = Field(
        description="One-line confirmation question for the user, e.g. 'Is the grey chair at 0:42 the same one seen at 1:15?'."
    )
    provenance: Provenance


class HomeVideoInventory(BaseModel):
    """Output of `perceive_home_video` (Prompt 2)."""

    model_config = ConfigDict(extra="forbid")

    home_id: str = Field(default="src-home")
    rooms: list[Room]
    items: list[FurnitureItem]
    groups: list[FunctionalGroup] = Field(default_factory=list)
    uncertain: list[UncertainItem] = Field(default_factory=list)
    scale_check: ScaleCheck
    provenance: Provenance

    @field_validator("home_id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        return _check_prefixed_id(v)


class HardPlanWorking(BaseModel):
    """Step-by-step audit trail produced when reading low-quality/hand-drawn plans (Prompt 3)."""

    model_config = ConfigDict(extra="forbid")

    raw_dimension_labels: list[str] = Field(
        description="Step 1: every dimension label read, with its unit as written."
    )
    converted_metres: list[str] = Field(
        description="Step 2: each label converted to metres."
    )
    polygon_closure_checks: list[str] = Field(
        description="Step 3: whether each room's walls close into a polygon, and any distrusted labels."
    )
    area_check: Optional[str] = Field(
        default=None,
        description="Step 4: comparison of summed room area against any stated total area.",
    )


class HardPlanResult(BaseModel):
    """Output of `perceive_hard_plan` (Prompt 3)."""

    model_config = ConfigDict(extra="forbid")

    working: HardPlanWorking
    rooms: list[Room]
    plan_confidence: float = Field(ge=0.0, le=1.0)
    unreadable: list[str] = Field(default_factory=list)
    scale_check: ScaleCheck
    provenance: Provenance


class Placement(BaseModel):
    """Rigid furniture placement in a target room."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    category: Optional[str] = Field(default=None)
    group_id: Optional[str] = Field(default=None)
    dims_m: Optional[DimensionsM] = Field(default=None)
    room_id: Optional[str] = Field(default=None)
    room_pair: Optional[list[str]] = Field(
        default=None, description="[src_room_id, new_room_id]."
    )
    pose: Pose2D
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    matched_neighbors: list[str] = Field(default_factory=list)
    provenance: Provenance

    @field_validator("item_id")
    @classmethod
    def _valid_item_id(cls, v: str) -> str:
        return _check_prefixed_id(v)


class LeftoverItem(BaseModel):
    """Item excluded from a layout due to a hard constraint violation, with numeric detail."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    reason: str = Field(
        description="Short machine/human reason, e.g. 'needs 3.40 m; longest free wall is 3.10 m'."
    )
    suggestions: list[str] = Field(default_factory=list)
    provenance: Provenance

    @field_validator("item_id")
    @classmethod
    def _valid_item_id(cls, v: str) -> str:
        return _check_prefixed_id(v)


class CostBreakdown(BaseModel):
    """Soft-cost terms from `arrange.py`."""

    model_config = ConfigDict(extra="forbid")

    fixture_mismatch: float = 0.0
    wall_mismatch: float = 0.0
    clearance_penalty: float = 0.0
    staging_distance: float = 0.0
    group_overlap: float = 0.0
    group_distance_delta: float = 0.0
    total: float = 0.0


class CheckResult(BaseModel):
    """Deterministic geometry checker metrics from `check.py`."""

    model_config = ConfigDict(extra="forbid")

    overlaps: list[list[str]] = Field(
        default_factory=list, description="Pairs of intersecting item IDs."
    )
    door_blocks: list[str] = Field(
        default_factory=list, description="Item IDs intersecting any door swing arc."
    )
    clearance_fails: int = Field(
        default=0, ge=0, description="Unreachable door-to-door or door-to-centroid A* paths at 0.9 m width."
    )
    clearance_fail_paths: list[str] = Field(default_factory=list)
    outside_room: list[str] = Field(
        default_factory=list, description="Item IDs whose footprint is not fully inside the room polygon."
    )
    relations_kept: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Share of same-group pairs within 0.3 m distance delta and 30 deg relative facing.",
    )
    violations_total: int = Field(default=0, ge=0)
    flagged_items: list[str] = Field(default_factory=list)


class LayoutJudgeScore(BaseModel):
    """Decisions API (`gemini-decision-1`) calibrated scores on 4 layout renders (Prompt 4)."""

    model_config = ConfigDict(extra="forbid")

    scope_is_furnished_render_p: float = Field(ge=0.0, le=1.0)
    conversation_faces_seat_or_tv_p: float = Field(ge=0.0, le=1.0)
    clear_path_from_doors_p: float = Field(ge=0.0, le=1.0)
    blocks_window_p: float = Field(ge=0.0, le=1.0)
    bed_clear_long_side_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    balance_score: float = Field(ge=1.0, le=5.0)
    comfort_score: float = Field(ge=1.0, le=5.0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(description="Sum computed in Python, never by the model.")


class Layout(BaseModel):
    """Candidate furnished layout ranked by `arrange.py` and optionally `judge_layouts`."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Layout ID, e.g. 'new-layout-01' or 'layout-01'.")
    rank: int = Field(ge=1)
    room_pair: Optional[list[str]] = Field(default=None)
    placements: list[Placement]
    costs: CostBreakdown = Field(default_factory=CostBreakdown)
    check: Optional[CheckResult] = Field(default=None)
    judge_score: Optional[LayoutJudgeScore] = Field(default=None)
    warnings: list[str] = Field(default_factory=list)
    leftovers: list[LeftoverItem] = Field(default_factory=list)
    provenance: Provenance

    @field_validator("id")
    @classmethod
    def _valid_layout_id(cls, v: str) -> str:
        return _check_prefixed_id(v, LAYOUT_ID_RE)


class BaselinePlacementItem(BaseModel):
    """Minimal placement entry returned by the Gemini-only baseline (Prompt 7)."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    pose: Pose2D


class BaselineLeftoverItem(BaseModel):
    """Minimal leftover entry returned by the Gemini-only baseline (Prompt 7)."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    reason: str


class BaselineLayoutResponse(BaseModel):
    """Exact JSON schema requested from `gemini-4-neon` in Prompt 7."""

    model_config = ConfigDict(extra="forbid")

    placements: list[BaselinePlacementItem]
    leftovers: list[BaselineLeftoverItem] = Field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


class ValidatedStageResult(BaseModel, Generic[T]):
    """Envelope enforcing Prompt 0's 'validate -> retry once -> flag' contract."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    flagged: bool = False
    attempts: int = Field(ge=1, le=2)
    data: Optional[T] = None
    errors: list[str] = Field(default_factory=list)


def to_gemini_response_format(schema_cls: type[BaseModel]) -> dict[str, object]:
    """Build the `response_format` dict for `client.interactions.create` (google-genai >= 3.2.0)."""
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema_cls.model_json_schema(),
    }


def validate_with_retry(
    producer: Callable[[Optional[str]], str | dict[str, object]],
    schema_cls: type[T],
) -> ValidatedStageResult[T]:
    """Call `producer(previous_error)` up to twice; validate against `schema_cls` or flag."""
    errors: list[str] = []
    last_err: Optional[str] = None
    for attempt in (1, 2):
        try:
            raw = producer(last_err)
            if isinstance(raw, str):
                parsed = schema_cls.model_validate_json(raw)
            else:
                parsed = schema_cls.model_validate(raw)
            return ValidatedStageResult[T](
                ok=True,
                flagged=False,
                attempts=attempt,
                data=parsed,
                errors=errors,
            )
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_err = str(exc)
            errors.append(f"attempt {attempt}: {last_err}")
    return ValidatedStageResult[T](
        ok=False,
        flagged=True,
        attempts=2,
        data=None,
        errors=errors,
    )


if __name__ == "__main__":
    sample_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "samples" / "sample_schemas.json"
    )
    payload = json.loads(sample_path.read_text(encoding="utf-8"))

    home = Home.model_validate(payload["home"])
    inventory = HomeVideoInventory.model_validate(payload["inventory"])
    layout = Layout.model_validate(payload["layout"])

    # Verify retry-once-then-flag contract:
    calls = 0

    def _flaky_producer(prev_err: Optional[str]) -> dict[str, object]:
        global calls
        calls += 1
        if prev_err is None:
            # First call omits null_reason on an item with unknown dims_m -> fails validation
            bad = json.loads(json.dumps(payload["inventory"]))
            bad["items"][0]["dims_m"] = None
            bad["items"][0]["null_reason"] = None
            return bad
        return payload["inventory"]

    recovered = validate_with_retry(_flaky_producer, HomeVideoInventory)
    assert recovered.ok and recovered.attempts == 2 and not recovered.flagged

    flagged = validate_with_retry(lambda _err: '{"broken": true}', Home)
    assert not flagged.ok and flagged.flagged and flagged.attempts == 2

    rf = to_gemini_response_format(Home)
    assert rf["type"] == "text" and rf["mime_type"] == "application/json"

    print(
        json.dumps(
            {
                "status": "ok",
                "home_id": home.id,
                "rooms": [r.id for r in home.rooms],
                "inventory_items": [i.id for i in inventory.items],
                "layout_id": layout.id,
                "placements": len(layout.placements),
                "leftovers": len(layout.leftovers),
                "retry_recovery_attempts": recovered.attempts,
                "flagged_after_two_failures": flagged.flagged,
            },
            indent=2,
        )
    )
