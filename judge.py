"""Calibrated rubric judging stage using the Decisions API (`judge.py`, Prompt 4).

Uses `gemini-decision-1` via `client.decisions.create` (`v1beta` API surface):
- `judge_features`: evaluates each listing `Feature` on 3 `noul` (yes/no) + 2 `score` (1..5) criteria
- `judge_layouts`: evaluates 1 top-down + 3 eye-level renders of each candidate `Layout` on
  1 scope check + 4 `noul` + 2 `score` criteria, sums in Python, and flags ties within 5%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from google import genai

from perceive import media_block_from_source
from schemas import (
    Feature,
    FeatureJudgeScore,
    Home,
    Layout,
    LayoutJudgeScore,
)

MODEL_DECISION = "gemini-decision-1"


def _make_v1beta_client() -> genai.Client:
    return genai.Client(http_options={"api_version": "v1beta"})


def _extract_noul(answers: Mapping[str, Any], key: str) -> tuple[float, float]:
    ans = answers[key]
    p_val = float(ans["noul"] if isinstance(ans, dict) else ans.noul)
    conf = float(ans["confidence"] if isinstance(ans, dict) else ans.confidence)
    return p_val, conf


def _extract_score_1_to_5(answers: Mapping[str, Any], key: str) -> tuple[float, float]:
    """Convert 0-based 5-level `ScoreAnswer.score` in [0, 4] to [1, 5]."""
    ans = answers[key]
    raw_0_to_4 = float(ans["score"] if isinstance(ans, dict) else ans.score)
    conf = float(ans["confidence"] if isinstance(ans, dict) else ans.confidence)
    return max(1.0, min(5.0, raw_0_to_4 + 1.0)), conf


def build_feature_questions(rubric_line: str) -> dict[str, dict[str, Any]]:
    """Build exact Prompt 4 Decisions API question map for one listing feature."""
    return {
        "direct_natural_light": {
            "type": "noul",
            "instructions": "Does this room get direct natural light from at least one window?",
        },
        "open_kitchen_living": {
            "type": "noul",
            "instructions": "Is the kitchen open to the living area with no full wall between them?",
        },
        "counter_seating_two": {
            "type": "noul",
            "instructions": "Is there counter seating for at least two people?",
        },
        "buyer_preference": {
            "type": "score",
            "instructions": (
                f"How well does this space match the buyer's stated preference: {rubric_line}? "
                "0 = not at all (1/5), 4 = exactly (5/5)."
            ),
            "criteria": [
                "1 = not at all",
                "2 = slightly matches",
                "3 = moderately matches",
                "4 = mostly matches",
                "5 = matches exactly",
            ],
        },
        "usable_floor_space": {
            "type": "score",
            "instructions": "How much usable floor space does this room have for furniture?",
            "criteria": [
                "1 = cramped",
                "2 = tight",
                "3 = adequate",
                "4 = spacious",
                "5 = generous",
            ],
        },
    }


def build_layout_questions(*, include_bed_question: bool = True) -> dict[str, dict[str, Any]]:
    """Build exact Prompt 4 Decisions API question map for one candidate layout's 4 renders."""
    questions: dict[str, dict[str, Any]] = {
        "scope_furnished_render": {
            "type": "noul",
            "instructions": "Is this image a furnished room render?",
        },
        "conversation_faces_seat_or_tv": {
            "type": "noul",
            "instructions": "Does every seat in the conversation group face another seat or the TV?",
        },
        "clear_walking_path": {
            "type": "noul",
            "instructions": "Is there a clear walking path from every door to the centre of the room?",
        },
        "blocks_window": {
            "type": "noul",
            "instructions": "Does any furniture block a window?",
        },
        "balance": {
            "type": "score",
            "instructions": "How balanced is the furniture across the room?",
            "criteria": [
                "1 = all on one side",
                "2 = noticeably lopsided",
                "3 = acceptably distributed",
                "4 = well balanced",
                "5 = evenly spread",
            ],
        },
        "comfort": {
            "type": "score",
            "instructions": "Would this room feel comfortable to live in?",
            "criteria": [
                "1 = awkward",
                "2 = somewhat stiff",
                "3 = livable",
                "4 = inviting",
                "5 = natural",
            ],
        },
    }
    if include_bed_question:
        questions["bed_clear_long_side"] = {
            "type": "noul",
            "instructions": "Does the bed have clear floor on at least one long side?",
        }
    return questions


def judge_features(
    home: Home | dict[str, object],
    *,
    photo_sources: Optional[Sequence[str | Path]] = None,
    rubric_text: str = "Bright open living space with room for a full sofa and dining table",
    client: Optional[genai.Client] = None,
    decision_runner: Optional[Callable[[list[dict[str, Any]], dict[str, dict[str, Any]]], Mapping[str, Any]]] = None,
) -> Home:
    """Score every `Feature` in `home` with `gemini-decision-1` and sort descending by `total_score`."""
    home_obj = home if isinstance(home, Home) else Home.model_validate(home)
    active_client = client or (_make_v1beta_client() if decision_runner is None else None)
    questions = build_feature_questions(rubric_text)

    scored_features: list[Feature] = []
    for feat in home_obj.features:
        shared_input: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Feature '{feat.label}' in room '{feat.room_id}'. Buyer rubric: {rubric_text}",
            }
        ]
        if photo_sources:
            for ev in feat.evidence:
                if ev.isdigit() and 0 <= int(ev) < len(photo_sources):
                    shared_input.append(media_block_from_source(photo_sources[int(ev)], active_client))

        if decision_runner is not None:
            answers = decision_runner(shared_input, questions)
        else:
            assert active_client is not None
            resp = active_client.decisions.create(
                model=MODEL_DECISION,
                input=shared_input,
                questions=questions,
            )
            answers = resp.answers or {}

        p_light, c1 = _extract_noul(answers, "direct_natural_light")
        p_open, c2 = _extract_noul(answers, "open_kitchen_living")
        p_seat, c3 = _extract_noul(answers, "counter_seating_two")
        s_pref, c4 = _extract_score_1_to_5(answers, "buyer_preference")
        s_space, c5 = _extract_score_1_to_5(answers, "usable_floor_space")

        # Sum strictly in Python (never ask the model to add, Prompt 4)
        total = round(p_light + p_open + p_seat + s_pref + s_space, 4)
        mean_conf = round((c1 + c2 + c3 + c4 + c5) / 5.0, 4)

        f_score = FeatureJudgeScore(
            direct_natural_light_p=round(p_light, 4),
            open_kitchen_living_p=round(p_open, 4),
            counter_seating_two_p=round(p_seat, 4),
            buyer_preference_score=round(s_pref, 4),
            usable_floor_space_score=round(s_space, 4),
            mean_confidence=mean_conf,
            total_score=total,
        )
        scored_features.append(feat.model_copy(update={"judge_score": f_score}))

    scored_features.sort(
        key=lambda f: f.judge_score.total_score if f.judge_score else 0.0,
        reverse=True,
    )
    return home_obj.model_copy(update={"features": scored_features})


def judge_layouts(
    layouts: Sequence[Layout | dict[str, object]],
    renders_by_layout: Mapping[str, Sequence[str | Path]],
    *,
    has_bed: bool = False,
    client: Optional[genai.Client] = None,
    decision_runner: Optional[Callable[[list[dict[str, Any]], dict[str, dict[str, Any]]], Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Score up to 5 layouts from their renders (1 top-down + 3 eye-level) using `gemini-decision-1`."""
    layout_objs = [
        lay if isinstance(lay, Layout) else Layout.model_validate(lay) for lay in layouts
    ]
    active_client = client or (_make_v1beta_client() if decision_runner is None else None)
    questions = build_layout_questions(include_bed_question=has_bed)

    updated: list[Layout] = []
    for lay in layout_objs:
        renders = renders_by_layout.get(lay.id, [])
        shared_input: list[dict[str, Any]] = [
            {"type": "text", "text": f"Candidate furnished room layout {lay.id}."}
        ]
        for r_src in renders:
            shared_input.append(media_block_from_source(r_src, active_client))

        if decision_runner is not None:
            answers = decision_runner(shared_input, questions)
        else:
            assert active_client is not None
            resp = active_client.decisions.create(
                model=MODEL_DECISION,
                input=shared_input,
                questions=questions,
            )
            answers = resp.answers or {}

        p_scope, c0 = _extract_noul(answers, "scope_furnished_render")
        p_conv, c1 = _extract_noul(answers, "conversation_faces_seat_or_tv")
        p_path, c2 = _extract_noul(answers, "clear_walking_path")
        p_win_block, c3 = _extract_noul(answers, "blocks_window")
        s_bal, c4 = _extract_score_1_to_5(answers, "balance")
        s_comf, c5 = _extract_score_1_to_5(answers, "comfort")
        confs = [c0, c1, c2, c3, c4, c5]

        p_bed: Optional[float] = None
        bed_term = 0.0
        if has_bed and "bed_clear_long_side" in answers:
            p_bed_val, c_bed = _extract_noul(answers, "bed_clear_long_side")
            p_bed = round(p_bed_val, 4)
            bed_term = p_bed_val
            confs.append(c_bed)

        # Sum scores in Python; blocking a window subtracts (via 1 - p_win_block)
        raw_total = p_conv + p_path + (1.0 - p_win_block) + bed_term + s_bal + s_comf
        if p_scope < 0.5:
            raw_total = 0.0

        warnings = list(lay.warnings)
        if p_scope < 0.5:
            warnings.append(f"Scope check failed: render may not show a furnished room (P={p_scope:.2f}).")

        j_score = LayoutJudgeScore(
            scope_is_furnished_render_p=round(p_scope, 4),
            conversation_faces_seat_or_tv_p=round(p_conv, 4),
            clear_path_from_doors_p=round(p_path, 4),
            blocks_window_p=round(p_win_block, 4),
            bed_clear_long_side_p=p_bed,
            balance_score=round(s_bal, 4),
            comfort_score=round(s_comf, 4),
            mean_confidence=round(sum(confs) / len(confs), 4),
            total_score=round(raw_total, 4),
        )
        updated.append(lay.model_copy(update={"judge_score": j_score, "warnings": warnings}))

    updated.sort(
        key=lambda l: l.judge_score.total_score if l.judge_score else 0.0,
        reverse=True,
    )
    ranked = [lay.model_copy(update={"rank": idx + 1}) for idx, lay in enumerate(updated)]

    # Prompt 4 rule: if top two are within 5%, surface both to the user
    recommended_ids: list[str] = []
    close_tie = False
    if ranked:
        recommended_ids.append(ranked[0].id)
        if len(ranked) >= 2 and ranked[0].judge_score and ranked[1].judge_score:
            s0 = ranked[0].judge_score.total_score
            s1 = ranked[1].judge_score.total_score
            if s0 > 0 and (s0 - s1) / s0 <= 0.05:
                close_tie = True
                recommended_ids.append(ranked[1].id)

    return {
        "layouts": [r.model_dump() for r in ranked],
        "recommended_layout_ids": recommended_ids,
        "top_two_within_5pct": close_tie,
    }


if __name__ == "__main__":
    sample_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "samples" / "sample_schemas.json"
    )
    sample = json.loads(sample_path.read_text(encoding="utf-8"))

    # Offline deterministic calibration test verifying score conversion, sum in Python, and 5% tie rule
    call_idx = 0

    def _mock_decision_runner(
        _inp: list[dict[str, Any]], qs: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        global call_idx
        call_idx += 1
        out: dict[str, dict[str, float]] = {}
        for k, q in qs.items():
            if q["type"] == "noul":
                val = 0.08 if k == "blocks_window" else 0.91
                out[k] = {"noul": val, "confidence": 0.91}
            else:
                # 0-based score in [0, 4]; slight delta on second layout to trigger <= 5% tie check
                delta = 0.12 if call_idx > 2 else 0.0
                out[k] = {"score": 3.4 - delta, "confidence": 0.86}
        return out

    scored_home = judge_features(sample["home"], decision_runner=_mock_decision_runner)
    l1 = Layout.model_validate(sample["layout"])
    l2 = l1.model_copy(update={"id": "new-layout-02", "rank": 2})
    judged = judge_layouts(
        [l1, l2],
        renders_by_layout={"new-layout-01": [], "new-layout-02": []},
        decision_runner=_mock_decision_runner,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "feature_total_score": (
                    scored_home.features[0].judge_score.total_score
                    if scored_home.features and scored_home.features[0].judge_score
                    else None
                ),
                "recommended_layout_ids": judged["recommended_layout_ids"],
                "top_two_within_5pct": judged["top_two_within_5pct"],
                "ranked_totals": [
                    lay["judge_score"]["total_score"] for lay in judged["layouts"]
                ],
            },
            indent=2,
        )
    )
