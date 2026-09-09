import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.atomic_claims import extract_facts
from models.boxes import clip_box, suppress
from models.owl_wrapper import parse_count


def test_count_action_split_preserves_source():
    source = "5 cats are eating"
    rows = extract_facts(source)
    assert [r["claim_type"] for r in rows] == ["object", "count", "action"]
    assert rows[1]["text"] == "There are 5 cats."
    assert all(source[slice(*r["span"])] == r["source"] for r in rows)
    assert parse_count(source) == (5, "cats")


def test_clothing_has_separate_type_and_color():
    rows = extract_facts("the boy with the black colour dress")
    assert len(rows) == 2 and all(r["claim_type"] == "attribute" for r in rows)
    assert rows[0]["text"] == "The boy wears a dress."
    assert "black" in rows[1]["text"]


def test_negative_hedged_and_complex_syntax_not_misdecomposed():
    for text in ("There are no cats.", "Maybe two cats are eating.", "Two cats are eating near a red ball."):
        rows = extract_facts(text)
        assert len(rows) == 1 and rows[0]["text"] == text
        assert rows[0]["decomposition"] == "requires_full_claim_check"


def test_nms_keeps_distinct_objects_removes_duplicates():
    selected = suppress([[0,0,30,30], [1,1,29,29], [50,0,80,30]], [.9,.8,.7], 100,100)
    assert len(selected) == 2
    assert selected[0][0] == (0.,0.,30.,30.)


def test_invalid_and_outside_boxes_clipped():
    assert clip_box([-10,-10,20,20], 10,10) == (0.,0.,10,10)
    assert clip_box([10,10,1,1], 20,20) is None
    assert clip_box([float("nan"),0,1,1],10,10) is None
    assert suppress([[0,0,10,10]], [float("nan")], 20,20) == []
