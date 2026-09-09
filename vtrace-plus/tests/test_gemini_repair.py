import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.gemini_wrapper import GeminiWrapper


@pytest.mark.parametrize("raw", ['{"supported":true,"answers_question":false}',
                                  '```json\n{"supported":true,"answers_question":false}\n```'])
def test_relevance_rejection_parsed(monkeypatch, raw):
    w = GeminiWrapper.__new__(GeminiWrapper)
    monkeypatch.setattr(w, "_call", lambda *a, **k: raw)
    assert w.verify_answer(Image.new("RGB", (2,2)), "cats", "How many?")["answers_question"] is False


def test_malformed_and_nonboolean_verification_never_accepted(monkeypatch):
    w = GeminiWrapper.__new__(GeminiWrapper)
    for raw in ('yes', '{"supported":"true","answers_question":true}', '[]'):
        monkeypatch.setattr(w, "_call", lambda *a, **k: raw)
        result = w.verify_answer(Image.new("RGB", (2,2)), "cats", "How many?")
        assert result["supported"] is None and result["answers_question"] is None


def test_words_starting_with_no_are_not_refutations(monkeypatch):
    w = GeminiWrapper.__new__(GeminiWrapper)
    monkeypatch.setattr(w, "_call", lambda *a, **k: "Not enough evidence to decide.")
    assert w.verify(Image.new("RGB", (2,2)), "cats")["supported"] is None
