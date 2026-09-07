"""
Decompose a VLM response into atomic, independently-checkable claims.

Each claim carries:
  text        - the atomic claim, rewritten to stand alone
  source      - the verbatim substring of the response it came from
  span        - (char_start, char_end) into the original response
  claim_type  - object / attribute / count / spatial / relation / action / ocr / scene

The span matters: it is how signal extraction pulls the matching token log-probs
out of the original generation. Since LLMs will not reliably emit character
offsets, we ask for a verbatim source substring and locate that ourselves,
falling back to fuzzy matching when the model paraphrases anyway.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """Break the following image description into atomic factual claims.

Rules:
- Each claim must be independently checkable against the image on its own.
- Each claim must be self-contained: replace pronouns with the noun they refer to.
- Do not add information that is not in the description.
- Split conjunctions ("a red car and a blue bike") into separate claims.
- Ignore hedges, opinions, and meta-commentary ("it appears that", "this is a nice photo").

Return ONLY a JSON array. Each element must have exactly two keys:
  "claim":  the standalone claim
  "source": the VERBATIM substring of the description this came from, copied exactly

Description:
\"\"\"{response}\"\"\"

JSON:"""

TYPE_PROMPT = """Classify this claim about an image into exactly one category.

Categories:
- object: something exists in the image
- attribute: a property of something (colour, material, size, state)
- count: a quantity
- spatial: position or location ("on the left", "behind")
- relation: how two things relate ("holding", "next to")
- action: something being done
- ocr: text visible in the image
- scene: overall setting, place, or time of day

Claim: {claim}
Answer with one word only:"""

# Ordered: first match wins, so more specific patterns come first.
TYPE_RULES: list[tuple[str, str]] = [
    ("ocr", r"\b(reads?|says?|written|text|sign|label|logo|caption|spelled)\b|[\"']"),
    ("count", r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b|\b(several|many|few|multiple|pair|dozen)\b"),
    ("spatial", r"\b(left|right|above|below|behind|in front|top|bottom|centre|center|beside|corner|background|foreground)\b"),
    ("relation", r"\b(holding|wearing|carrying|next to|near|attached|connected|on top of|leaning|touching|riding)\b"),
    ("action", r"\b(walking|running|sitting|standing|jumping|eating|drinking|playing|looking|driving|flying|smiling|talking)\b"),
    ("attribute", r"\b(red|blue|green|yellow|black|white|brown|orange|purple|pink|grey|gray|silver|golden|wooden|metal|plastic|glass|large|small|tall|short|old|new|open|closed|empty|full|bright|dark|striped|round|square)\b"),
    ("scene", r"\b(indoor|outdoor|kitchen|beach|street|park|forest|city|room|office|restaurant|night|daytime|sunny|rainy|snowy|weather)\b"),
]


@dataclass
class Claim:
    text: str
    source: str
    span: tuple[int, int]
    claim_type: str
    label: int | None = None  # 1 = hallucinated, 0 = supported, None = unlabelled
    claim_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ClaimExtractor:
    def __init__(self, cfg: dict, vlm=None, llm=None):
        """
        vlm: a VLMWrapper, used when backend == "vlm" (reuses the loaded model's text head)
        llm: an optional separate HF text model, used when backend == "hf_llm"
        """
        self.cfg = cfg
        self.ccfg = cfg["claim_extractor"]
        self.vlm = vlm
        self.llm = llm
        self.taxonomy = set(self.ccfg["type_taxonomy"])

    # ------------------------------------------------------------- text calls

    def _ask(self, prompt: str, max_new_tokens: int = 512) -> str:
        backend = self.ccfg.get("backend", "vlm")
        if backend == "vlm":
            if self.vlm is None:
                raise RuntimeError("backend='vlm' requires a VLMWrapper instance")
            # Text-only call: no image passed, so the model runs as a plain LM.
            return self.vlm.generate(
                image=None, prompt=prompt, max_new_tokens=max_new_tokens, temperature=0.0
            ).text
        if backend == "hf_llm":
            if self.llm is None:
                raise RuntimeError("backend='hf_llm' requires an llm instance")
            return self.llm(prompt, max_new_tokens=max_new_tokens)
        raise ValueError(f"unknown claim_extractor backend {backend}")

    # --------------------------------------------------------------- parsing

    @staticmethod
    def _parse_json_array(raw: str) -> list[dict]:
        """Tolerant JSON extraction — models wrap arrays in prose and code fences."""
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.warning("no JSON array found in decomposition output")
            return []
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            # last resort: pull out individual {...} objects
            parsed = []
            for m in re.finditer(r"\{[^{}]*\}", raw[start : end + 1]):
                try:
                    parsed.append(json.loads(m.group()))
                except json.JSONDecodeError:
                    continue
        return [p for p in parsed if isinstance(p, dict) and "claim" in p]

    @staticmethod
    def _locate(source: str, response: str) -> tuple[int, int]:
        """Find `source` in `response`; fall back to best fuzzy window."""
        if not source:
            return (0, len(response))
        idx = response.find(source)
        if idx != -1:
            return (idx, idx + len(source))

        # normalized retry
        norm = re.sub(r"\s+", " ", source).strip()
        idx = re.sub(r"\s+", " ", response).find(norm)
        if idx != -1:
            return (idx, min(idx + len(norm), len(response)))

        matcher = difflib.SequenceMatcher(None, response, source)
        match = matcher.find_longest_match(0, len(response), 0, len(source))
        if match.size > max(4, len(source) // 4):
            return (match.a, match.a + match.size)

        logger.debug("could not locate source %r; defaulting to whole response", source[:40])
        return (0, len(response))

    # ---------------------------------------------------------------- typing

    def classify_type(self, claim_text: str) -> str:
        lowered = claim_text.lower()
        for ctype, pattern in TYPE_RULES:
            if ctype in self.taxonomy and re.search(pattern, lowered):
                return ctype
        try:
            guess = self._ask(TYPE_PROMPT.format(claim=claim_text), max_new_tokens=8)
            guess = guess.strip().lower().split()[0].strip(".,:;")
            if guess in self.taxonomy:
                return guess
        except Exception as exc:
            logger.debug("type LLM fallback failed: %s", exc)
        return "object"

    # ------------------------------------------------------------------ main

    def extract(self, response: str, response_id: str = "r") -> list[Claim]:
        if not response or not response.strip():
            return []

        raw = self._ask(DECOMPOSE_PROMPT.format(response=response.strip()))
        entries = self._parse_json_array(raw)

        if not entries:
            # Degenerate fallback: sentence split. Logged loudly because a high
            # rate here is a hidden confound in every downstream number.
            logger.warning("decomposition failed; falling back to sentence split")
            entries = [
                {"claim": s.strip(), "source": s.strip()}
                for s in re.split(r"(?<=[.!?])\s+", response)
                if len(s.strip()) > 3
            ]

        claims: list[Claim] = []
        for i, entry in enumerate(entries[: self.ccfg["max_claims"]]):
            text = str(entry.get("claim", "")).strip()
            if not text:
                continue
            source = str(entry.get("source", text)).strip()
            claims.append(
                Claim(
                    text=text,
                    source=source,
                    span=self._locate(source, response),
                    claim_type=self.classify_type(text),
                    claim_id=f"{response_id}_c{i}",
                )
            )
        return claims
