"""
Decompose a VLM response into atomic, independently-checkable claims.

Each claim carries the character span of the response it came from, which is how
the fusion layer pulls matching token log-probs out of the original generation.

LLMs will not reliably emit character offsets, so we ask for a verbatim source
substring and locate that ourselves, falling back to fuzzy matching when the
model paraphrases anyway.
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

# Ordered — first match wins, so more specific patterns come first.
TYPE_RULES: list[tuple[str, str]] = [
    ("ocr", r"\b(reads?|says?|written|text|sign|label|logo|caption|spelled)\b|[\"']"),
    ("count", r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b"
              r"|\b(several|many|few|multiple|pair|dozen)\b"),
    ("spatial", r"\b(left|right|above|below|behind|in front|top|bottom|centre|center"
                r"|beside|corner|background|foreground)\b"),
    ("relation", r"\b(holding|wearing|carrying|next to|near|attached|connected"
                 r"|on top of|leaning|touching|riding)\b"),
    ("action", r"\b(walking|running|sitting|standing|jumping|eating|drinking|playing"
               r"|looking|driving|flying|smiling|talking)\b"),
    ("attribute", r"\b(red|blue|green|yellow|black|white|brown|orange|purple|pink|grey"
                  r"|gray|silver|golden|wooden|metal|plastic|glass|large|small|tall|short"
                  r"|old|new|open|closed|empty|full|bright|dark|striped|round|square)\b"),
    ("scene", r"\b(indoor|outdoor|kitchen|beach|street|park|forest|city|room|office"
              r"|restaurant|night|daytime|sunny|rainy|snowy|weather)\b"),
]


@dataclass
class Claim:
    text: str
    source: str
    # Char span into the original response, or None if the claim could not be
    # located in it. None must propagate to a NaN confidence — see _locate.
    span: tuple[int, int] | None
    claim_type: str
    claim_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ClaimExtractor:
    def __init__(self, cfg: dict, vlm=None):
        """vlm: a VLMWrapper. Its text head is reused for decomposition."""
        self.cfg = cfg
        self.ccfg = cfg["claim_extractor"]
        self.vlm = vlm
        self.taxonomy = set(self.ccfg["type_taxonomy"])

    def _ask(self, prompt: str, max_new_tokens: int = 512) -> str:
        if self.vlm is None:
            raise RuntimeError("ClaimExtractor requires a VLMWrapper instance")
        # Text-only call: no image, so the model runs as a plain LM.
        return self.vlm.generate(
            image=None, prompt=prompt, max_new_tokens=max_new_tokens, temperature=0.0
        ).text

    # --------------------------------------------------------------- parsing

    @staticmethod
    def _parse_json_array(raw: str) -> list[dict]:
        """Tolerant extraction — models wrap arrays in prose and code fences."""
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.warning("no JSON array found in decomposition output")
            return []
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            parsed = []
            for m in re.finditer(r"\{[^{}]*\}", raw[start : end + 1]):
                try:
                    parsed.append(json.loads(m.group()))
                except json.JSONDecodeError:
                    continue
        return [p for p in parsed if isinstance(p, dict) and "claim" in p]

    @staticmethod
    def _locate(source: str, response: str) -> tuple[int, int] | None:
        """
        Find `source` in `response`; fall back to the best fuzzy window.

        Returns None when the claim cannot be located, and that is deliberate.
        The tempting fallback — span = the whole response — assigns every
        unlocated claim the mean log-prob of the entire generation: a number that
        is identical across such claims, unrelated to the claim itself, and
        indistinguishable from a real measurement once it reaches fusion. NaN is
        recoverable; a plausible fabrication is not.
        """
        if not source:
            return None

        idx = response.find(source)
        if idx != -1:
            return (idx, idx + len(source))

        # Whitespace-insensitive retry. Offsets come from the NORMALIZED string,
        # so they are only usable when normalization did not shift anything
        # before the match — otherwise they would point at the wrong tokens.
        norm_source = re.sub(r"\s+", " ", source).strip()
        norm_response = re.sub(r"\s+", " ", response)
        idx = norm_response.find(norm_source)
        if idx != -1 and norm_response[:idx] == response[:idx]:
            return (idx, min(idx + len(norm_source), len(response)))

        match = difflib.SequenceMatcher(None, response, source).find_longest_match(
            0, len(response), 0, len(source)
        )
        if match.size > max(4, len(source) // 4):
            return (match.a, match.a + match.size)

        logger.warning(
            "could not locate claim source %r in the response; confidence for this "
            "claim will be nan",
            source[:60],
        )
        return None

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

        entries: list[dict] = []
        if self.vlm is not None:
            try:
                raw = self._ask(DECOMPOSE_PROMPT.format(response=response.strip()))
                entries = self._parse_json_array(raw)
            except Exception as exc:
                logger.warning("VLM decomposition failed (%s); using sentence split", exc)
                entries = []

        if not entries:
            # Degenerate fallback. Logged loudly: a high rate here silently
            # degrades every downstream score.
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
