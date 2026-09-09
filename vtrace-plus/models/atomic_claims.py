"""Conservative local fact extraction. Unhandled syntax stays explicitly compound.

No language model calls here. Never strip negation or invent implied attributes.
Exact source spans refer to the source sentence, not paraphrased claim text.
"""
from __future__ import annotations

import re

from models.claim_extractor import ClaimExtractor
from fusion.negation import has_hedge, has_negation

NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12}
COLORS = "black|white|red|blue|green|yellow|pink|purple|brown|orange|gray|grey|golden"
NOUNS = "cats?|dogs?|birds?|cups?|cars?|balls?|chairs?|bottles?|remotes?|people|persons?"
COUNT = re.compile(rf"^(?:there (?:are|is)\s+|the image (?:has|contains) (?:the )?)?"
                   rf"(?P<n>\d+|{'|'.join(NUMBERS)})\s+(?P<noun>{NOUNS})"
                   r"(?:\s+(?P<rest>.*))?$", re.I)
ACTION = re.compile(r"^(?:are|is)\s+(eating|sleeping|running|sitting|standing|playing|walking|resting|jumping)$", re.I)
WEARING = re.compile(rf"^(?P<subject>(?:the |a )?(?:boy|girl|child|man|woman|person)) "
                     rf"(?:wears|is wearing|with) (?:a |an |the )?(?P<color>{COLORS}) "
                     r"(?:(?:colour|color) )?(?P<garment>dress|jacket|shirt|coat|hat|trousers|pants)$", re.I)
COPULA = re.compile(rf"^(?P<subject>(?:the |a |an ).+?) (?:is|are) (?P<color>{COLORS})$", re.I)
TAXONOMY = ["object", "attribute", "count", "spatial", "relation", "action", "ocr", "scene"]


def extract_facts(response, entries=None):
    """Return records compatible with Claim and fusion, with decomposition metadata.

    API entries are still model predictions, not guaranteed atomic. Local scope
    is deliberately narrow; unsupported forms are routed to full-claim checking.
    """
    classifier = ClaimExtractor({"claim_extractor": {"max_claims": 40, "type_taxonomy": TAXONOMY}}, vlm=None)
    records = []

    def add(text, source, span, kind, status, label=None):
        records.append({"text": text, "claim_text": text, "source": source, "span": span,
                        "claim_type": kind, "claim_id": f"ui_c{len(records)}",
                        "detector_label": label, "decomposition": status})

    if entries is not None:
        for e in entries:
            text = str(e.get("claim", "")).strip()
            if not text:
                continue
            source = str(e.get("source", "")).strip()
            start = response.find(source) if source else -1
            add(text, source, [start, start+len(source)] if start >= 0 else None,
                classifier.classify_type(text), "api_atomic_candidate")
        if records:
            return records

    for match in re.finditer(r"[^.!?\n]+[.!?]?", response):
        source = match.group().strip()
        if not source:
            continue
        start = match.start() + len(match.group()) - len(match.group().lstrip())
        span = [start, start+len(source)]
        text = source.rstrip(".!?").strip()
        if has_negation(text) or has_hedge(text):
            add(source, source, span, classifier.classify_type(source), "requires_full_claim_check")
            continue
        counted = COUNT.fullmatch(text)
        if counted:
            raw = counted["n"].lower()
            n = int(raw) if raw.isdigit() else NUMBERS[raw]
            noun = counted["noun"].lower()
            rest = counted["rest"] or ""
            action = ACTION.fullmatch(rest)
            # Split only cases whose entire syntax is recognized, preserving
            # all clauses otherwise. This avoids silently dropping relations.
            if not rest or action:
                if n > 0:
                    add(f"{noun.capitalize()} are present.", source, span, "object", "local_rule", noun)
                add(f"There are {n} {noun}.", source, span, "count", "local_rule", noun)
                if action and n > 0:
                    add(f"The {noun} are {action[1].lower()}.", source, span, "action", "local_rule", noun)
                elif action:
                    # An action with zero entities has uncertain scope.
                    add(source, source, span, "action", "requires_full_claim_check")
                continue
        wearing = WEARING.fullmatch(text)
        if wearing:
            # Person is enough for detection; retain original subject wording
            # for verification rather than treating gender as an inferred fact.
            subject, garment, color = wearing["subject"], wearing["garment"], wearing["color"]
            add(f"{subject.capitalize()} wears a {garment}.", source, span, "attribute", "local_rule", "person")
            add(f"The {garment} worn by {subject} is {color}.", source, span, "attribute", "local_rule", "person")
            continue
        color = COPULA.fullmatch(text)
        if color:
            add(source, source, span, "attribute", "local_rule")
            continue
        add(source, source, span, classifier.classify_type(source), "requires_full_claim_check")
    return records
