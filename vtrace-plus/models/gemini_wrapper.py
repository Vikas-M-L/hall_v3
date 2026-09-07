"""Gemini API backend — real VLM responses with zero GPU.

Uses plain HTTPS (proven against transient SDK hangs) with the key from
`GEMINI_API_KEY` (vtrace-plus/.env, gitignored). Default model
`gemini-3-flash-preview` — verified working for text + vision 2026-09-04;
2.x models are gated for new keys, so do not default to them.

Provides the three VLM services the pipeline needs:
  describe(image, prompt)  -> response text (replaces local VLM generation)
  decompose(response)      -> atomic claims (replaces sentence-split fallback)
  sample(image, prompt, n, temperature) -> diverse responses (self-consistency)

No logprobs exist on this API, so confidence stays sample-based, not
logprob-based. Every failure degrades loudly (raise), never silently.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.1-flash-lite"  # stable + clean JSON; preview 503s often
FALLBACK_MODEL = "gemini-3-flash-preview"


def _key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(str(__import__("pathlib").Path(__file__).resolve().parent.parent / ".env"))
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set (vtrace-plus/.env)")
    return key


def _post(model: str, parts: list[dict], temperature: float = 0.0,
          max_tokens: int = 512, timeout: int = 120) -> str:
    import time as _time
    import urllib.error

    body = json.dumps({"contents": [{"parts": parts}],
                       "generationConfig": {"temperature": temperature,
                                            "maxOutputTokens": max_tokens}}).encode()
    last = None
    for attempt, wait in enumerate([0, 15, 45]):
        if wait:
            logger.warning("gemini retry %d after %ds", attempt, wait)
            _time.sleep(wait)
        req = urllib.request.Request(
            f"{API_BASE}/{model}:generateContent?key={_key()}",
            data=body, headers={"Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=timeout))
            return d["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503):
                raise
            logger.warning("gemini %s, will retry", exc.code)
    raise last


def _img_part(image) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return {"inline_data": {"mime_type": "image/jpeg",
                            "data": base64.b64encode(buf.getvalue()).decode()}}


class GeminiWrapper:
    """Real-VLM backend. Construct once; methods raise on API failure."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        _key()  # fail fast when the key is missing

    def _call(self, parts: list[dict], **kw) -> str:
        import urllib.error

        try:
            return _post(self.model, parts, **kw)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503) or self.model == FALLBACK_MODEL:
                raise
            logger.warning("gemini %s overloaded, falling back to %s",
                           self.model, FALLBACK_MODEL)
            return _post(FALLBACK_MODEL, parts, **kw)

    def describe(self, image, prompt: str = "Describe this image in detail.",
                 temperature: float = 0.0) -> str:
        return self._call([{"text": prompt}, _img_part(image)],
                          temperature=temperature).strip()

    def sample(self, image, prompt: str, n: int = 5,
               temperature: float = 0.7) -> list[str]:
        parts = [{"text": prompt}, _img_part(image)]
        return [self._call(parts, temperature=temperature).strip()
                for _ in range(n)]

    def decompose(self, response: str) -> list[dict]:
        from models.claim_extractor import DECOMPOSE_PROMPT, ClaimExtractor

        raw = self._call([{"text": DECOMPOSE_PROMPT.format(response=response.strip())}],
                         max_tokens=1024)
        entries = ClaimExtractor._parse_json_array(raw)
        if not entries:
            logger.warning("gemini decomposition unparseable; caller falls back")
        return entries

    def verify(self, image, claim: str) -> dict:
        """Direct VQA cross-check (cascade stage-2): is the claim true of the
        image? Forced YES/NO first token, parsed robustly. Returns
        {supported: True/False/None, raw}. None = unparseable, never a verdict.
        Costs one API call — reserve for ambiguous claims."""
        prompt = ("Look at the image. Is the following claim TRUE of the image? "
                  "Answer with exactly one word first: YES or NO. Then one short "
                  "sentence of reason.\nClaim: " + claim.strip())
        raw = self._call([{"text": prompt}, _img_part(image)],
                         temperature=0.0, max_tokens=128).strip()
        first = (raw.split() or [""])[0].strip(".,:;\"'").upper()
        if first.startswith("YES"):
            return {"supported": True, "raw": raw}
        if first.startswith("NO"):
            return {"supported": False, "raw": raw}
        logger.warning("gemini verify unparseable: %r", raw[:80])
        return {"supported": None, "raw": raw}
