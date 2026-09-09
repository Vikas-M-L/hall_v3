"""Recorded, replayable model calls. Gold labels never enter this layer."""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from research.core import canonical_hash, read_jsonl, write_jsonl
from research.preservation import ALIGNMENT_PROMPT

PROMPTS = {
    "preservation": ALIGNMENT_PROMPT,
    "generate": 'Answer the user question using only visible image evidence. Return JSON {"response":"answer"}. State uncertainty when the image is insufficient.',
    "detect": "Inspect the image, user question and original response. Return JSON: "
              '{"claims":[{"text":"atomic assertion", "source":"exact response substring", '
              '"type":"object|attribute|relation|counting|spatial|ocr|reasoning|knowledge|unsupported_inference|grounding|unknown", '
              '"risk":0.5,"verdict":"supported|contradicted|unresolved","reason":"visual evidence or limitation"}]}. '
              "Risk is your uncertainty estimate, not a ground-truth probability. Do not treat the user's assertions as facts.",
    "repair": "Reinspect the image and correct the response to the original question. "
              "Use the specified action to focus verification; supplied evidence is fallible. "
              'Return JSON {"response":"complete corrected answer"}. Preserve supported details; do not add guesses.',
    "verify": "Independently inspect the image, question and candidate response. "
              'Return JSON {"supported":true, "answers_question":true, "reason":"explanation"}. '
              "Use false when contradicted or incomplete, null when unverifiable. "
              "Do not approve omission of information required by the question."
}


class ReplayBackend:
    def __init__(self, path):
        self.records = {}
        self.events = []
        for r in read_jsonl(path):
            if r.get("key") in self.records:
                raise ValueError("duplicate replay key")
            self.records[r["key"]] = r

    @staticmethod
    def key(stage, inputs, action):
        # Image content hash is portable; local path is not part of identity.
        return canonical_hash({"stage": stage, "inputs": {k: v for k, v in inputs.items() if k != "image"},
                               "action": action, "prompt": PROMPTS[stage]})

    def call(self, stage, inputs, action):
        key = self.key(stage, inputs, action)
        if key not in self.records:
            raise ValueError(f"missing recorded {stage} request {key}; no fallback predictions")
        r = self.records[key]
        self.events.append({"key": key, "stage": stage, "replayed": True,
                            "network_calls": 0, "recorded_usage": r.get("usage"),
                            "recorded_seconds": r.get("seconds")})
        return json.loads(json.dumps(r["output"]))


class GeminiRecorder(ReplayBackend):
    """No silent model fallback. One HTTP request per call, explicit budget.

    Research key comes from the environment. Only image/question/response and
    prediction evidence are sent. Raw response and usage are stored for replay.
    """
    def __init__(self, path, model, verifier_model, max_calls=10, timeout=60):
        self.path = Path(path)
        self.records, self.events = {}, []
        if self.path.exists():
            super().__init__(path)
        self.model, self.verifier_model = model, verifier_model
        self.max_calls, self.calls, self.timeout = max_calls, 0, timeout
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY in the environment")

    def call(self, stage, inputs, action):
        model = self.verifier_model if stage in ("verify", "preservation") else self.model
        key = self.key(stage, inputs, action)
        if key in self.records:
            if self.records[key].get("model_requested") != model:
                raise ValueError("recording model mismatch; use a separate recording file")
            return super().call(stage, inputs, action)
        if self.calls >= self.max_calls:
            raise RuntimeError("API call budget exhausted")
        import hashlib
        import mimetypes
        image = Path(inputs["image"]).read_bytes()
        if hashlib.sha256(image).hexdigest() != inputs["image_sha256"]:
            raise ValueError("image hash changed")
        text_input = {k: v for k, v in inputs.items() if k not in ("image", "image_sha256")}
        parts = [{"text": PROMPTS[stage] + "\nAction: " + action + "\nInputs: " + json.dumps(text_input)}]
        if stage != "preservation":
            parts.append({"inline_data": {"mime_type": mimetypes.guess_type(inputs["image"])[0] or "image/jpeg",
                                          "data": base64.b64encode(image).decode()}})
        body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 2048,
                                 "responseMimeType": "application/json"}}
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        self.calls += 1
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            self.events.append({"stage": stage, "network_calls": 1,
                                "seconds": time.perf_counter()-started, "failed": True})
            raise RuntimeError(f"{stage} request failed ({type(exc).__name__}); no automatic fallback") from None
        parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        output = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
        record = {"key": key, "stage": stage, "action": action,
                  "model_requested": model, "model_returned": raw.get("modelVersion"),
                  "prompt": PROMPTS[stage], "input": text_input,
                  "image_sha256": inputs["image_sha256"], "output": output,
                  "raw_response": raw, "usage": raw.get("usageMetadata"),
                  "seconds": time.perf_counter()-started, "synthetic": False}
        self.records[key] = record
        write_jsonl(self.path, list(self.records.values()))
        self.events.append({"key": key, "stage": stage, "network_calls": 1,
                            "usage": record["usage"], "seconds": record["seconds"], "replayed": False})
        return output
