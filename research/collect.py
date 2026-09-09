"""Collect an actual VLM answer before detection, with no reference answer input."""
import argparse
import hashlib
import json
from pathlib import Path

from research.backends import GeminiRecorder
from research.core import Case, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--group-id", required=True)
    ap.add_argument("--config", default="configs/closed_loop.json")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    path = Path(args.image)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    backend = GeminiRecorder(args.recording, cfg["model"], cfg["verifier_model"], 1, cfg.get("timeout", 60))
    output = backend.call("generate", {"image": str(path), "image_sha256": digest,
                                      "question": args.question, "response": ""}, "none")
    if not isinstance(output.get("response"), str) or not output["response"].strip():
        raise ValueError("generation failed: no response")
    from dataclasses import asdict
    c = Case(args.id, args.group_id, str(path), digest, args.question, output["response"], "development")
    write_jsonl(args.out, [asdict(c)])
    print("Recorded actual VLM response; gold/adjudication still required.")


if __name__ == "__main__":
    main()
