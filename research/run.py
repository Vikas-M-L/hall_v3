"""Run label-free closed-loop experiments: python -m research.run --help."""
import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from research.backends import GeminiRecorder, PROMPTS, ReplayBackend
from research.core import Case, STRATEGIES, canonical_hash, read_jsonl, run_case, validate_cases, write_json
from research.telemetry import MemorySampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", required=True, help="model-visible JSONL, no gold fields")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seal", help="preregistered input/config hash commitment required for test")
    ap.add_argument("--live", action="store_true", help="send image inputs to Gemini and record usage")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cases = [Case.parse(r) for r in read_jsonl(args.inputs)]
    inventory = validate_cases(cases, verify_images=args.live)
    if not cases:
        ap.error("empty input dataset")
    strategies = config.get("strategies", list(STRATEGIES))
    if not set(strategies) <= set(STRATEGIES):
        ap.error("unknown strategy")
    if any(c.split == "test" for c in cases) and config.get("status") != "frozen":
        ap.error("test evaluation requires a frozen preregistered configuration")
    if any(c.split == "test" for c in cases):
        if not args.seal:
            ap.error("test inputs require --seal")
        seal = json.loads(Path(args.seal).read_text(encoding="utf-8"))
        if seal.get("config_sha256") != canonical_hash(config) or seal.get("input_sha256") != hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest():
            ap.error("input/config changed after sealing")
    backend = (GeminiRecorder(args.recording, config["model"], config["verifier_model"],
                              config.get("max_calls", 10), config.get("timeout", 60))
               if args.live else ReplayBackend(args.recording))
    start, traces, errors = time.perf_counter(), [], []
    memory = MemorySampler().start()
    for case in cases:
        for strategy in strategies:
            begin, event_start = time.perf_counter(), len(backend.events)
            try:
                trace = run_case(case, backend, strategy, config.get("threshold", .5), config["seed"])
                trace["events"] = backend.events[event_start:]
                trace["wall_seconds"] = time.perf_counter()-begin
                traces.append(trace)
            except Exception as exc:
                errors.append({"id": case.id, "strategy": strategy, "error_type": type(exc).__name__,
                               "message": str(exc), "events": backend.events[event_start:]})
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    write_json(args.out, {"config": config, "config_sha256": canonical_hash(config),
                         "input_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
                         "recording_sha256": hashlib.sha256(Path(args.recording).read_bytes()).hexdigest(),
                         "source_revision": revision, "python": platform.python_version(),
                         "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in Path(__file__).parent.glob("*.py")},
                         "platform": platform.platform(), "prompts": PROMPTS, "inventory": inventory,
                         "live": args.live, "total_seconds": time.perf_counter()-start,
                         "memory": memory.finish(),
                         "traces": traces, "errors": errors,
                         "scientific_status": "unscored; independent adjudication required"})
    print(f"Recorded {len(traces)} traces; {len(errors)} failures. No gold was passed to models.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
