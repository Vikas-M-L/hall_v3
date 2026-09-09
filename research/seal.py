"""Create a timestamped protocol/input commitment; organizational custody still needed."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from research.core import canonical_hash, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise ValueError("finish development, then explicitly set status=frozen")
    if Path(args.out).exists():
        raise ValueError("seal exists; do not overwrite preregistration")
    write_json(args.out, {"config_sha256": canonical_hash(config),
                         "input_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
                         "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         "note": "Commit/archive before test execution; does not retroactively remove contamination."})


if __name__ == "__main__":
    main()
