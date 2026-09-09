"""Hash all research-relevant source/artifacts without reading secrets."""
import hashlib
import ast
import importlib.metadata
import platform
from pathlib import Path

from research.core import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    entries = []
    allowed = {".py", ".md", ".json", ".csv", ".yaml", ".yml", ".tex", ".joblib", ".png", ".jpg", ".pdf"}
    for base in ("vtrace-plus", "adaptive-vtrace", "paper"):
        for p in sorted((root / base).rglob("*")):
            if not p.is_file() or p.suffix not in allowed or any(
                    x in p.parts for x in ("__pycache__", ".pytest_cache", ".git", ".venv")):
                continue
            if p.name.startswith(".env"):
                continue
            content = p.read_bytes()
            entry = {"path": p.relative_to(root).as_posix(), "bytes": len(content),
                     "sha256": hashlib.sha256(content).hexdigest()}
            if p.suffix == ".py":
                tree = ast.parse(content.decode("utf-8-sig"), filename=str(p))
                entry["classes"] = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                entry["functions"] = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                entry["imports"] = sorted({n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                                          {alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names})
            entries.append(entry)
    write_json(root / "results/audited/inventory.json", {"files": entries, "python": platform.python_version(),
                                                         "scope": "source/artifacts, secrets excluded"})
    print("Inventoried", len(entries), "source/artifact files.")


if __name__ == "__main__":
    main()
