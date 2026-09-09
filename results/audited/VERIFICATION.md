# Verification performed in this audit

- New protocol suite: **21 passed** (`python -m pytest research/tests -q`).
- Legacy product suite before the final one-test diagnosis regression addition:
  **67 passed**; updated diagnosis sub-suite: **10 passed**.
- Legacy research suite: **40 passed**. Passing includes known legacy semantic
  defects documented in the audit; it is not validation of research claims.
- New plus updated diagnosis tests together: **31 passed**.
- Streamlit AppTest: Checker initial render, Benchmarks and Methods each
  completed with no reported app exceptions. Upload/interaction paths were not
  exhaustively UI-tested.
- Synthetic replay: nine strategy traces, zero missing-record failures.
- Actual-answer live smoke: three strategy traces, zero execution failures;
  controlled author-edit smoke: three traces, zero execution failures.
- Both live recordings replayed with three traces and zero failures each.
  Their correctness has not been independently adjudicated.
- Source inventory parses legacy Python files and records source/artifact hashes.
- `git diff --check` passed. No commits or pushes were made during this audit.
- Audited PDF/PNG figures and LaTeX tables were generated. Image attachment
  viewing was unavailable; no visual-layout inspection is claimed.
- No local `pdflatex` or `tectonic` executable was found. Paper PDF compilation
  and exact page count are unverified.

Deprecation warnings from SciPy/sklearn, SWIG and Streamlit remain. They did
not fail these checks. Environment versions are in `environment.json`.
