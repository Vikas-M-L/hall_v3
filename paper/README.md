# Audited paper

`main.tex` is the current IEEE-style **audit/protocol draft**. It reports only
recomputed archived scores, a clearly synthetic diagnosis shift experiment,
and recorded API smoke functionality. It does not claim validated repair
superiority, SOTA, formal risk guarantees or a completed human benchmark.

`legacy_draft_unvalidated.tex` is preserved for provenance and must not be
submitted. Its stale claims and incorrect citation are listed in the audits.
Historical `figs/` images are archived, not used by the new paper.

Generate the current figures/tables from the repository root:

```
python -m research.analyze
python -m research.synthetic_diagnosis
python -m research.report_assets
```

Then run pdfLaTeX twice from this directory. IEEEtran is required. A TeX
compiler was not present during the audit, so PDF compilation and exact page
count have not been verified. Figures were generated in vector PDF and PNG;
visual attachment inspection was unavailable in the tool session.

## Alternative titles

1. **V-TRACE+: Auditing Diagnosis-Aware Hallucination Detection, Repair Routing,
   and Verification for Vision-Language Models** (current evidence supports this).
2. **Does Diagnosis Improve Visual Hallucination Repair? A Preservation-Aware
   Evaluation Protocol** (methodology/negative-result framing).
3. **Selective Verification and Error-Specific Repair for Vision-Language
   Models** (use only after clean closed-loop experiments).

No target venue/page limit was finalized for this revision. Do not compress
away limitations or negative results to fit an unverified page count.
