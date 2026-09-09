# Failure inspection protocol

`results/audited/failure_review_queue.jsonl` contains 67 unique legacy cases
with existing object-presence gold and raw method scores. It does **not** contain
67 human-reviewed failures; it includes correct cases and candidates for review.
No five-category error examples or fifty manual annotations were invented.

The existing gallery's 14 claims have hand-written expectations, but no
independent votes. Its boxes/counts cannot establish natural ground truth.
The live controlled edit is author-created; the original live answer is real.
Both use an already-exposed development image.

## Review procedure

1. A custodian samples at least 50 failures across prespecified types and
   strategies, plus controls. Keep random seed and selection query.
2. Two annotators see image/question/answer and official evidence but not method
   names. Mark all unsupported spans, required facts, answer correctness and
   ambiguity. Annotators also rate actual candidate and final output.
3. Compare votes; adjudicate independently; save original votes and agreement.
4. Assign detector false positive/negative, diagnosis error, routing error,
   failed repair, overcorrection, verifier failure, ambiguous evidence, or no
   system error. A case can have multiple categories.
5. Export image, question, original response, detected span, predicted type,
   selected action, candidate, verifier output, accepted/rejected result,
   adjudicated correctness and licensing attribution. Publish representative
   examples with counts and sampling procedure, not only success screenshots.

Missing: source images for every queued case, original generated answers for
legacy POPE noun scores, independent votes, candidate repairs for the legacy
dataset, completed adjudication and inter-annotator agreement.
