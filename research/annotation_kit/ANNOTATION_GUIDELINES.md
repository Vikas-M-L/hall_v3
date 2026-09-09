# Annotation kit — human labeling for preservation-aware repair evaluation

Two independent annotators label each case; a third adjudicates disagreements.
Annotators see: image, question, original VLM answer. They NEVER see method
names, system verdicts, or repair suggestions.

## Files

- `annotation_template.csv` — one row per case, filled by each annotator
  separately (copy to `annotator_A.csv`, `annotator_B.csv`).
- `ANNOTATION_GUIDELINES.md` — this file: label definitions, span rules,
  worked examples, agreement procedure.
- Filled sheets feed `research.prepare_dataset` (field names match exactly).

## Label definitions

| hallucination_type | Meaning | Example |
|---|---|---|
| object | named object absent from image | "a dog" with only cats visible |
| attribute | object present, property wrong | "black dress" when jacket is pink |
| counting | wrong number of instances | "5 cats" when 2 visible |
| spatial | wrong position/layout | "cat on the left" when on the right |
| relation | wrong interaction between objects | "cat holding a ball" |
| ocr | misread visible text | sign says "14 km", answer says "40 km" |
| reasoning | wrong inference from visible facts | miscounted total across regions |
| knowledge | needs external facts | breed, artist, historical claim |
| unsupported_inference | guess beyond visible evidence | emotions, intentions, off-screen events |
| grounding | vague/partially supported | "scene looks nice" used as fact |
| none | no hallucination in this answer | fully supported response |

## Span rules

- Mark the MINIMAL substring carrying the falsehood: "5 cats are eating" →
  span "5" for the count error, not the whole sentence.
- `hallucinated_spans`: JSON list of [start, end] offsets into `response`.
- `severity`: `minor` (detail), `material` (answer-changing), `unanswerable`.

## Visual evidence

- `visual_evidence`: bounding boxes `[x0,y0,x1,y1]` in pixels + what they show,
  or empty list with reason "occluded" / "absent" / "illegible".
- Missing evidence is an empty list, never an invented box.

## Corrected answer

- Rewrite minimally: fix false spans, keep all supported facts, no new guesses.
- If the image cannot answer, write "Cannot determine from this image."

## Agreement procedure

1. A and B annotate independently (separate file copies).
2. Adjudicator C resolves every disagreement; original A/B votes are kept.
3. `annotator_ids`: ["A","B"], `adjudicator`: "C", status `adjudicated` only
   after step 2. Report Cohen's kappa on type labels + span overlap separately.
4. Minimum 50 adjudicated failure cases across types before any paper claim.
