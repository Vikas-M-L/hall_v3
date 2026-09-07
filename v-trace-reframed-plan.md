# V-TRACE+ Reframed: Mechanism-Attributive Hallucination Diagnosis

**Version 2** — revised after adversarial review. Changelog at §14.
**Scope:** 8-week build, single-GPU feasible, workshop or short-paper target.
`[VERIFY]` marks Day-1 checks. `[PREREG]` marks decisions to write down before seeing results.

---

## 0. What "a 10" means under an 8-week constraint

A 10 is not a bigger system. It is **one conceptual claim that reframes the problem, executed cleanly enough that the result is unambiguous.**

Be clear-eyed about the trade: this reframe is not less engineering than the original plan. It drops fine-tuning and a 2×2 grid, and adds a diagnostic signal, synthetic induction sets, repair operators, a routing policy, and — the piece most easily underestimated — a claim-verification pipeline. It fits in 8 weeks only because of one aggressive scope decision, stated here because everything downstream depends on it:

> **The headline repair experiment runs on single-claim responses only** — short-answer VQA and POPE-style prompts, where one claim *is* the response.

This is not a convenience. Repairs like contrastive decoding regenerate an entire response, so on a multi-claim caption you cannot repair one claim without rewriting the others. That makes claim-level routing undefined, makes "correct claims broken" a cross-claim interference measurement, and forces a re-decompose-and-realign step after every repair. Restricting to single-claim responses makes repair well-defined, verification near-exact, and removes three confounds at once. Free-form captioning still appears — but only in the **diagnostic** results, never in the repair table.

---

## 1. The diagnosis: a contradiction inside the original design

The original motivation opens correctly:

> VLMs hallucinate for different reasons — weak visual grounding, miscounting, confusing similar objects, over-relying on prior/common-sense bias, or internal uncertainty that doesn't show up in output confidence.

The pipeline then collapses all of it into a single scalar.

The premise is multi-causal and the output is unicausal. Everything expensive in the design — heterogeneous signals, adaptive gating, counterfactual interventions — exists *because* causes differ, and the final layer discards the cause and keeps the magnitude. Worse, the information is already computed: the gating weights **are** a cause estimate. Recovering and supervising that representation is nearly free.

### The motivating example

Two claims, identical fused risk of 0.8. Any scalar detector treats them identically.

- **A:** *"a fork is on the table."* Confident, the image contributed little, no fork present. Co-occurrence statistics talking. Correct repair: **contrastive decoding**.
- **B:** *"the sign reads 14 km."* Confident, the image genuinely drove the prediction, internal representation uncertain, text is 40 pixels wide. Correct repair: **abstain or re-query at higher resolution**. Contrastive decoding is actively *harmful* here — it suppresses a prior the model legitimately needs to read degraded text.

Same score, opposite correct action. That is the paper.

**Caveat to carry forward:** these two examples differ in string frequency as well as in mechanism, and §3 explains why that is a live confound rather than a rhetorical convenience.

---

## 2. The reframe

**From:** *is this claim hallucinated?* **To:** *why is it wrong, and what repair does that imply?*

1. **Mechanism is identifiable** at claim level from cheap signals.
2. **Mechanisms requiring opposite repairs are separable under the full signal vector but confusable under scalar risk.** Tested directly by rung 6.
3. **Mechanism-routed repair beats uniform repair** — and, critically, beats *detect-then-apply-one-fixed-repair*, which is the baseline that isolates whether choosing *which* repair matters at all.

Claim 3's second half is the whole ballgame and is where v1 of this plan was weakest. See §7.

---

## 3. The diagnostic anchor: Visual Information Gain, positioned honestly

```
VIG(c) = log P(c | I, x)  −  log E_{I'}[ P(c | I', x) ]
```

for claim `c`, image `I`, and null images `I'` — the pointwise mutual information between claim and image, in nats.

### This quantity is not new, and the paper must say so

`log p(c|I,x) − log p(c|x)` is essentially the PMI objective in **M3ID** (Favero et al., CVPR 2024), and with a noised null it is **VCD**'s (Leng et al., CVPR 2024) per-step contrast summed over claim tokens. `[VERIFY]` both in Week 0.

So do not claim VIG as a novel signal — that framing will not survive anyone who knows this literature, and VCD appearing as one of your *repair operators* while its internals are your *headline feature* is an awkwardness a reviewer will enjoy. The honest and still-strong framing:

> We repurpose a known decoding-time grounding signal as a **claim-level diagnostic feature**, and show that its joint distribution with output confidence and independent uncertainty predicts *which* decoding-time intervention to apply.

The contribution is the routing, not the ratio. That was always true; v1 obscured it.

### The rarity confound — test this before trusting anything

For claims the model asserts confidently, the first term is near-saturated, so VIG ≈ −log P(c | prior) + const, i.e. **mostly claim rarity**. And §1's motivating pair differs in exactly that: *"a fork is on the table"* is common, *"the sign reads 14 km"* is rare. VIG would separate them even if the mechanism story were false.

Two consequences, both non-negotiable:

- **Add `log E[P(c|I',x)]` — the prior term — to the feature vector as its own feature.** M1 is characterized by *high prior*, not merely *low gain*. Low VIG has a large false-positive basin: generic claims, true-but-obvious claims, and claims the model would never generate.
- **Gate 1 tests incremental power.** Does VIG separate M1 from M3 *after regressing out the prior term*? If not, the information-theoretic framing is empty and you have re-derived "common claims are prior-driven." Cheap, decisive, Week 2.

### Estimator and implementation details that will bite

**Use `logsumexp(log p_i) − log N`, not the mean of log-probs.** v1 said "mean over N=4 random images," which by Jensen systematically *overestimates* VIG, with bias growing in variance — worst exactly for the rare, specific claims you are calling M3.

**Fix visual token counts across all null constructions.** Qwen2.5-VL uses dynamic resolution and M-RoPE; if nulls produce different token counts, VIG absorbs prefix-length and position effects. Resize everything to identical token budgets. Token-removal nulls cannot be made comparable this way, which is itself a finding worth one sentence.

**Define `x` explicitly and preregister it.** `[PREREG]` If `x` includes the generated response, the claim is entailed, both terms saturate, and VIG ≈ 0 everywhere. If `x` is a fresh verification template, VIG measures likelihood under a context that did not produce the claim — so soften "how many nats the image contributed" to a statement about the scoring template.

**Report summed and per-token VIG.** The difference cancels length bias but not length scale; variance grows with token count. Standardize per claim type — count and spatial claims live on different likelihood scales, and an additive `length` feature cannot repair a multiplicative effect.

**Null construction ablation stays** — four constructions (token removal, grey, Gaussian noise, random-image expectation), all extracted in the *same* Week 2 pass. "How you construct the null materially changes prior-reliance estimates" is a real minor finding.

### The signal v1 was missing entirely

**Sampling-based self-consistency** (SelfCheckGPT-style: sample k responses at temperature, measure claim agreement / semantic entropy). It is the standard baseline in this literature, backbone-agnostic, and a more natural operationalization of "unsurfaced internal uncertainty" than a trained probe. It does triple duty here: independent M3 label source (§6), external baseline (§8 rung 10), and hedge against UniProbe being unavailable.

---

## 4. The mechanism taxonomy

Three mechanisms plus an explicit no-op, one optional. Orthogonal to the claim-type taxonomy (object/attribute/count/...): claim type is *what* is asserted, mechanism is *why it failed*. Both enter the feature vector; only mechanism is predicted.

| | Output conf. | Prior term | VIG | Grounding | Indep. uncertainty | Repair |
|---|---|---|---|---|---|---|
| **M0 No fault** — claim is correct | any | any | any | any | any | none |
| **M1 Prior override** | High | **High** | Low | Weak | Low | Contrastive decoding |
| **M2 Perceptual failure** | Low–mid | Mid | Low–mid | Weak, localizable | Mid | Crop-and-zoom re-query |
| **M3 Unsurfaced uncertainty** | High | Low | Mid–high | Mid | **High** | Abstain / hedge |
| **M4 Binding error** *(optional)* | High | Mid | High | **High** | Mid | Isolated-region re-query |

M0 is not decoration. Without an explicit no-op the router must repair everything it sees, which silently converts the routing comparison into a detection-threshold comparison (§7).

**The M1/M3 pair is the crux.** Both present as confident and wrong; they need opposite treatment. Note that separation rests on *two* signals — the prior term and independent uncertainty — not on VIG alone. v1 claimed VIG was "the only signal that separates M1 from M3" while its own table showed the probe separating them too. The honest claim: VIG plus the prior term plus an independent uncertainty estimate jointly separate them, and rung 8 measures each one's marginal contribution.

**Anticipate the sharpest taxonomy attack:** *"M2 and M3 differ mainly in output confidence — you have renamed thresholds on a two-dimensional plane as causes."* The defence is not rhetorical, it is §6b: mechanisms that predict *which repair works* are doing more than partitioning a plane. If they do not, the attack lands and you should concede it.

**On CED:** confidence − evidence remains in the span of its inputs and gives a linear model nothing new. The reframe does not solve that; it makes it moot, because the (confidence, prior, VIG, uncertainty) coordinate system is now the object of interest and the interpretable direction survives inside it. Do not claim a problem was solved.

---

## 5. Architecture

Two-stage factorization, because a correct claim has no failure mechanism and `risk = Σ_m p(m)·risk_m` incoherently implies otherwise:

```
p(hallucinated | features)          ← detection head, all claims
p(m | features, hallucinated)       ← mechanism head, trained on hallucinated claims only
router: argmax over {M0, M1, M2, M3}, where M0 = decline to repair
```

Model class: **gradient boosting first**, logistic regression as an interpretable reference. Every row of §4's table is a *conjunction* (M1 = high confidence AND high prior AND low VIG), which multinomial LR cannot express without explicit interaction terms. For the mechanism-signature figure use per-mechanism feature distributions or SHAP, not LR coefficients.

Terminology: call it a two-stage classifier. "Gating network" and "mixture of experts" for a boosted tree over eight features is inflation a reviewer will notice.

**Probe leakage:** if you train the hidden-state probe yourself, feed the mechanism classifier **out-of-fold** probe predictions or the feature will be leaked and over-trusted.

---

## 6. Validation

### 6a. Synthetic induction, with the artifacts named

**M1 from POPE.** POPE's adversarial split queries frequently co-occurring but absent objects — a prior-override probe by construction. `[VERIFY]` the construction against the POPE paper. Two refinements over v1: run the **three-way** ordering (random < popular < adversarial) rather than a two-way test, since a monotone trend across three conditions is far stronger evidence and costs nothing; and note that documented **yes-bias** on POPE means a yes may reflect answer-format prior rather than visual co-occurrence prior — different mechanism, same label. Say so.

**M2 by degradation, validated on natural data.** Progressive blur/downscale/occlusion of a correctly-described region gives known-cause failures with known regions. But blur shifts VIG, grounding and CLIP together, so a classifier can learn "image is blurry → M2" and score perfectly while learning nothing. **Train on degradation; validate on naturally small or occluded objects**, stratifying COCO by instance area, and hold out degradation type.

**M3 is not constructible.** Label it with sampling-based self-consistency and use the hidden-state probe as the feature — never both, or you validate a prediction against the feature that made it. v1's "validate by probe agreement" was circular. Concede in limitations that M3 has no synthetic ground truth, which means the M1/M3 crux has constructed ground truth for only one of its two members.

**Distribution shift is real:** POPE is forced-choice VQA; deployment claims come from decomposed generation. A classifier trained on POPE-M1 and applied to caption claims is doing unexamined domain transfer. The §0 single-claim restriction largely aligns these for the repair experiment; for the diagnostic results, report transfer explicitly.

### 6b. The repair experiment *is* the validation

If M1 diagnoses are fixed by contrastive decoding but not by zoom, and M2 the reverse, the taxonomy has **predictive power over interventions** — validation no annotation exercise can match. State this explicitly: it converts "is your taxonomy correct?", unanswerable about latent causes, into "does it predict intervention outcomes?", answerable with a table.

---

## 7. The headline experiment, restructured

**This section is the one that was broken in v1.** The old table compared routing to uniform strategies and had no row isolating *why* routing wins. Three cheaper hypotheses predicted the same result, and a reviewer gets to pick whichever one kills you. Run on **all** claims, not just flagged ones — the collateral-damage story requires applying uniform repairs to unflagged claims too.

| Strategy | Halluc. fixed | Correct broken | Net gain | Extra passes |
|---|---|---|---|---|
| No repair | — | — | — | 0 |
| Uniform contrastive decoding | | | | |
| Uniform zoom-requery | | | | |
| Uniform abstain above risk threshold | | | | |
| All three repairs applied to all claims | | | | |
| **Detect-then-single-best-uniform** ← isolates *whether* vs *which* | | | | |
| **Policy trained on observed repair outcomes** ← taxonomy-free control | | | | |
| Route by claim type (object/attribute/count) ← null taxonomy | | | | |
| Random routing among operators | | | | |
| **Routed by predicted mechanism (ours)** | | | | |
| Routed, abstention disabled | | | | |
| Oracle: best repair per claim, chosen post hoc | | | | |

**The four rows that decide the paper's fate:**

*Detect-then-single-best-uniform* — the detector decides *whether* to repair, a fixed policy decides *what*. Without this, any gap between "routed" and "uniform" is partly just the flag/no-flag decision, which is ordinary scalar detection.

*Policy trained on observed repair outcomes* — running all repairs on all claims (a row you need anyway) hands you per-claim labels of which repair actually worked. Train a direct 4-way policy on the same features, no mechanisms, no synthetic induction, no taxonomy. It costs an afternoon. If it ties routing, the taxonomy is decorative — better you learn that in Week 3 than a reviewer in month four.

*Claim-type routing* — free, already in your feature vector, and the obvious null taxonomy. If it ties, M1/M2/M3 is a relabeling.

*Routed with abstention disabled* — abstention is available from the risk score alone and is the only operator that cannot hurt under a naive metric. If routing wins only because it abstains well, the mechanism story contributed nothing.

**Oracle is now "best repair per claim, post hoc," not "oracle mechanism."** It bounds total headroom without presuming the taxonomy is the right decomposition.

`[PREREG]` **Net gain**: hallucinations removed minus correct claims removed or corrupted, with **abstention counting as removal on both sides** — otherwise abstain-everything scores 100% fixed. Report a risk–coverage curve alongside.

`[PREREG]` **The paper's claim is specifically the delta over detect-then-best-uniform and over the repair-outcome policy.** Not over no-repair.

**Statistics:** these deltas will be small. Use a **paired design** — same claims through every arm — and adjudicate only claims where arms *disagree*. That makes human adjudication affordable and gives McNemar or a paired bootstrap instead of two independent proportions, which is also how you recover power. Run a power calculation in Week 1, before committing to an evaluation set size.

---

## 8. Experiment ladder

1. Individual signals alone, for detection.
2. Fixed equal-weight fusion.
3. Learned fixed-weight fusion.
4. Unconstrained predictor, no simplex constraint — prices interpretability honestly.
5. Two-stage model (ours): detection parity plus mechanism output.
6. **Claim 2 test:** M1-vs-M3 discrimination AUROC from scalar risk alone vs. the full feature vector. *This rung tests §2 Claim 2 and nothing in v1 did.*
7. **Routed vs. uniform vs. taxonomy-free controls** (§7) — the headline.
8. Leave-one-signal-out, plus the incremental test of VIG over the prior term alone.
9. Null-construction ablation.
10. External baseline — sampling-based self-consistency, backbone-agnostic, no porting.

---

## 9. Cut list

**Multi-claim repair — cut (the important one).** See §0. Diagnostic results still use free-form captions; repair results do not.

**Fine-tuning and the 2×2 — dropped.** Rigor, not contribution. Keep already-working components if free; do not build on them.

**Cross-backbone generalization — dropped**, and note this constrains §8 rung 10: most published baselines (VCD, OPERA, HALC, Woodpecker, LURE) ship LLaVA-1.5-centric code, so porting one to Qwen2.5-VL *is* the cross-backbone work. Hence a backbone-agnostic baseline instead.

**Learned image-level aggregation — cut.** Report max and top-k mean; `[PREREG]` which is primary.

**Contradiction-injection — cut.** Leaves M4 partly ambiguous; M4 is optional anyway.

**Snowballing — cut, with an honest limitation paragraph.** v1 cut this while §7 silently depended on it; §0's restriction is what makes the cut legitimate.

**MLP — cut.**

**Localization scope:** masked counterfactuals, M2 induction, and zoom-requery all need claim-aligned regions, which no benchmark provides for arbitrary claims. Restrict to object claims whose referent maps to a COCO category with an instance mask, and say so.

---

## 10. Schedule

**Week 0 (3 days).** `[VERIFY]` VHILT and UniProbe — I could not verify either, and neither is load-bearing here. Fallback: POPE + AMBER + M-HalDetect (Gunjal et al., AAAI 2024, span-level human labels), and **swap UniProbe for a self-trained mid-layer logistic probe** (v1 had this substitution backwards). `[VERIFY]` M3ID and VCD for the §3 positioning, and 2025–26 work on mitigation routing. **Kill gate:** if mechanism-conditioned repair routing is published, pivot to diagnostic-only.

**Week 1.** Lock benchmark. Claim decomposition. 70/15/15 with leakage check. Hand-audit ~100 decompositions and report an error rate. **Build the claim-verification pipeline** — for single-claim responses this is near-exact matching, which is exactly why §0's restriction pays. Power calculation for §7.

**Week 2.** VIG plus all four null constructions, extracted in one pass. **Gate 1:** three-way POPE ordering, *and* incremental power over the prior term. If VIG adds nothing beyond prior likelihood, stop and rethink.

**Week 3.** Remaining signals, including sampling self-consistency. M1/M2 induction sets. **Gate 2, now actually runnable:** prototype two repair operators and run routed-vs-uniform-vs-repair-outcome-policy on the *synthetic* M1/M2 sets, where labels exist by construction and claims are single-response. v1 put this gate in Week 3 while scheduling repairs in Week 5 — it could not have run.

**Week 4.** Two-stage model. Ladder rungs 1–6. External baseline.

**Week 5.** Harden repair operators. Begin writing methods.

**Week 6.** Full §7 table with cost accounting and paired statistics.

**Week 7.** Ablations (rungs 8–9), oracle gap analysis. Genuine slack — do not pre-spend it.

**Week 8.** Figures and writing. Redraw the pipeline with per-claim lanes, an image skip-connection into signal extraction, the counterfactual as a loop back through the VLM, M0 in the router, and the router as terminal node.

### Compute

Per claim: 1 base + 4 null (random-image expectation) + 1 masked ≈ **6 passes**, ~9 with all ablation constructions — roughly 3× v1's estimate. **Prefix caching is mandatory**: with the claim scored after the image, a naive loop re-prefills ~1,300 visual tokens per claim, and at ~5 claims per image that is a 5× waste. Prefill each (image, null) once and reuse the KV cache (vLLM automatic prefix caching or HF `past_key_values`). With caching and `max_pixels` capped near `1280*28*28`, expect **2–8 hours on a 24GB card** for all constructions; without it, a day and a half.

Qwen2.5-VL-7B is ~16.6 GB in bf16, so 24 GB works but you cannot hold the VLM, CLIP, verifier and decomposition LLM resident at once. Serialize stages, cache features to disk.

**The binding constraint is not GPU.** It is claim verification and repair-experiment engineering. Budget accordingly.

---

## 11. Risks and fallbacks

**Routing ties the taxonomy-free controls.** The main risk, and now measured in Week 3 rather than month four. `[PREREG]` Fallback: the paper becomes the diagnostic result — mechanisms are identifiable and separable (rungs 5, 6) — plus an honest negative result on routing with analysis of why. Weaker, still a workshop paper. Write this framing down now.

**VIG adds nothing over the prior term.** Caught Week 2. Check the estimator and null construction first; token-removal distribution shift is the likeliest culprit.

**VCD positioning.** Frame VCD as a repair operator *and* acknowledge its contrast is your diagnostic feature. Your contribution is deciding when it is the right tool, plus evidence that unconditional application damages correct claims.

**Cost claims.** Routing costs six-plus diagnostic passes per claim plus repair; uniform abstention costs almost nothing. Do not claim "lower compute" without qualification — say **lower repair cost than applying all operators**, and argue the diagnostic amortizes because detection is needed regardless.

**Over-claiming identifiability.** You infer latent cause from correlational signatures. Write "signature consistent with," not "caused by," except where induction gives real ground truth.

**Mixed etiology.** Claims can have more than one cause. The posterior handles it; show an example rather than hiding it.

---

## 12. Novelty statement

> A claim-level framework that attributes vision-language hallucinations to distinct failure mechanisms — prior override, perceptual failure, and unsurfaced internal uncertainty — by repurposing a decoding-time visual-grounding signal as a diagnostic feature, and shows that routing claims to mechanism-matched repair operators outperforms both uniform mitigation and detect-then-fixed-repair, with less collateral damage to correct claims.

Every clause maps to a row or rung. "Detection" does not appear; neither does an unqualified compute claim.

---

## 13. Lock down before writing

Benchmark confirmed accessible and licensed. Claim-type taxonomy provenance footnoted. Decomposition error rate. `[PREREG]`: the scoring template `x`; primary null construction; max vs. top-k aggregation; the net-gain definition including abstention; the §11 fallback framing; and the specific baselines the headline delta is claimed over.

---

## 14. Changelog from v1

The adversarial review found one fatal issue, three schedule-breakers, and a novelty misattribution.

**Fatal:** repairs are response-level while routing is claim-level — resolved by restricting the repair experiment to single-claim responses (§0), which also eliminated the re-alignment step and the interference confound.

**Attribution gap:** v1's §7 could not show its gain came from the taxonomy. Added detect-then-best-uniform, a repair-outcome policy, claim-type routing, random routing, and abstention-disabled rows; oracle changed from mechanism to best-repair.

**Novelty:** VIG is essentially M3ID's PMI term and VCD's contrast. Repositioned from "new principled signal" to "known signal repurposed as diagnostic."

**Rarity confound:** VIG largely reduces to claim rarity; added the prior term as its own feature and an incremental-power gate.

**Schedule:** Gate 2 required Week 5 deliverables and a verification pipeline that appeared nowhere; both fixed and the verification pipeline budgeted into Week 1.

**Also fixed:** Jensen bias in the null estimator (logsumexp, not mean of logs); M3's circular probe validation (self-consistency labels, probe as feature); incoherent `Σ p(m)·risk_m` (two-stage factorization plus M0 no-op); LR cannot express the table's conjunctions (GBM first); missing self-consistency signal and baseline; compute off ~3× plus mandatory prefix caching; three-way POPE ordering; M2 degradation artifacts; undefined net-gain metric; a rung testing Claim 2; the backwards UniProbe substitution; and internal contradictions on VIG uniqueness, mechanism count, and compute cost.

*Naming is cosmetic: V-TRACE reads as Visual Tracing of Response Attribution and Causal Etiology.*
