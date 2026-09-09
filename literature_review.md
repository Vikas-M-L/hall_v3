# Literature review and novelty boundary

Checked 2026-09-09 against the primary links below. This is a targeted review,
not an exhaustive systematic survey. Abstracts/metadata establish identity and
high-level findings; numerical results not checked against full tables are not
used as experimental comparators. Prior-paper results are not V-TRACE+ results.
`N/V` means the exact dataset/model/table quantity has not been verified in the
accessible source and must be checked before implementation or quotation.

Searches: arXiv `vision language hallucination` (newest first), `hallucination
routing`, exact titles, and primary abstracts. Older benchmark/method papers
are included because they define the evaluation; 2026 work changes the novelty
assessment. Papers discovered under incorrect guessed identifiers were rejected.

## A. Benchmarks and diagnosis

| Paper / verified source | Year / venue evidence | Problem, method | Dataset; model | Metrics / main finding | Limitation and V-TRACE+ distinction |
|---|---|---|---|---|---|
| [Object Hallucination in Image Captioning](https://arxiv.org/abs/1809.02156), Rohrbach et al. | EMNLP 2018 | Language-driven object invention; CHAIR checks generated nouns against image labels | MSCOCO; captioning architectures | CHAIR object/sentence rates; caption similarity can miss hallucination | Object vocabulary limited; V-TRACE+ aims beyond objects, not a replacement for CHAIR |
| [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355), Li et al. | EMNLP 2023 | Polling-based yes/no object evaluation (POPE), random/popular/adversarial negative sampling | COCO and LVLM evaluation; exact per-model tables N/V here | Accuracy/precision/recall/F1 and yes tendency; frequent/co-occurring objects hallucinated | Binary presence does not supply counts, spans or causes; our legacy noun-score task differs from answering POPE |
| [Aligning Large Multimodal Models with Factually Augmented RLHF](https://arxiv.org/abs/2309.14525), Sun et al. | 2023 preprint | Factual reward augmentation; introduces MMHal-Bench | MMHal-Bench/LLaVA-Bench; LLaVA-RLHF | Human/model-rated quality and hallucination; abstract reports improvement over its baselines | Judge dependence and training cost; our repair router must preserve answers without treating judge output as gold |
| [AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation](https://arxiv.org/abs/2311.07397), Wang et al. | 2023, revised 2024 | Generative and discriminative existence/attribute/relation evaluation | AMBER; multiple MLLMs including GPT-4V | LLM-free evaluation dimensions; exact headline table N/V | Curated ontology and label coverage; use official scorer for external generalization |
| [HallusionBench](https://arxiv.org/abs/2310.14566), Guan et al. | CVPR 2024 | Controlled image-context questions separate hallucination and illusion | Human-designed images/questions; GPT-4V, Gemini, Claude, LLaVA and others | Question-pair accuracy and consistency; difficult even for strong models | Small curated reasoning domain; prevents assuming a detector's disagreement identifies a cause |
| [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://arxiv.org/abs/2306.13394), Fu et al. | 2023 preprint; current metadata NeurIPS DB 2025 | Perception/cognition evaluation, manually written QA | MME; many MLLMs | Task scores, accuracy and paired consistency | General capability not solely hallucination; useful answer-preservation control |
| [FaithScore: Fine-grained Evaluations of Hallucinations in Large Vision-Language Models](https://arxiv.org/abs/2311.01477), Jing et al. | Findings EMNLP 2024 | Descriptive sub-sentences → atomic facts → image consistency | LLaVA-1k, MSCOCO-Cap; LVLMs | Atomic factuality, correlation with human judgment | Verifier errors; atomic decomposition is prior art, not our standalone novelty |
| [Detecting and Preventing Hallucinations in Large Vision Language Models](https://arxiv.org/abs/2308.06394), Gunjal et al. | AAAI 2024 | M-HalDetect fine-grained annotation, reward modeling and FDPO | M-HalDetect; InstructBLIP, transfer to LLaVA/mPLUG-OWL | Human hallucination rates, reward correlation; reports reductions with FDPO/rejection | Label cost and model-specific training; natural span labels preferable to our generated six-feature clusters |

## B. Verification, mitigation and intervention

| Paper / source | Year | Problem, method | Dataset; model | Metrics / main finding | Limitation and distinction |
|---|---|---|---|---|---|
| [Woodpecker: Hallucination Correction for Multimodal Large Language Models](https://arxiv.org/abs/2310.16045), Yin et al. | 2023; SCIS 2024 | Concept extraction, questions, visual validation, visual claims, correction | POPE; MiniGPT-4/mPLUG-Owl | POPE accuracy improvements reported; interpretable intermediates | Closest post-remedy baseline; cannot claim first extract/verify/repair loop. Corrects old paper's unrelated citation `2310.06554` |
| [Mitigating Object Hallucinations ... through Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922), Leng et al. | 2023; CVPR 2024 | Contrast original/distorted image logits, no training | Multiple LVLM families and object/general benchmarks; exact settings N/V | Hallucination and general capability improvements | Needs logits and extra passes; remote API prompting is not VCD |
| [OPERA](https://arxiv.org/abs/2311.17911), Huang et al. | CVPR 2024 | Over-trust penalty and retrospection-allocation during beam decoding | Multiple MLLMs; exact table settings N/V | Reduced hallucination without extra training/data | Access to attention/decoding; not comparable to an API-only method at unmeasured cost |
| [HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding](https://arxiv.org/abs/2403.00425), Chen et al. | ICML 2024 | Local auto-focal grounding plus global beam search | Four benchmarks and LVLMs, exact table details N/V | Reduced object hallucination while preserving generation quality | Localization and decoding costs; adaptive visual intervention already exists |
| [Multi-Modal Hallucination Control by Visual Information Grounding](https://arxiv.org/abs/2403.14003), **Alessandro** Favero et al. | CVPR 2024 | M3ID mutual-information decoding; optional DPO | LLaVA-13B and captioning/POPE evaluations | Grounding amplification improves benchmark metrics | Logit access; existing VIG code is not a validated new estimator. Corrects the draft's author initial |
| [Trusting Your Evidence: Hallucinate Less with Context-aware Decoding](https://arxiv.org/abs/2305.14739), Shi et al. | 2023 | Contrast with/without context against parametric priors | Summarization and knowledge-conflict tasks; OPT/GPT/LLaMA/FLAN-T5 | Factuality and conflict-resolution gains | Text evidence; contrastive evidence weighting is established |
| [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651), Madaan et al. | 2023 | Same model generates, critiques and refines | Seven text tasks; GPT-3.5/ChatGPT/GPT-4 | Human and automatic preference gains | Self-confirmation; compare an explicitly labeled vision adaptation, not claim original replication |
| [A Stitch in Time Saves Nine](https://arxiv.org/abs/2307.03987), Varshney et al. | 2023 | Validate low-confidence generations and mitigate during generation | Article generation, multi-hop/false-premise QA; GPT-3.5/Vicuna | Detection recall and mitigation/false-positive preservation | Text-only evidence; selective verification is prior art |
| [When Visual Signals Mislead: A Mechanistic Study of Attribute Hallucination in Vision-Language Models](https://arxiv.org/abs/2608.11024), Zhang et al. | 2026 preprint | **VISOR: null-image diagnosis with routed remediation**; calibration, abstention or visual adaptation by failure signature | Three attribute types; Qwen, InternVL, LLaVA | False positives and visual/prior decomposition | **Direct overlap:** broad diagnosis→routing novelty is not defensible. Our possible distinction is repair acceptance, preservation and multi-type cost-matched evaluation |
| [VisER: Visual Evidence and Reliance for Object Hallucination Detection in LVLMs](https://arxiv.org/abs/2608.30480), Hasanebrahimi et al. | 2026, metadata EMNLP main | Separate object-specific image evidence from generated-prefix support | Multiple LVLMs/benchmarks; full tables N/V | AUROC/AUPR improvements, no object-level verification generations | Source-confounding directly challenges our use of raw similarity as evidence of prior override |
| [Does Playing it Safe Count as Faithfulness?](https://arxiv.org/abs/2609.01888), Fazli et al. | 2026 preprint | Reassesses hallucination reduction vs informativeness and capability | Six methods, three LVLMs, four benchmarks including MMStar | Reduced hallucination can accompany reduced recall/detail/capability | Direct rationale for answer preservation and abstention-aware metrics; no claim that lower hallucination alone is progress |
| [SpanCalib-VLM](https://arxiv.org/abs/2608.29974), Abebe and Moslem | 2026 preprint/shared-task report | Generative span proposals plus calibrated discriminative tagger | SHROOM-Visions; XLM-R-large/SigLIP and Qwen3.5-4B SFT | Span IoU, calibration correlation, detection accuracy | Training cost and domain dependence; useful fine-grained detector baseline after checking code/license |

## C. Uncertainty, routing and training foundations

| Paper / source | Year | Problem, method | Dataset; model | Metrics / main finding | Limitation and distinction |
|---|---|---|---|---|---|
| [SelfCheckGPT](https://arxiv.org/abs/2303.08896), Manakul et al. | EMNLP 2023 | Sampling consistency for black-box factuality | WikiBio; GPT-3 | Sentence AUC-PR, passage correlation | Stable shared hallucinations; our caption token F1 is only a crude proxy, not this full method |
| [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599), Guo et al. | ICML 2017 | Temperature scaling/post-hoc probability calibration | Image/document classifiers | Calibration errors; temperature scaling effective | Fit on calibration set; cosine rescaling without labels is not calibration |
| [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500), Geifman and El-Yaniv | 2017 | Reject option with risk/coverage control | CIFAR/ImageNet; DNNs | Risk at target coverage/probability guarantee | Assumptions/calibration matter; zero errors on ten decided claims proves no guarantee |
| [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665), Ong et al. | 2024; revised 2025 | Preference-trained strong/weak model routing | General LLM benchmarks; exact pair details N/V | Cost/quality and cross-model transfer | Not visual failure-specific repair; budget-aware routing itself is not new |
| [Mitigating Hallucination ... via Robust Instruction Tuning](https://arxiv.org/abs/2306.14565), Liu et al. | ICLR 2024 | LRV positive/negative instructions and GAVIE evaluation | LRV-Instruction; MiniGPT-4/mPLUG-Owl | Hallucination/general-task improvements | Synthetic instruction artifacts; benchmark domain shift needs testing |
| [LoRA](https://arxiv.org/abs/2106.09685), Hu et al. | 2021 preprint; ICLR 2022 | Freeze base weights, train low-rank updates | Language adaptation tasks; RoBERTa/DeBERTa/GPT | Quality, trainable parameters, memory/throughput | Efficient adaptation is not evidence of diagnosis learning; rank/data-size sweeps still needed |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), Rafailov et al. | 2023; revised 2024 | Preference objective without online RL reward optimization | Sentiment, summarization, dialogue; language models | Preference/task quality | Not a hallucination guarantee; use independently grounded chosen/rejected answers |

## What the literature does to our research question

The older statement "none routes by mechanism" is withdrawn. VISOR is a direct
counterexample; Woodpecker and FaithScore invalidate broader first-pipeline
claims. A falsifiable, narrower question remains:

> With the detector, candidate repair budget and final verifier held fixed,
> does predicted error-type routing improve independently judged answer
> correctness/preservation over uniform correction, surface-type routing and
> a taxonomy-free outcome policy on unseen images and generator families?

Testing this can produce a useful positive **or negative** result. Current
code and exploratory observations do not establish the positive result.

## Literature gaps to close before submission

- Full-text reproduction recipes and license/version checks for VISOR,
  Woodpecker, FaithScore, VCD, SpanCalib-VLM and VisER.
- A broader 2025–2026 venue search and forward/backward citation screening;
  this targeted search cannot establish absence of all closer work.
- Evidential deep-learning/subjective-logic methods: not implemented here;
  no Dirichlet evidence or formal conflict-mass claim is justified.
- LURE and other revisor methods require a separately verified citation and
  implementation before inclusion; no guessed venue/identifier is retained.
