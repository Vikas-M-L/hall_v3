"""
UniProbe wrapper — SCAFFOLD, NOT IMPLEMENTED.

=============================================================================
DO NOT FILL IN THE MODEL CALL BY GUESSING.

UniProbe is a recent release and its call signature has not been confirmed.
Writing a plausible-looking implementation here would produce a wrapper that
imports cleanly, runs without error, and returns meaningless numbers — which is
strictly worse than a stub that announces itself.

Before implementing, confirm from the actual repo / paper page:
  1. the real Hugging Face repo ID (config key `uniprobe.checkpoint`)
  2. what it consumes — hidden states? attention? logits? which layers?
  3. whether it scores per token, per span, or per sequence
  4. the orientation of its output: is a HIGH score hallucinated or faithful?
  5. whether it is VLM-specific, and whether Qwen2.5-VL is supported
Item 4 is the one that silently inverts your results if you get it wrong.
=============================================================================

Contract this wrapper must satisfy
----------------------------------
    input:   image, VLM internal trace (hidden states / attention), claim span
    output:  hallucination probability in [0, 1], where 1.0 = hallucinated

`VLMWrapper.hidden_states_for_text()` is available if the real model needs the
internal trace rather than the output text.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_WARNED: set[str] = set()

# Log-prob standard deviation that counts as "maximally dispersed" for the
# logprob_dispersion stand-in. Log-prob spread has no natural upper bound, so
# some scale is required; 2.0 nats is a guess and this mode is a smoke-test
# device, not a measurement. Named rather than inlined so it is at least visible.
DISPERSION_SCALE = 2.0


class UniProbeWrapper:
    """
    Modes
    -----
    placeholder        (default) returns a fixed constant so the pipeline runs
                       end-to-end before the real model is wired in
    logprob_dispersion a training-free heuristic stand-in. NOT UniProbe, and it
                       does not read attention at all — it uses the dispersion of
                       token log-probs across the claim span as a crude proxy for
                       internal uncertainty. Its only real use is smoke-testing
                       the fusion logic with a signal that actually varies.
    model              the real thing — TODO, see the header

    Both non-`model` modes are STUBS and report themselves as such through
    `is_stubbed`, which the runner forwards into `SignalBundle.stubbed` so the
    fusion layer can exclude them.
    """

    VALID_MODES = ("placeholder", "logprob_dispersion", "model")
    # The old name said "attention" while reading log-probs. Kept as an alias so
    # existing configs do not break, with a warning.
    _ALIASES = {"attention_entropy": "logprob_dispersion"}

    def __init__(self, cfg: dict, vlm=None):
        self.cfg = cfg
        self.ucfg = cfg["uniprobe"]
        self.vlm = vlm
        self.placeholder = float(self.ucfg.get("placeholder_value", 0.5))
        self.checkpoint = self.ucfg.get("checkpoint")
        self.dispersion_scale = float(
            self.ucfg.get("dispersion_scale", DISPERSION_SCALE)
        )

        raw_mode = self.ucfg.get("mode", "placeholder")
        if raw_mode in self._ALIASES:
            logger.warning(
                "uniprobe.mode %r is deprecated (it never read attention); "
                "use %r",
                raw_mode,
                self._ALIASES[raw_mode],
            )
        self.mode = self._ALIASES.get(raw_mode, raw_mode)
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"uniprobe.mode must be one of {self.VALID_MODES}, got {raw_mode!r}"
            )

        if self.mode == "model":
            if not self.checkpoint:
                raise ValueError(
                    "uniprobe.mode == 'model' but uniprobe.checkpoint is null. "
                    "Confirm the real repo ID before enabling this mode."
                )
            raise NotImplementedError(
                "UniProbe model loading is not implemented. See the module header "
                "for the five things to confirm before writing it."
            )

    @property
    def is_stubbed(self) -> bool:
        """True whenever this is not the real model."""
        return self.mode != "model"

    def score(
        self,
        image=None,
        claim_text: str = "",
        generation=None,
        tok_span: tuple[int, int] | None = None,
    ) -> float:
        """
        Returns hallucination probability in [0, 1]. Higher = more likely hallucinated.

        Parameters
        ----------
        generation : GenerationOutput from VLMWrapper.generate()
        tok_span   : (start, end) token indices of the claim within that generation
        """
        if self.mode == "placeholder":
            if "placeholder" not in _WARNED:
                logger.warning(
                    "UniProbe is STUBBED — returning constant %.2f for every claim. "
                    "This signal contributes no information until the real model is "
                    "wired in; set drop_stubbed: true to exclude it from fusion.",
                    self.placeholder,
                )
                _WARNED.add("placeholder")
            return self.placeholder

        if self.mode == "logprob_dispersion":
            if "dispersion" not in _WARNED:
                logger.warning(
                    "UniProbe is running the logprob_dispersion STAND-IN, not "
                    "UniProbe. Do not report these numbers as UniProbe results."
                )
                _WARNED.add("dispersion")
            return self._dispersion_proxy(generation, tok_span, self.dispersion_scale)

        # mode == "model"
        # ------------------------------------------------------------------
        # TODO: real UniProbe call. Expected shape, once confirmed:
        #
        #   trace = self.vlm.hidden_states_for_text(image, prompt, claim_text,
        #                                           layer=self.ucfg["layer"])
        #   out   = self._model(trace)            # <- signature unconfirmed
        #   return float(out.hallucination_prob)  # <- check orientation!
        # ------------------------------------------------------------------
        raise NotImplementedError("UniProbe model call not implemented")

    # ------------------------------------------------------------------------

    @staticmethod
    def _dispersion_proxy(generation, tok_span, scale: float) -> float:
        """
        Training-free stand-in. NOT UniProbe, and should never be reported as such.

        Rationale: a claim whose token log-probs are uniformly high is one the
        model committed to smoothly; high variance across the span suggests it
        hesitated somewhere inside the claim. Dispersion is normalized into
        [0, 1] by DISPERSION_SCALE, since log-prob spread has no natural bound.
        """
        if generation is None or tok_span is None:
            return float("nan")
        lps = generation.token_logprobs[tok_span[0] : tok_span[1]]
        if len(lps) < 2:
            return float("nan")
        return float(np.clip(float(np.std(lps)) / scale, 0.0, 1.0))
