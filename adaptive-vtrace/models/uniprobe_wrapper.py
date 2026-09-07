"""
UniProbe wrapper — hallucination probability from internal representations.

=============================================================================
THE REAL UNIPROBE MODEL IS NOT WIRED IN.

UniProbe's Hugging Face repo ID and exact call signature have not been
confirmed.  See the reframed plan §10 Week 0.  Before enabling ``mode: model``,
verify:
  1. the real HF repo ID (config key ``uniprobe.checkpoint``)
  2. what it consumes — hidden states? attention? logits? which layers?
  3. whether it scores per token, per span, or per sequence
  4. output orientation: is HIGH hallucinated or faithful?
  5. whether Qwen2.5-VL is a supported backbone
=============================================================================

Three modes:

  placeholder         returns a fixed constant so the pipeline runs end-to-end
  substitute          (recommended) a self-trained mid-layer linear probe over
                      VLM hidden states — the plan's substitute for the
                      unverified external model
  model               scaffold — raises NotImplementedError until the real
                      UniProbe is confirmed

Contract
--------
    input:  image, VLM generation with token log-probs, claim token span
    output: hallucination probability in [0, 1], where 1.0 = hallucinated

``VLMWrapper.hidden_states_for_text()`` and ``GenerationOutput.hidden_states``
are both available for extracting the internal trace.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_WARNED: set[str] = set()

# Log-prob standard deviation that counts as "maximally dispersed" for the
# logprob_dispersion stand-in.
DISPERSION_SCALE = 2.0


class SubstituteProbe(nn.Module):
    """
    Small MLP probe trained on VLM hidden states to predict hallucination.

    Architecture: mean-pooled hidden states → Linear(hidden_dim, proj_dim)
    → ReLU → Linear(proj_dim, 1) → Sigmoid

    Trained by ``scripts/train_probe.py`` (separate script, not included here).
    This class only does inference over saved weights.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (hidden_dim,) or (batch, hidden_dim) → scalar probability."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


class UniProbeWrapper:
    """
    Modes
    -----
    placeholder          returns a fixed constant so the pipeline runs end-to-end
    substitute           a self-trained mid-layer linear probe over VLM hidden
                         states.  Loads weights from config if available, else
                         falls back to placeholder with a warning.
    logprob_dispersion   a training-free heuristic stand-in — uses the dispersion
                         of token log-probs across the claim span as a crude
                         proxy for internal uncertainty.  NOT UniProbe.
    model                the real thing — TODO, see the header
    """

    VALID_MODES = ("placeholder", "substitute", "logprob_dispersion", "model")
    _ALIASES = {"attention_entropy": "logprob_dispersion"}

    def __init__(self, cfg: dict, vlm=None):
        self.cfg = cfg
        self.ucfg = cfg["uniprobe"]
        self.vlm = vlm
        self.placeholder_value = float(
            self.ucfg.get("placeholder_value", 0.5)
            if "placeholder_value" in self.ucfg
            else 0.5
        )
        self.checkpoint = self.ucfg.get("checkpoint")

        # Resolve mode
        sub_cfg = self.ucfg.get("substitute", {})
        if self.checkpoint is not None:
            raw_mode = "model"
        elif sub_cfg.get("enabled", True):
            raw_mode = "substitute"
        else:
            raw_mode = "placeholder"

        self.mode = self._ALIASES.get(raw_mode, raw_mode)
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"uniprobe mode resolved to {self.mode!r}, which is not one of "
                f"{self.VALID_MODES}"
            )

        # Substitute probe config
        self.layer = sub_cfg.get("layer", -12)
        self.hidden_dim = sub_cfg.get("hidden_dim", 64)
        self.dispersion_scale = float(
            self.ucfg.get("dispersion_scale", DISPERSION_SCALE)
        )

        # Load substitute probe weights if available
        self._probe: SubstituteProbe | None = None
        if self.mode == "substitute":
            self._init_substitute_probe(sub_cfg)

        # Model mode scaffold
        if self.mode == "model":
            if not self.checkpoint:
                raise ValueError(
                    "uniprobe.checkpoint is set but empty. Confirm the real "
                    "repo ID before enabling model mode."
                )
            raise NotImplementedError(
                "UniProbe model loading is not implemented. See the module "
                "header for the five things to confirm before writing it."
            )

    def _init_substitute_probe(self, sub_cfg: dict) -> None:
        """Load the trained probe weights, or fall back to placeholder."""
        weights_path = sub_cfg.get("trained_weights")

        if weights_path is None:
            logger.warning(
                "UniProbe substitute mode enabled but no trained_weights path "
                "set.  Will use logprob_dispersion as a fallback signal until "
                "a probe is trained via scripts/train_probe.py."
            )
            self._probe = None
            return

        weights_path = Path(weights_path)
        if not weights_path.exists():
            logger.warning(
                "UniProbe substitute weights not found at %s; falling back to "
                "logprob_dispersion.  Train a probe with scripts/train_probe.py.",
                weights_path,
            )
            self._probe = None
            return

        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            # Infer input dimension from saved weights
            input_dim = state["net.0.weight"].shape[1]
            self._probe = SubstituteProbe(input_dim, self.hidden_dim)
            self._probe.load_state_dict(state)
            self._probe.eval()
            logger.info(
                "loaded substitute probe from %s (input_dim=%d, hidden=%d)",
                weights_path,
                input_dim,
                self.hidden_dim,
            )
        except Exception as exc:
            logger.warning(
                "failed to load substitute probe from %s (%s); falling back "
                "to logprob_dispersion",
                weights_path,
                exc,
            )
            self._probe = None

    @property
    def is_stubbed(self) -> bool:
        """True whenever this is not producing real probe-based scores."""
        if self.mode == "placeholder":
            return True
        if self.mode == "substitute" and self._probe is None:
            return True
        return False

    def score(
        self,
        image=None,
        claim_text: str = "",
        generation=None,
        tok_span: tuple[int, int] | None = None,
    ) -> float:
        """
        Returns hallucination probability in [0, 1]. Higher = more likely
        hallucinated.

        Parameters
        ----------
        image       : PIL Image (needed for substitute probe with no cached
                      hidden states)
        claim_text  : the claim string
        generation  : GenerationOutput from VLMWrapper.generate()
        tok_span    : (start, end) token indices of the claim within that
                      generation
        """
        if self.mode == "placeholder":
            if "placeholder" not in _WARNED:
                logger.warning(
                    "UniProbe is STUBBED — returning constant %.2f for every "
                    "claim. This signal contributes no information until a "
                    "probe is trained; set drop_stubbed: true to exclude it.",
                    self.placeholder_value,
                )
                _WARNED.add("placeholder")
            return self.placeholder_value

        if self.mode == "substitute":
            return self._substitute_score(image, claim_text, generation, tok_span)

        if self.mode == "logprob_dispersion":
            if "dispersion" not in _WARNED:
                logger.warning(
                    "UniProbe is running the logprob_dispersion STAND-IN, not "
                    "UniProbe.  Do not report these numbers as UniProbe results."
                )
                _WARNED.add("dispersion")
            return self._dispersion_proxy(generation, tok_span, self.dispersion_scale)

        # mode == "model"
        raise NotImplementedError("UniProbe model call not implemented")

    # ------------------------------------------------------------------ substitute

    def _substitute_score(
        self, image, claim_text, generation, tok_span
    ) -> float:
        """
        Score using the trained substitute probe over VLM hidden states.
        Falls back to logprob_dispersion if the probe is not loaded.
        """
        if self._probe is None:
            # No trained weights — use logprob_dispersion as the best available
            # untrained proxy
            if "substitute_fallback" not in _WARNED:
                logger.warning(
                    "Substitute probe has no trained weights; using "
                    "logprob_dispersion as fallback."
                )
                _WARNED.add("substitute_fallback")
            return self._dispersion_proxy(generation, tok_span, self.dispersion_scale)

        # Try to get hidden states from the generation output first (cheaper)
        hidden = None
        if generation is not None and hasattr(generation, "hidden_states"):
            hidden = generation.hidden_states

        if hidden is not None and tok_span is not None:
            # Extract hidden states for the claim span and mean-pool
            start, end = tok_span
            if start < hidden.shape[0] and end <= hidden.shape[0]:
                span_hidden = hidden[start:end]
                if span_hidden.shape[0] > 0:
                    pooled = span_hidden.mean(dim=0)
                    with torch.no_grad():
                        return float(self._probe(pooled).item())

        # Fallback: re-run the VLM to get hidden states for the claim text
        if self.vlm is not None and image is not None and claim_text:
            try:
                prompt = self.cfg.get("prompt", "Describe this image in detail.")
                hs = self.vlm.hidden_states_for_text(
                    image, prompt, claim_text, layer=self.layer
                )
                if hs.shape[0] > 0:
                    pooled = hs.mean(dim=0)
                    with torch.no_grad():
                        return float(self._probe(pooled).item())
            except Exception as exc:
                logger.warning("substitute probe scoring failed: %s", exc)

        return float("nan")

    # ------------------------------------------------------------------ dispersion

    @staticmethod
    def _dispersion_proxy(generation, tok_span, scale: float) -> float:
        """
        Training-free stand-in. NOT UniProbe, and should never be reported as
        such.

        Rationale: a claim whose token log-probs are uniformly high is one the
        model committed to smoothly; high variance across the span suggests it
        hesitated somewhere inside the claim. Dispersion is normalized into
        [0, 1] by ``scale``, since log-prob spread has no natural bound.
        """
        if generation is None or tok_span is None:
            return float("nan")
        lps = generation.token_logprobs[tok_span[0] : tok_span[1]]
        if len(lps) < 2:
            return float("nan")
        return float(np.clip(float(np.std(lps)) / scale, 0.0, 1.0))
