"""
Qwen2.5-VL-7B-Instruct wrapper — frozen, inference only.

Provides the two things the stock `generate()` API does not give you cleanly:

  1. generation with per-token log-probs, and a way to map a character span in
     the generated text back to the token indices that produced it
  2. teacher-forced scoring of a fixed continuation

Both are pure inference. No gradients are computed anywhere in this file.

Implementation notes
--------------------
* Image-token expansion: the processor expands a single `<|image_pad|>` into
  hundreds of tokens. Computing a prefix length with the raw tokenizer will be
  wrong. Everything here measures lengths on *processed* input_ids.
* `min_pixels` / `max_pixels` are pinned in config and applied to every image so
  visual-token counts stay stable across calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    text: str
    token_ids: list[int]
    token_strings: list[str]
    token_logprobs: list[float]

    def logprob_span(self, tok_start: int, tok_end: int, mode: str = "mean") -> float:
        """Aggregate log-probs over [tok_start, tok_end)."""
        span = self.token_logprobs[tok_start:tok_end]
        if not span:
            return float("nan")
        return float(np.mean(span)) if mode == "mean" else float(np.sum(span))

    def confidence_span(self, tok_start: int, tok_end: int) -> float:
        """
        Geometric-mean token probability over the span, in [0, 1].

        exp(mean log p) is the natural [0,1] mapping of a log-prob aggregate and
        is length-invariant, which matters because claims vary a lot in length.
        """
        mean_lp = self.logprob_span(tok_start, tok_end, mode="mean")
        if np.isnan(mean_lp):
            return float("nan")
        return float(np.clip(np.exp(mean_lp), 0.0, 1.0))


@dataclass
class ScoreOutput:
    sum_logprob: float
    mean_logprob: float
    n_tokens: int
    token_logprobs: list[float] = field(default_factory=list)
    n_visual_tokens: int = 0


class VLMWrapper:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        vcfg = cfg["vlm"]
        self.min_pixels = vcfg["min_pixels"]
        self.max_pixels = vcfg["max_pixels"]
        self.checkpoint = vcfg["checkpoint"]

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[cfg.get("dtype", "bfloat16")]

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "attn_implementation": vcfg.get("attn_implementation", "sdpa"),
        }

        if vcfg.get("load_in_4bit", False):
            # NB: `from transformers import BitsAndBytesConfig` succeeds even
            # without bitsandbytes installed — it is a plain config dataclass, and
            # the real ImportError is raised later inside from_pretrained. So
            # check for the package explicitly rather than catching ImportError
            # around the import, which would never fire.
            import importlib.util

            if importlib.util.find_spec("bitsandbytes") is None:
                logger.warning("bitsandbytes not installed; loading unquantized")
            elif not torch.cuda.is_available():
                logger.warning("no CUDA device; 4-bit unavailable, loading unquantized")
            else:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

        logger.info("loading %s ...", self.checkpoint)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint, **load_kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        self.tokenizer = self.processor.tokenizer
        self._image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _messages(image: Image.Image | None, prompt: str) -> list[dict]:
        content: list[dict] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _prepare(self, image: Image.Image | None, text: str) -> dict[str, torch.Tensor]:
        images = [image] if image is not None else None
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _render_prefix(self, image: Image.Image | None, prompt: str) -> str:
        return self.processor.apply_chat_template(
            self._messages(image, prompt), tokenize=False, add_generation_prompt=True
        )

    def _count_visual_tokens(self, input_ids: torch.Tensor) -> int:
        if self._image_token_id is None or self._image_token_id < 0:
            return 0
        return int((input_ids == self._image_token_id).sum().item())

    # ------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image | None,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationOutput:
        text = self._render_prefix(image, prompt)
        inputs = self._prepare(image, text)
        n_prompt = inputs["input_ids"].shape[1]

        temp = self.cfg["vlm"]["temperature"] if temperature is None else temperature
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.cfg["vlm"]["max_new_tokens"],
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if temp and temp > 0:
            gen_kwargs.update(do_sample=True, temperature=temp, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        out = self.model.generate(**inputs, **gen_kwargs)

        gen_ids = out.sequences[0, n_prompt:]
        # compute_transition_scores returns log-probs of the chosen tokens,
        # correctly accounting for any logits processors applied during sampling.
        scores = self.model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )[0]

        token_ids = gen_ids.tolist()
        token_logprobs = [float(s) for s in scores[: len(token_ids)].cpu()]
        token_strings = [self.tokenizer.decode([t]) for t in token_ids]

        return GenerationOutput(
            text=self.tokenizer.decode(token_ids, skip_special_tokens=True),
            token_ids=token_ids,
            token_strings=token_strings,
            token_logprobs=token_logprobs,
        )

    def char_span_to_token_indices(
        self, gen: GenerationOutput, char_start: int, char_end: int
    ) -> tuple[int, int]:
        """
        Map a character span in `gen.text` to token indices in `gen.token_ids`.

        Uses incremental decode rather than offset_mapping: the generated ids are
        known exactly, and offset mappings are unreliable across the special-token
        boundaries Qwen inserts.

        Note the decoded token strings here include no special tokens, while
        `gen.text` was decoded with skip_special_tokens=True — so the cumulative
        lengths line up as long as no special token falls inside the span.
        """
        cursor = 0
        tok_start, tok_end = 0, len(gen.token_ids)
        started = False
        for i, tok in enumerate(gen.token_strings):
            nxt = cursor + len(tok)
            if not started and nxt > char_start:
                tok_start, started = i, True
            if started and cursor >= char_end:
                tok_end = i
                break
            cursor = nxt
        return tok_start, max(tok_end, tok_start + 1)

    # ---------------------------------------------------------------- scoring

    @torch.no_grad()
    def score_continuation(
        self, image: Image.Image | None, prompt: str, continuation: str
    ) -> ScoreOutput:
        """
        Teacher-forced log P(continuation | image, prompt). Pure inference.

        Not used by the default v1/v2 fusion, but kept because it is the
        primitive behind image-ablation signals (score the same claim with and
        without the image) and costs nothing to leave here.
        """
        prefix_text = self._render_prefix(image, prompt)
        full_text = prefix_text + continuation

        prefix_ids = self._prepare(image, prefix_text)["input_ids"]
        n_prefix = prefix_ids.shape[1]
        full_inputs = self._prepare(image, full_text)
        n_full = full_inputs["input_ids"].shape[1]
        if n_full <= n_prefix:
            return ScoreOutput(float("nan"), float("nan"), 0, [], 0)

        # The whole slice below assumes the prefix tokenizes identically inside
        # the full sequence. That holds for image-token expansion (same image,
        # pinned min/max_pixels -> identical grid), but BPE can merge across the
        # prefix/continuation boundary. Fail loudly rather than score the wrong
        # tokens: a silent off-by-a-few here corrupts every confidence value.
        if not torch.equal(full_inputs["input_ids"][0, :n_prefix], prefix_ids[0]):
            logger.warning(
                "prefix retokenized inside the full sequence (BPE merge at the "
                "boundary); returning nan rather than scoring the wrong span"
            )
            return ScoreOutput(float("nan"), float("nan"), 0, [], 0)

        out = self.model(**full_inputs)

        # Slice BEFORE casting to float32. Qwen2.5-VL's vocab is 152k and the
        # sequence is ~1300 visual tokens plus text, so casting the full logits
        # tensor would allocate ~850 MB, and log_softmax another ~850 MB, to read
        # 5-20 positions. That is the most likely OOM on a 16 GB card.
        # logits[t] predicts token t+1, so continuation [n_prefix, n_full)
        # unshifted becomes [n_prefix-1, n_full-1) shifted.
        lo, hi = n_prefix - 1, n_full - 1
        logits = out.logits[0, lo:hi, :].float()
        targets = full_inputs["input_ids"][0, n_prefix:n_full]
        tok_lp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        vals = [float(v) for v in tok_lp.cpu()]

        return ScoreOutput(
            sum_logprob=float(np.sum(vals)),
            mean_logprob=float(np.mean(vals)) if vals else float("nan"),
            n_tokens=len(vals),
            token_logprobs=vals,
            n_visual_tokens=self._count_visual_tokens(full_inputs["input_ids"]),
        )

    @torch.no_grad()
    def hidden_states_for_text(
        self, image: Image.Image | None, prompt: str, continuation: str, layer: int = -12
    ) -> torch.Tensor:
        """
        Hidden states over the continuation span. Exposed for uniprobe_wrapper,
        which may need the internal trace rather than just the output text.
        """
        prefix_text = self._render_prefix(image, prompt)
        n_prefix = self._prepare(image, prefix_text)["input_ids"].shape[1]
        full_inputs = self._prepare(image, prefix_text + continuation)
        out = self.model(**full_inputs, output_hidden_states=True)
        return out.hidden_states[layer][0][n_prefix:].float().cpu()
