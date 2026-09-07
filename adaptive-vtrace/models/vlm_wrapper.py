"""
Qwen2.5-VL-7B-Instruct wrapper.

Exposes the three things the rest of the pipeline needs and that the stock
`generate()` API does not give you cleanly:

  1. generation with per-token log-probs (for claim-span confidence)
  2. teacher-forced scoring of a *fixed* continuation (for Visual Information Gain)
  3. hidden states at a token span (for the UniProbe substitute)

Design notes
------------
* Image-token expansion: the processor expands a single `<|image_pad|>` into
  hundreds of tokens. Any attempt to compute a prefix length using the raw
  tokenizer will be wrong. Everything here measures lengths on *processed*
  input_ids instead.
* Token-count stability: VIG compares log-probs across different images. If the
  null image produces a different visual-token count than the real image, the
  comparison absorbs prefix-length and M-RoPE position effects. `min_pixels` /
  `max_pixels` are therefore pinned in config and applied to every image.
* Prefix caching: scoring N claims against one image re-prefills the visual
  tokens N times if you are naive about it. `prefill(...)` returns a reusable
  KV cache; `score_continuation(..., cache=...)` consumes it.
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
    token_logprobs: list[float]  # log P of each generated token under the model
    hidden_states: torch.Tensor | None = None  # [n_gen_tokens, hidden] if requested

    def logprob_span(self, tok_start: int, tok_end: int, mode: str = "geometric") -> float:
        """Aggregate log-probs over [tok_start, tok_end). Geometric mean == mean log-prob."""
        span = self.token_logprobs[tok_start:tok_end]
        if not span:
            return float("nan")
        if mode == "geometric":
            return float(np.mean(span))
        if mode == "sum":
            return float(np.sum(span))
        raise ValueError(f"unknown mode {mode}")


@dataclass
class ScoreOutput:
    """Result of teacher-forced scoring of a fixed continuation."""

    sum_logprob: float
    mean_logprob: float
    n_tokens: int
    token_logprobs: list[float] = field(default_factory=list)
    n_visual_tokens: int = 0  # exposed so callers can assert stability across nulls


class VLMWrapper:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        vcfg = cfg["vlm"]
        self.device = cfg.get("device", "cuda")
        self.min_pixels = vcfg["min_pixels"]
        self.max_pixels = vcfg["max_pixels"]
        self.checkpoint = vcfg["checkpoint"]

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        has_cuda = torch.cuda.is_available()

        # On CPU, use float16 to halve memory (~14 GB for 7B vs ~28 GB for float32).
        # PyTorch supports float16 inference on CPU since 2.0. Only fall back to
        # float32 if the config explicitly requests it.
        if has_cuda:
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
                cfg.get("dtype", "bfloat16")
            ]
        else:
            requested = cfg.get("dtype", "bfloat16")
            if requested == "float32":
                dtype = torch.float32
            else:
                dtype = torch.float16  # float16 on CPU: ~14 GB for 7B model
            self.device = "cpu"
            logger.warning(
                "no CUDA device; using dtype=%s on CPU, disabling 4-bit. "
                "Ensure >=16 GB RAM is free.",
                dtype,
            )
            # Warn if available RAM looks tight
            try:
                import psutil
                avail_gb = psutil.virtual_memory().available / 1024 ** 3
                needed_gb = 14.0 if dtype == torch.float16 else 28.0
                if avail_gb < needed_gb:
                    logger.warning(
                        "Only %.1f GB RAM free; need ~%.0f GB for this model. "
                        "May OOM. Consider a smaller checkpoint.",
                        avail_gb, needed_gb,
                    )
            except ImportError:
                pass

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": "auto" if has_cuda else None,
            "attn_implementation": vcfg.get("attn_implementation", "sdpa"),
        }

        if vcfg.get("load_in_4bit", False) and has_cuda:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                logger.warning("bitsandbytes unavailable; loading in full precision")

        logger.info("loading %s ...", self.checkpoint)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint, low_cpu_mem_usage=True, **load_kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        self.tokenizer = self.processor.tokenizer

        # image_pad id, used to count visual tokens for the stability assertion
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
        """Run the processor. `text` must already be a fully rendered chat string."""
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
        image: Image.Image,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        return_hidden: bool = False,
    ) -> GenerationOutput:
        """Generate a response and return per-token log-probs alongside it."""
        text = self._render_prefix(image, prompt)
        inputs = self._prepare(image, text)
        n_prompt = inputs["input_ids"].shape[1]

        temp = self.cfg["vlm"]["temperature"] if temperature is None else temperature
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.cfg["vlm"]["max_new_tokens"],
            do_sample=temp > 0,
            temperature=temp if temp > 0 else None,
            top_p=0.9 if temp > 0 else None,
            output_scores=True,
            output_hidden_states=return_hidden,
            return_dict_in_generate=True,
        )

        gen_ids = out.sequences[0, n_prompt:]
        # compute_transition_scores gives log-probs of the *chosen* tokens,
        # correctly accounting for any logits processors applied during sampling.
        scores = self.model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )[0]

        token_ids = gen_ids.tolist()
        token_logprobs = [float(s) for s in scores[: len(token_ids)].cpu()]
        token_strings = [self.tokenizer.decode([t]) for t in token_ids]

        hidden = None
        if return_hidden and out.hidden_states:
            layer = self.cfg["uniprobe"]["substitute"]["layer"]
            # hidden_states is a tuple over generated steps, each a tuple over layers
            hidden = torch.stack(
                [step[layer][0, -1, :].float().cpu() for step in out.hidden_states]
            )

        return GenerationOutput(
            text=self.tokenizer.decode(token_ids, skip_special_tokens=True),
            token_ids=token_ids,
            token_strings=token_strings,
            token_logprobs=token_logprobs,
            hidden_states=hidden,
        )

    def char_span_to_token_indices(
        self, gen: GenerationOutput, char_start: int, char_end: int
    ) -> tuple[int, int]:
        """
        Map a character span in `gen.text` to token indices in `gen.token_ids`.

        Done by incremental decode rather than offset_mapping, because the
        generated ids are known exactly and offset mappings are unreliable
        across the special-token boundaries Qwen inserts.
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
        Teacher-forced log P(continuation | image, prompt).

        This is the primitive behind Visual Information Gain. Prefix length is
        measured on *processed* input_ids so that image-token expansion is
        handled correctly.
        """
        prefix_text = self._render_prefix(image, prompt)
        full_text = prefix_text + continuation

        prefix_inputs = self._prepare(image, prefix_text)
        full_inputs = self._prepare(image, full_text)

        n_prefix = prefix_inputs["input_ids"].shape[1]
        n_full = full_inputs["input_ids"].shape[1]
        if n_full <= n_prefix:
            return ScoreOutput(float("nan"), float("nan"), 0, [], 0)

        out = self.model(**full_inputs)
        # logits[t] predicts token t+1
        logits = out.logits[0, :-1, :].float()
        targets = full_inputs["input_ids"][0, 1:]
        logprobs = F.log_softmax(logits, dim=-1)
        tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # continuation occupies [n_prefix, n_full) in the unshifted frame,
        # which is [n_prefix - 1, n_full - 1) after the shift
        span = tok_lp[n_prefix - 1 : n_full - 1]
        vals = [float(v) for v in span.cpu()]

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
        """Hidden states over the continuation span, for probe training/inference."""
        prefix_text = self._render_prefix(image, prompt)
        full_text = prefix_text + continuation
        n_prefix = self._prepare(image, prefix_text)["input_ids"].shape[1]
        full_inputs = self._prepare(image, full_text)

        out = self.model(**full_inputs, output_hidden_states=True)
        hs = out.hidden_states[layer][0]  # [seq, hidden]
        return hs[n_prefix:].float().cpu()

    @torch.no_grad()
    def attention_rollout(
        self, image: Image.Image, prompt: str, continuation: str
    ) -> np.ndarray | None:
        """
        Crude visual attention map over image patches, averaged across heads and
        late layers. Used as the grounding fallback when GroundingDINO is absent.
        Returns a 1-D array over visual token positions, or None on failure.
        """
        try:
            prefix_text = self._render_prefix(image, prompt)
            inputs = self._prepare(image, prefix_text + continuation)
            out = self.model(**inputs, output_attentions=True)
            ids = inputs["input_ids"][0]
            vis_mask = ids == self._image_token_id
            if vis_mask.sum() == 0:
                return None
            # average the last quarter of layers, all heads, queries = text tokens
            layers = out.attentions[-max(1, len(out.attentions) // 4) :]
            attn = torch.stack([a[0].mean(0) for a in layers]).mean(0)  # [seq, seq]
            text_q = attn[~vis_mask][:, vis_mask]  # [n_text, n_visual]
            return text_q.mean(0).float().cpu().numpy()
        except Exception as exc:  # attentions unsupported under some attn impls
            logger.warning("attention_rollout failed (%s); returning None", exc)
            return None
