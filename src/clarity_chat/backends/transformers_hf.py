"""The Transformers backend: the model in-process, on MPS (default), CPU or CUDA, in bf16.

Install with `uv sync --extra hf`. It loads the official weights that `fetch` downloaded
for the convert route (the same repo and commit the GGUF came from) and feeds them the
same token IDs as llama-server.

Every `GenerationConfig` field that can change the output is set explicitly; Transformers
otherwise fills unset fields from the model's `generation_config.json` and then from its
own defaults (`top_k` 50, for one). The model's own `generation_config` is replaced by an
empty one at load, and after every call the effective config is checked field by field.
Only `eos_token_id` comes from the model's `generation_config.json`, and it is logged.

Transformers applies its processors in a fixed order: repetition penalty, then
temperature, top_k, top_p and min_p. A profile whose `samplers` lists them in another
order runs in this order, and the record says so under `not_applied`. presence_penalty,
frequency_penalty and repeat_last_n have no Transformers equivalent: not applied.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clarity_chat.backends import BackendError, Result
from clarity_chat.logs import machine, new_id
from clarity_chat.profiles import ConfigError, ModelEntry

NAME = "transformers"
# Set, not left to the library: with None Transformers picks sdpa but falls back to eager
# without saying so. Analysis tools that reload a call need to know which one ran.
ATTN_IMPLEMENTATION = "sdpa"
FIXED_ORDER = ["penalties", "temperature", "top_k", "top_p", "min_p"]

# Fields Clarity sets to a fixed value on every call (logged with origin "backend").
NEUTRAL_CONFIG: dict[str, Any] = {
    "num_beams": 1,
    "num_beam_groups": 1,
    "num_return_sequences": 1,
    "min_length": 0,
    "min_new_tokens": 0,
    "typical_p": 1.0,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
    "top_h": None,
    "encoder_repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0,
    "diversity_penalty": 0.0,
    "renormalize_logits": False,
    "remove_invalid_values": False,
    "bad_words_ids": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "sequence_bias": None,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "exponential_decay_length_penalty": None,
    "guidance_scale": None,
    "watermarking_config": None,
    "penalty_alpha": None,
    "dola_layers": None,
    "token_healing": None,
    "max_time": None,
    "use_cache": True,
    "cache_implementation": None,
    "output_scores": False,
    "output_logits": False,
    "return_dict_in_generate": False,
}
# Fields that can't change a single greedy or sampled sequence without an assistant model
# (speculative decoding), beams, compilation or batching; their values are only logged.
INERT_FIELDS = {
    "max_length",  # max_new_tokens takes precedence
    "early_stopping",  # beam search only
    "num_assistant_tokens",
    "num_assistant_tokens_schedule",
    "assistant_confidence_threshold",
    "assistant_lookbehind",
    "target_lookbehind",
    "assistant_early_exit",
    "assistant_ensemble_weight",
    "is_assistant",
    "speculation_type",
    "prompt_lookup_num_tokens",
    "max_matching_ngram_size",
    "compile_config",
    "disable_compile",
    "continuous_batching_config",
    "low_memory",
    "prefill_chunk_size",
    "output_attentions",
    "output_hidden_states",
    "cache_config",
    "max_cache_len",
    "constraints",
    "force_words_ids",
    "decoder_start_token_id",
    "bos_token_id",
    "transformers_version",
    "_from_model_config",
}
# Environment variables that change how PyTorch computes on MPS. Their values are logged
# (like GGML_* for llama.cpp), and any set at all is a start warning.
TORCH_ENV_PREFIXES = ("PYTORCH_", "TORCH_")


def torch_env() -> dict[str, str]:
    from clarity_chat.backends.llama_cpp import REDACTED, SECRET_NAME

    return {
        k: REDACTED if SECRET_NAME.search(k) else v
        for k, v in sorted(os.environ.items())
        if k.startswith(TORCH_ENV_PREFIXES)
    }


def generation_values(
    settings: dict[str, Any], stop: list[str], eos_ids: list[int]
) -> dict[str, Any]:
    """Every output-affecting GenerationConfig field for one call."""
    temperature = settings["temperature"]
    sampling = temperature > 0
    return {
        **NEUTRAL_CONFIG,
        "max_new_tokens": settings["max_tokens"],
        "do_sample": sampling,
        # Greedy decoding ignores the warpers, so they are set to their "off" values.
        "temperature": temperature if sampling else 1.0,
        "top_k": settings["top_k"] if sampling else 0,
        "top_p": settings["top_p"] if sampling else 1.0,
        # min_p 0 means off; None adds no processor at all.
        "min_p": settings["min_p"] if sampling and settings["min_p"] > 0 else None,
        "repetition_penalty": settings["repeat_penalty"],
        "stop_strings": list(stop) or None,
        "eos_token_id": list(eos_ids),
        "pad_token_id": eos_ids[0],
    }


def active_samplers(settings: dict[str, Any]) -> list[str]:
    """The listed samplers that change anything at these values."""
    if settings["temperature"] <= 0:
        return []  # greedy: no warper runs
    off = {
        "penalties": settings["repeat_penalty"] == 1.0,
        "temperature": settings["temperature"] == 1.0,
        "top_k": settings["top_k"] == 0,
        "top_p": settings["top_p"] >= 1.0,
        "min_p": settings["min_p"] <= 0,
    }
    return [s for s in settings["samplers"] if s in off and not off[s]]


def not_applied(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Settings Transformers can't apply as written, with the reason. `affects_output`
    says whether, at these values, the difference can change the reply."""
    out = []
    order = [s for s in settings["samplers"] if s in FIXED_ORDER]
    used = [s for s in FIXED_ORDER if s in order]
    if order != used:
        active = active_samplers(settings)
        out.append(
            {
                "name": "samplers",
                "value": settings["samplers"],
                "reason": f"Transformers applies its processors in a fixed order; used {used}",
                "affects_output": active != [s for s in FIXED_ORDER if s in active],
            }
        )
    for name in ("presence_penalty", "frequency_penalty"):
        out.append(
            {
                "name": name,
                "value": settings[name],
                "reason": "no Transformers equivalent",
                "affects_output": settings[name] != 0,
            }
        )
    out.append(
        {
            "name": "repeat_last_n",
            "value": settings["repeat_last_n"],
            "reason": "Transformers' repetition penalty covers the whole sequence",
            "affects_output": settings["repeat_penalty"] != 1.0,
        }
    )
    return out


def check_config(effective: dict[str, Any], sent: dict[str, Any]) -> dict[str, Any]:
    """Every field of the config generate() would use is what Clarity sent, or inert."""
    mismatched = {
        k: {"sent": sent[k], "effective": effective.get(k)}
        for k in sent
        if k in effective and effective[k] != sent[k]
    }
    unaccounted = {
        k: v
        for k, v in effective.items()
        if k not in sent and k not in INERT_FIELDS and v is not None
    }
    return {
        "ok": not mismatched and not unaccounted,
        "mismatched": mismatched,
        "unaccounted": unaccounted,
    }


def cut_at_stop(text: str, stop: list[str]) -> tuple[str, str]:
    """The text before the earliest stop string, and that string ("" if none)."""
    found = [(text.index(s), s) for s in stop if s in text]
    if not found:
        return text, ""
    index, word = min(found)
    return text[:index], word


def held_back(text: str, stop: list[str]) -> int:
    """How many trailing characters could start a stop string (not streamed yet)."""
    longest = 0
    for word in stop:
        for n in range(min(len(word) - 1, len(text)), 0, -1):
            if text.endswith(word[:n]):
                longest = max(longest, n)
                break
    return longest


class TransformersBackend:
    name = NAME

    def __init__(self, device: str, n_ctx: int) -> None:
        self.device = device
        self._n_ctx = n_ctx
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_entry: ModelEntry | None = None
        self.weights: Path | None = None
        self.eos_ids: list[int] = []
        self.record: dict[str, Any] | None = None

    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    @property
    def server_record(self) -> dict[str, Any] | None:
        return self.record

    def start(self, model: ModelEntry, weights: Path) -> dict[str, Any] | None:
        if self.model is not None and self.weights == weights:
            return None
        self.close()
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        except ImportError as e:
            raise ConfigError(
                "The Transformers backend needs torch: run `uv sync --extra hf`."
            ) from e
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ConfigError(
                'device "mps" is not available here (no Apple GPU, or the process can\'t see '
                'it). Set device = "cpu" in the profile to run on the CPU.'
            )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ConfigError('device "cuda" is not available. Set device = "cpu" or "mps".')
        env = torch_env()
        if self.device == "mps" and env.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
            raise ConfigError(
                "PYTORCH_ENABLE_MPS_FALLBACK is set, so operations MPS lacks would run on the "
                'CPU without notice. Unset it, or set device = "cpu".'
            )
        generation_file = weights / "generation_config.json"
        if not generation_file.is_file():
            raise ConfigError(f"{generation_file} is missing: no eos_token_id to stop on.")
        own = GenerationConfig.from_pretrained(weights)
        eos = own.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        if not self.eos_ids:
            raise ConfigError(f"{generation_file} has no eos_token_id.")
        self.tokenizer = AutoTokenizer.from_pretrained(weights)
        loaded = AutoModelForCausalLM.from_pretrained(
            weights, dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION
        )
        attention = getattr(loaded.config, "_attn_implementation", None)
        if attention != ATTN_IMPLEMENTATION:
            raise ConfigError(
                f"{model.name} loaded with attention {attention!r}, not "
                f"{ATTN_IMPLEMENTATION!r} as requested."
            )
        loaded.generation_config = GenerationConfig()  # nothing of the model's merges in
        loaded.to(self.device)
        loaded.eval()
        positions = getattr(loaded.config, "max_position_embeddings", None)
        if positions is not None and positions < self._n_ctx:
            raise ConfigError(
                f"{model.name} handles {positions} positions, less than ctx-size {self._n_ctx}."
            )
        self.model, self.model_entry, self.weights = loaded, model, weights
        warnings = []
        deterministic = torch.are_deterministic_algorithms_enabled()
        if deterministic:
            # Clarity never turns it on (it makes nothing in inference deterministic on MPS or
            # CPU, and can only raise errors); transformers' set_seed(deterministic=True) does.
            warnings.append(
                "torch.use_deterministic_algorithms is on (something in this process turned "
                "it on). It can make operations raise errors; Clarity doesn't need it."
            )
        if env:
            warnings.append(
                f"PyTorch inherits {list(env)} from your environment. Some PYTORCH_* "
                "variables change how MPS computes; unset them for comparable results."
            )
        self.record = {
            "type": "server_start",
            "server_id": new_id("backend"),
            "backend": NAME,
            "model": model.to_json(),
            "weights": weights.as_posix(),
            "device": self.device,
            "dtype": "bfloat16",
            "attn_implementation": ATTN_IMPLEMENTATION,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "eos_token_id": self.eos_ids,
            "max_position_embeddings": positions,
            "env_torch_kept": env,
            "deterministic_algorithms": deterministic,
            "warnings": warnings,
            "machine": machine(),
        }
        return self.record

    def describe(self) -> dict[str, Any]:
        record = self.record or {}
        return {
            "backend": NAME,
            "server_id": record.get("server_id"),
            "device": self.device,
            "dtype": "bfloat16",
            "attn_implementation": ATTN_IMPLEMENTATION,
            "torch": record.get("torch"),
            "transformers": record.get("transformers"),
        }

    def complete(
        self,
        token_ids: list[int],
        settings: dict[str, Any],
        stop: list[str],
        on_text: Callable[[str], None] | None = None,
        request: dict[str, Any] | None = None,
    ) -> Result:
        if self.model is None:
            raise BackendError("No model loaded.")
        import torch
        from transformers import GenerationConfig
        from transformers.generation.streamers import BaseStreamer

        values = (
            dict(request["generation_config"])
            if request is not None
            else generation_values(settings, stop, self.eos_ids)
        )
        seed = request["seed"] if request is not None else settings["seed"]
        logged_request = {"generation_config": values, "seed": seed, "device": self.device}
        try:
            config = GenerationConfig(**values)
            effective, _ = self.model._prepare_generation_config(config)
        except Exception as e:  # e.g. a logged config this Transformers version rejects
            raise BackendError(
                f"Transformers rejected the generation config: {type(e).__name__}: {e}",
                Result(text="", token_ids=[], stop_type="none", request=logged_request),
            ) from e
        check = check_config(effective.to_dict(), values)
        backend_settings = {
            k: {"value": v, "origin": "backend"} for k, v in NEUTRAL_CONFIG.items()
        } | {"eos_token_id": {"value": self.eos_ids, "origin": "model"}}
        stop_words = values["stop_strings"] or []
        ids: list[int] = []
        shown = 0
        first_token_at: list[float] = []
        decode = self.tokenizer.decode

        def visible() -> str:
            text = decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            text, word = cut_at_stop(text, stop_words)
            if not word:
                text = text[: len(text) - held_back(text, stop_words)]
            return text.rstrip("\ufffd")  # a character still incomplete

        class Collect(BaseStreamer):
            first = True

            def put(self, value: Any) -> None:
                nonlocal shown
                if self.first:  # the prompt
                    self.first = False
                    return
                if not first_token_at:
                    first_token_at.append(time.perf_counter())
                ids.extend(int(t) for t in value.reshape(-1).tolist())
                if on_text:
                    text = visible()
                    if len(text) > shown:
                        on_text(text[shown:])
                        shown = len(text)

            def end(self) -> None:
                pass

        cancelled = False
        started = time.perf_counter()
        try:
            torch.manual_seed(seed)
            if self.device == "mps":
                torch.mps.manual_seed(seed)
            inputs = torch.tensor([token_ids], device=self.device)
            with torch.inference_mode():
                self.model.generate(
                    inputs,
                    attention_mask=torch.ones_like(inputs),
                    generation_config=config,
                    tokenizer=self.tokenizer,
                    streamer=Collect(),
                )
        except KeyboardInterrupt:
            cancelled = True
        except Exception as e:
            raise BackendError(
                f"Transformers generate failed: {type(e).__name__}: {e}",
                Result(text="", token_ids=list(ids), stop_type="none", request=logged_request),
            ) from e
        ended = time.perf_counter()
        first = first_token_at[0] if first_token_at else ended
        # llama-server's names; prompt_ms here is the time to the first generated token.
        timings = {
            "prompt_n": len(token_ids),
            "prompt_ms": (first - started) * 1000,
            "predicted_n": len(ids),
            "predicted_ms": (ended - first) * 1000,
        }
        full = decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        text, word = cut_at_stop(full, stop_words)
        if cancelled:
            stop_type = "cancelled"
        elif ids and ids[-1] in self.eos_ids:
            stop_type = "eos"
        elif word:
            stop_type = "word"
        elif len(ids) >= values["max_new_tokens"]:
            stop_type = "limit"
        else:
            stop_type = "none"
        if on_text and not cancelled and len(text) > shown:
            on_text(text[shown:])
        flags = ["cancelled"] if cancelled else []
        if stop_type == "limit":
            flags.append("length")
        if not check["ok"]:
            flags.append("check_failed:generation_config")
        return Result(
            text=text,
            token_ids=ids,
            stop_type=stop_type,
            stopping_word=word,
            n_prompt=len(token_ids),
            n_predicted=len(ids),
            timings=timings,
            request=logged_request,
            response={"effective_generation_config": effective.to_dict()},
            backend_settings=backend_settings,
            not_applied=not_applied(settings),
            flags=flags,
            checks={"generation_config": check},
        )

    def close(self) -> None:
        if self.model is None:
            return
        self.model = None
        self.tokenizer = None
        self.weights = None
        import gc

        gc.collect()
        try:
            import torch

            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass
