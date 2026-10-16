"""OpenAI-backed corrector with a local, content-addressed response cache.

The API key is read only at runtime from a user-owned mode-600 file. It is
never placed in configuration, result files, prompts, or cache keys.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI

try:
    from .llm import Generation
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    # GPT-only worker hosts need not install the Qwen/Torch stack merely to
    # carry generation metadata.
    from dataclasses import dataclass
    @dataclass
    class Generation:
        text: str
        prompt_tokens: int = 0
        completion_tokens: int = 0
        elapsed_s: float = 0.0
        cache_hit: bool = False


SYSTEM_PROMPT = (
    "You are correcting text corrupted by transmission noise. Each character is "
    "printable ASCII. Noise substitutes characters with wrong ones. Fix only the "
    "corrupted characters. Do NOT paraphrase, reorder, insert, or delete anything. "
    "Output ONLY the corrected text, nothing else."
)
FEWSHOT_EXAMPLES = [
    ("I hatd a questyoo awound it.", "I have a question around it."),
    ("Tha economr is growint slowla thia yaar.", "The economy is growing slowly this year."),
]


class OpenAICorrector:
    """Interface-compatible corrector for the current receiver pipeline."""

    def __init__(self, model_id: str, cache_dir: str | Path, api_key_file: str | Path | None,
                 few_shot: bool = True):
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key and api_key_file:
            key = Path(api_key_file).read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("Set OPENAI_API_KEY or OPENAI_API_KEY_FILE")
        # Avoid optional zstd response decoding: several worker hosts carry an
        # older system zstandard binding that is incompatible with httpx2.
        self.client = OpenAI(api_key=key, default_headers={"Accept-Encoding": "gzip, deflate"})
        self.model_id, self.few_shot = model_id, few_shot
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _messages(self, corrupted_text: str) -> list[dict[str, str]]:
        length = len(corrupted_text)
        parts = []
        if self.few_shot:
            for received, corrected in FEWSHOT_EXAMPLES:
                parts.append(
                    f"The text is exactly {len(received)} characters long. Your output must be "
                    f"exactly {len(received)} characters. Correct only characters that look like "
                    f'noise-induced typos.\n\nInput:  "{received}"\nOutput: "{corrected}"'
                )
        parts.append(
            f"The text is exactly {length} characters long. Your output must be exactly {length} "
            f"characters. Correct only characters that look like noise-induced typos.\n\n"
            f'Input: "{corrupted_text}"\nOutput:'
        )
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(parts)}]

    @staticmethod
    def _received_from_pipeline_prompt(prompt: str) -> str:
        start, end = "<received_text>\n", "\n</received_text>"
        if start not in prompt or end not in prompt:
            raise ValueError("receiver prompt lacks delimited received text")
        return prompt.split(start, 1)[1].split(end, 1)[0]

    @staticmethod
    def _extract_output(raw: str) -> str:
        text = raw.strip()
        if text.startswith('"') and text.count('"') >= 2:
            text = text[1:text.rfind('"')]
        elif text.startswith('"'):
            text = text[1:]
        return text.split("\n", 1)[0].strip('"')

    def generate(self, prompt: str, *, do_sample: bool, seed: int, max_new_tokens: int = 512,
                 temperature: float | None = None, top_p: float | None = None,
                 top_k: int | None = None, prompt_version: str = "openai-v1",
                 use_cache: bool = True) -> Generation:
        del seed, top_k  # Chat Completions does not provide reproducible sampling for this model.
        received = self._received_from_pipeline_prompt(prompt)
        settings = {"provider": "openai", "model_id": self.model_id, "few_shot": self.few_shot,
                    "prompt_version": prompt_version, "received": received, "do_sample": do_sample,
                    "temperature": temperature if do_sample else 0.0, "top_p": top_p if do_sample else None,
                    "max_tokens": min(max_new_tokens, len(received) + 32)}
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if use_cache and path.exists():
            generation = dict(json.loads(path.read_text())["generation"])
            generation["cache_hit"] = True
            return Generation(**generation)
        kwargs = {"model": self.model_id, "messages": self._messages(received),
                  "temperature": settings["temperature"], "max_tokens": settings["max_tokens"]}
        if settings["top_p"] is not None:
            kwargs["top_p"] = settings["top_p"]
        started = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - started
        raw = response.choices[0].message.content or ""
        usage = response.usage
        result = Generation(self._extract_output(raw), int(usage.prompt_tokens if usage else 0),
                            int(usage.completion_tokens if usage else 0), elapsed, False)
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps({"settings": settings, "generation": asdict(result)}, ensure_ascii=False) + "\n")
        temporary.replace(path)
        return result

    def generate_freeform(self, prompt: str, *, do_sample: bool, seed: int,
                          max_new_tokens: int = 512, temperature: float | None = None,
                          top_p: float | None = None, prompt_version: str = "openai-freeform-v1",
                          use_cache: bool = True) -> Generation:
        """Generate for a non-proposed-template prompt with the same durable cache.

        This is used only by the masked-character voting baseline.  Unlike
        :meth:`generate`, it deliberately does not parse a ``<received_text>``
        wrapper or alter the model's response: exact-length acceptance is the
        baseline evaluation rule.
        """
        # Chat Completions does not honour a reproducibility seed for this
        # model, but the caller's sample id must remain part of the cache key:
        # the three voting draws are independent API samples, not aliases of
        # one cached response.
        settings = {"provider": "openai", "model_id": self.model_id, "few_shot": False,
                    "prompt_version": prompt_version, "prompt": prompt, "do_sample": do_sample,
                    "temperature": temperature if do_sample else 0.0,
                    "top_p": top_p if do_sample else None, "max_tokens": max_new_tokens,
                    "sample_id": seed}
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if use_cache and path.exists():
            generation = dict(json.loads(path.read_text())["generation"])
            generation["cache_hit"] = True
            return Generation(**generation)
        kwargs = {"model": self.model_id,
                  "messages": [{"role": "system", "content": "Follow the user's output-format instruction exactly."},
                               {"role": "user", "content": prompt}],
                  "temperature": settings["temperature"], "max_tokens": max_new_tokens}
        if settings["top_p"] is not None:
            kwargs["top_p"] = settings["top_p"]
        started = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - started
        raw = response.choices[0].message.content or ""
        usage = response.usage
        result = Generation(raw, int(usage.prompt_tokens if usage else 0),
                            int(usage.completion_tokens if usage else 0), elapsed, False)
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps({"settings": settings, "generation": asdict(result)}, ensure_ascii=False) + "\n")
        temporary.replace(path)
        return result
