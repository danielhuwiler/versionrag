"""OpenAI LLM client with structured output support via instructor."""

import asyncio
import logging
import time
from contextlib import nullcontext
from typing import TypeVar

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

from versionrag.clients.usage_tracker import get_tracker

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Models that do not support the temperature parameter
_NO_TEMPERATURE_MODELS = frozenset({"gpt-5-nano", "gpt-5.4-nano"})


class LLMClient:
    """Wraps AsyncOpenAI with two modes: structured output and plain chat.

    Structured output uses ``instructor`` to validate and retry LLM responses
    against a Pydantic model. Plain chat is a direct API call for free-form text.

    The model is NOT fixed in the constructor — each call receives a ``model``
    parameter so a single client instance can serve all three model tiers
    (fast / standard / strong) configured in settings.

    Provider routing: model names containing ``/`` (e.g. ``moonshotai/kimi-k2``)
    are sent through the OpenRouter client when ``openrouter_api_key`` is set;
    all other model names go through the direct OpenAI client as before. This
    keeps the judge (gpt-5.4) on direct OpenAI while allowing the small-model
    tier to be swapped to any OpenRouter-hosted model.
    """

    def __init__(
        self,
        api_key: str,
        temperature: float = 0.0,
        default_extract_model: str = "gpt-5-nano",
        default_chat_model: str = "gpt-4.1-mini",
        openrouter_api_key: str = "",
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        openrouter_max_concurrent: int = 10,
    ) -> None:
        self._openai = AsyncOpenAI(api_key=api_key)
        self._instructor = instructor.from_openai(self._openai)
        self._temperature = temperature
        self._default_extract_model = default_extract_model
        self._default_chat_model = default_chat_model

        if openrouter_api_key:
            self._openrouter = AsyncOpenAI(
                api_key=openrouter_api_key,
                base_url=openrouter_base_url,
            )
            # TOOLS mode for OpenRouter: same mechanism OpenAI uses natively.
            # We tried OPENROUTER_STRUCTURED_OUTPUTS first but Alibaba's
            # Qwen endpoint rejects every request with:
            #   "'messages' must contain the word 'json' in some form, to
            #    use 'response_format' of type 'json_object'"
            # Tool-calling sidesteps that validation entirely. Qwen 3.6 Plus
            # reports ~4.3% tool-call error rate on OpenRouter (vs Kimi's
            # ~80% finish_reason='stop' problem), so a few retries are
            # expected but not the per-call failures we saw with K2.6.
            self._openrouter_instructor = instructor.from_openai(
                self._openrouter,
                mode=instructor.Mode.TOOLS,
            )
            # Caps in-flight OpenRouter requests. Callers may use
            # asyncio.gather() to launch many tasks; this semaphore
            # serialises them at the LLM-call boundary. Direct OpenAI
            # calls are NOT throttled by this.
            self._openrouter_semaphore: asyncio.Semaphore | None = (
                asyncio.Semaphore(openrouter_max_concurrent)
            )
        else:
            self._openrouter = None
            self._openrouter_instructor = None
            self._openrouter_semaphore = None

    def _pick_clients(self, model: str) -> tuple[AsyncOpenAI, instructor.AsyncInstructor]:
        """Return (raw_client, instructor_client) for a given model name.

        Routing rule: a "/" in the model name means OpenRouter (its slugs
        are always of the form ``vendor/model``). Anything else stays on
        OpenAI. If an OpenRouter model is requested but no OpenRouter key
        was configured, raise a clear error instead of silently calling
        OpenAI with an unrecognised model name.
        """
        if "/" in model:
            if self._openrouter is None:
                raise RuntimeError(
                    f"Model '{model}' looks like an OpenRouter slug but "
                    f"OPENROUTER_API_KEY is not set."
                )
            return self._openrouter, self._openrouter_instructor
        return self._openai, self._instructor

    async def extract_structured(
        self,
        response_model: type[T],
        prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        max_retries: int = 3,
        purpose: str = "unknown",
    ) -> T:
        """Call the LLM and validate the response against a Pydantic model.

        instructor handles JSON extraction, validation, and retries automatically.
        ``purpose`` is recorded in the usage log to group calls by intent
        (e.g. "indexing.attribute_extraction").
        """
        model = model or self._default_extract_model
        logger.debug(
            "LLM structured call: model=%s, response_model=%s, purpose=%s",
            model, response_model.__name__, purpose,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Hard cap on output length (12k tokens). Without this, small models
        # can fall into degenerate generation loops (e.g. gpt-5.4-mini once
        # produced 36'694 tokens of repeating fake change entries while
        # extracting from a Node.js TLS diff, stalling the whole pipeline).
        # 12k fits 99% of legitimate responses while bailing out of runaway
        # loops in ~2 min instead of 6-12 min.
        #
        # Parameter-name compat:
        #   - OpenAI's newer GPT-5.x models REQUIRE ``max_completion_tokens``
        #     and reject ``max_tokens`` with 400 "Unsupported parameter".
        #   - OpenRouter providers (Mistral, Qwen, Llama, ...) typically only
        #     declare support for the legacy ``max_tokens`` and — combined
        #     with our ``provider.require_parameters=True`` — return 404
        #     "No endpoints found" when sent ``max_completion_tokens``.
        # Route on the same "/" heuristic as the client picker.
        cap_param = "max_tokens" if "/" in model else "max_completion_tokens"
        kwargs: dict = {
            "model": model,
            "response_model": response_model,
            "messages": messages,
            "max_retries": max_retries,
            cap_param: 12000,
        }
        if model not in _NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = self._temperature

        # For OpenRouter calls, restrict routing to providers that support
        # every parameter we send (tools, tool_choice, response_format). Some
        # providers (e.g. Alibaba for qwen3.6-plus) only support a subset and
        # would return 404. require_parameters makes OpenRouter skip them.
        if "/" in model:
            kwargs["extra_body"] = {"provider": {"require_parameters": True}}

        _, instr = self._pick_clients(model)
        sem = self._openrouter_semaphore if "/" in model and self._openrouter_semaphore else nullcontext()
        async with sem:
            t0 = time.perf_counter()
            result, completion = await instr.chat.completions.create_with_completion(**kwargs)
            duration = time.perf_counter() - t0
        usage = getattr(completion, "usage", None)
        if usage is not None:
            get_tracker().record(
                model=model, purpose=purpose,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                duration_seconds=duration,
            )
        return result

    async def chat(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        purpose: str = "unknown",
    ) -> str:
        """Send a plain chat message and return the text response.

        ``purpose`` is recorded in the usage log to group calls by intent
        (e.g. "evaluation.answer_generation").
        """
        model = model or self._default_chat_model
        logger.debug("LLM chat call: model=%s, purpose=%s", model, purpose)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(model=model, messages=messages)
        if model not in _NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = self._temperature

        raw, _ = self._pick_clients(model)
        sem = self._openrouter_semaphore if "/" in model and self._openrouter_semaphore else nullcontext()
        async with sem:
            t0 = time.perf_counter()
            response = await raw.chat.completions.create(**kwargs)
            duration = time.perf_counter() - t0
        usage = getattr(response, "usage", None)
        if usage is not None:
            get_tracker().record(
                model=model, purpose=purpose,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                duration_seconds=duration,
            )
        return response.choices[0].message.content or ""
