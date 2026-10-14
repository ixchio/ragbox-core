"""
Unified LLM client interface for RAGBox.
Abstracts OpenAI, Anthropic, and local models.
"""

from __future__ import annotations

import os
import asyncio
import contextvars
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional
from loguru import logger

from ragbox.config.defaults import Settings
from ragbox.utils.cost_tracker import CostCircuitBreaker, CostBudget

if TYPE_CHECKING:
    from groq.types.chat import ChatCompletionMessageParam as GroqMessageParam
    from groq.types.chat.completion_create_params import (
        ResponseFormatResponseFormatJsonObject as GroqJSONResponseFormat,
    )
    from openai.types.chat import ChatCompletionMessageParam as OpenAIMessageParam
    from openai.types.shared_params import (
        ResponseFormatJSONObject as OpenAIJSONResponseFormat,
    )


@dataclass(frozen=True)
class ProviderTokenUsage:
    input_tokens: int
    output_tokens: int


_provider_usage: contextvars.ContextVar[
    Optional[ProviderTokenUsage]
] = contextvars.ContextVar("ragbox_provider_usage", default=None)


class LLMClient(ABC):
    """Abstract Base Class for LLM providers."""

    def __init__(self, budget: Optional[CostBudget] = None):
        self.circuit_breaker = CostCircuitBreaker(budget or CostBudget())

    def _model_name(self, kwargs: Dict[str, Any]) -> str:
        model = kwargs.get("model", getattr(self, "_model", "local"))
        return model if isinstance(model, str) else "local"

    @staticmethod
    def _prompt_for_accounting(prompt: str, system: Optional[str]) -> str:
        return f"{system}\n{prompt}" if system else prompt

    def _estimated_cost(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        max_tokens: int,
    ) -> float:
        from ragbox.utils.cost_tracker import CostEstimator

        estimator = CostEstimator(model)
        return estimator.estimate_generation(
            self._prompt_for_accounting(prompt, system),
            approx_output_tokens=max_tokens,
        ).total_cost_usd

    def _actual_cost(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        result_text: str,
    ) -> float:
        from ragbox.utils.cost_tracker import CostEstimator

        estimator = CostEstimator(model)
        usage = _provider_usage.get()
        if usage is not None:
            return estimator.cost_for_tokens(usage.input_tokens, usage.output_tokens)
        return estimator.cost_for_tokens(
            estimator.count_tokens(self._prompt_for_accounting(prompt, system)),
            estimator.count_tokens(result_text),
        )

    @staticmethod
    def _record_usage(input_tokens: Any, output_tokens: Any) -> None:
        try:
            _provider_usage.set(
                ProviderTokenUsage(int(input_tokens or 0), int(output_tokens or 0))
            )
        except (TypeError, ValueError):
            logger.debug("Provider returned unreadable token usage; using local count")

    def _record_response_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = getattr(
            usage,
            "prompt_tokens",
            getattr(usage, "input_tokens", None),
        )
        output_tokens = getattr(
            usage,
            "completion_tokens",
            getattr(usage, "output_tokens", None),
        )
        if input_tokens is not None or output_tokens is not None:
            self._record_usage(input_tokens, output_tokens)

    @abstractmethod
    async def _agenerate(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Asynchronously generate a response."""
        pass

    async def agenerate(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> str:
        # Merge default kwargs with provided ones
        merged_kwargs = {
            "system": system,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get(
                "max_tokens", 800
            ),  # Default max_tokens for general generation
        }
        merged_kwargs.update(kwargs)  # User provided kwargs override defaults
        model = self._model_name(merged_kwargs)
        max_tokens = int(merged_kwargs.get("max_tokens", 800))
        estimated_cost = self._estimated_cost(prompt, system, model, max_tokens)

        async def operation():
            return await self._agenerate(prompt, **merged_kwargs)

        usage_token = _provider_usage.set(None)
        try:
            return await self.circuit_breaker.execute(
                operation,
                estimated_cost=estimated_cost,
                operation_name=self.__class__.__name__,
                actual_cost_getter=lambda result: self._actual_cost(
                    prompt, system, model, result
                ),
            )
        finally:
            _provider_usage.reset(usage_token)

    async def agenerate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Merge default kwargs with provided ones
        merged_kwargs = {
            "system": system,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 800),
        }
        merged_kwargs.update(kwargs)
        model = self._model_name(merged_kwargs)
        max_tokens = int(merged_kwargs.get("max_tokens", 800))
        estimated_cost = self._estimated_cost(prompt, system, model, max_tokens)

        async def operation():
            return await self._agenerate_structured(prompt, schema, **merged_kwargs)

        usage_token = _provider_usage.set(None)
        try:
            return await self.circuit_breaker.execute(
                operation,
                estimated_cost=estimated_cost,
                operation_name=f"{self.__class__.__name__}_structured",
                actual_cost_getter=lambda result: self._actual_cost(
                    prompt, system, model, json.dumps(result, sort_keys=True)
                ),
            )
        finally:
            _provider_usage.reset(usage_token)

    @abstractmethod
    async def _agenerate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Asynchronously generate a response matching a JSON schema."""
        pass

    async def _astream(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Default streaming fallback: yields the full response as one chunk."""
        result = await self._agenerate(prompt, system=system, **kwargs)
        yield result

    async def astream(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream with the same reservation and accounting as non-streaming calls."""
        merged_kwargs = {
            "system": system,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 800),
        }
        merged_kwargs.update(kwargs)
        model = self._model_name(merged_kwargs)
        max_tokens = int(merged_kwargs.get("max_tokens", 800))
        estimated_cost = self._estimated_cost(prompt, system, model, max_tokens)
        reservation = self.circuit_breaker.reserve(
            estimated_cost, f"{self.__class__.__name__}_stream"
        )
        usage_token = _provider_usage.set(None)
        output_parts = []
        try:
            async for chunk in self._astream(prompt, **merged_kwargs):
                output_parts.append(chunk)
                yield chunk
            actual_cost = self._actual_cost(
                prompt, system, model, "".join(output_parts)
            )
            self.circuit_breaker.complete(reservation, actual_cost)
        except BaseException as exc:
            actual_cost = self._actual_cost(
                prompt, system, model, "".join(output_parts)
            )
            self.circuit_breaker.fail(
                reservation,
                count_failure=not isinstance(
                    exc, (GeneratorExit, asyncio.CancelledError)
                ),
                actual_cost=actual_cost,
            )
            raise
        finally:
            if reservation.active:
                self.circuit_breaker.fail(reservation, count_failure=False)
            _provider_usage.reset(usage_token)


class OpenAIClient(LLMClient):
    """OpenAI client wrapper. Also works with any OpenAI-compatible API (Ollama, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        budget: Optional[CostBudget] = None,
    ) -> None:
        super().__init__(budget)
        try:
            import os
            from openai import AsyncOpenAI

            base_url = os.environ.get(
                "OPENAI_BASE_URL"
            )  # e.g. http://localhost:11434/v1
            model = os.environ.get("OPENAI_MODEL", model)  # override via env
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,  # None = use default api.openai.com
            )
            self._model = model
            if base_url:
                logger.debug(
                    f"Initialized OpenAIClient (custom base) with {model} @ {base_url}"
                )
            else:
                logger.debug(f"Initialized OpenAIClient with {model}")
        except ImportError:
            logger.error("Failed to import openai.")
            raise

    async def _agenerate(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> str:
        messages: list[OpenAIMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 800),
        )
        self._record_response_usage(response)
        return response.choices[0].message.content or ""

    async def _agenerate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Force JSON generation with valid payload parsing."""
        sys_prompt = f"{system or 'You are a helpful assistant.'}\nAlways output strict, valid JSON matching this schema: {json.dumps(schema)}"
        messages: list[OpenAIMessageParam] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        response_format: OpenAIJSONResponseFormat = {"type": "json_object"}
        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            response_format=response_format,
            temperature=0.0,
            max_tokens=kwargs.get("max_tokens", 800),
        )
        self._record_response_usage(response)
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error("LLM did not output valid JSON")
            return {}


class AnthropicClient(LLMClient):
    """Anthropic client wrapper."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20240620",
        budget: Optional[CostBudget] = None,
    ) -> None:
        super().__init__(budget)
        try:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=api_key)
            self._model = model
            logger.debug(f"Initialized AnthropicClient with {model}")
        except ImportError:
            logger.error("Failed to import anthropic.")
            raise

    async def _agenerate(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> str:
        response = await self._client.messages.create(
            model=kwargs.get("model", self._model),
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.0),
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        self._record_response_usage(response)
        return "".join([b.text for b in response.content if hasattr(b, "text")])

    async def _astream(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=kwargs.get("model", self._model),
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.0),
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
            get_final_message = getattr(stream, "get_final_message", None)
            if get_final_message is not None:
                final_message = await get_final_message()
                self._record_response_usage(final_message)

    async def _agenerate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        sys_prompt = f"{system or ''}\nYou must output ONLY valid JSON matching this schema: {json.dumps(schema)}. Do not include any explanations, formatting ticks, or markdown."
        res = await self._agenerate(prompt, system=sys_prompt, **kwargs)
        try:
            # Handle possible markdown wrap in response
            cleaned = res.strip("`\n").replace("json\n", "", 1).strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Claude JSON response: {res}")
            return {}


class LlamaCppClient(LLMClient):
    """Local LLM via Llama-cpp-python."""

    def __init__(
        self,
        model_path: str = "models/llama-3.1-8b-instruct.gguf",
        budget: Optional[CostBudget] = None,
    ) -> None:
        super().__init__(budget)
        try:
            __import__("llama_cpp")

            self._model_path = model_path
            self._model = "local"
            self._llm = None
            logger.debug("Initialized LlamaCppClient (lazy load)")
        except ImportError:
            logger.error("Failed to import llama_cpp.")
            raise

    def _get_llm(self) -> Any:
        if self._llm is None:
            from llama_cpp import Llama

            if not os.path.exists(self._model_path):
                raise FileNotFoundError(f"Model file {self._model_path} not found.")
            self._llm = Llama(
                model_path=self._model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False
            )
        return self._llm

    async def _agenerate(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> str:
        def _sync_gen() -> str:
            llm = self._get_llm()
            formatted_prompt = f"System: {system or ''}\nUser: {prompt}\nAssistant:"
            res = llm.create_completion(
                formatted_prompt,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.0),
            )
            return res["choices"][0]["text"]

        return await asyncio.to_thread(_sync_gen)

    async def _agenerate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        res = await self._agenerate(
            f"{prompt}\nReturn ONLY JSON matching schema: {json.dumps(schema)}",
            system,
            **kwargs,
        )
        try:
            cleaned = res.strip("`\n").replace("json\n", "", 1).strip()
            return json.loads(cleaned)
        except Exception:
            return {}


class GroqClient(LLMClient):
    """Groq client wrapper."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        budget: Optional[CostBudget] = None,
    ) -> None:
        super().__init__(budget)
        try:
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=api_key)
            self._model = model
            logger.debug(f"Initialized GroqClient with {model}")
        except ImportError:
            logger.error("Failed to import groq.")
            raise

    async def _agenerate(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> str:
        messages: list[GroqMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 800),
        )
        self._record_response_usage(response)
        return response.choices[0].message.content or ""

    async def _astream(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        messages: list[GroqMessageParam] = []
        if system or kwargs.get("system"):
            messages.append(
                {"role": "system", "content": system or kwargs.get("system", "")}
            )
        messages.append({"role": "user", "content": prompt})

        stream = await self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 800),
            stream=True,
        )
        async for chunk in stream:
            self._record_response_usage(chunk)
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def _agenerate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Force JSON generation with valid payload parsing."""
        sys_prompt = f"{system or 'You are a helpful assistant.'}\nAlways output strict, valid JSON matching this schema: {json.dumps(schema)}"
        messages: list[GroqMessageParam] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        response_format: GroqJSONResponseFormat = {"type": "json_object"}
        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            response_format=response_format,
            temperature=0.0,
        )
        self._record_response_usage(response)
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error("LLM did not output valid JSON")
            return {}


class LLMAutoDetector:
    """Auto-detects the best LLM provider available."""

    @staticmethod
    def detect(config: Settings) -> LLMClient:
        budget = CostBudget(
            max_daily_cost=config.max_daily_cost_usd,
            max_query_cost=config.max_query_cost_usd,
            max_concurrent_queries=config.max_concurrent_queries,
        )
        openai_key = getattr(config, "openai_api_key", os.getenv("OPENAI_API_KEY"))
        if openai_key:
            logger.info("Using OpenAI for LLM.")
            return OpenAIClient(api_key=openai_key, budget=budget)

        anthropic_key = getattr(
            config, "anthropic_api_key", os.getenv("ANTHROPIC_API_KEY")
        )
        if anthropic_key:
            logger.info("Using Anthropic for LLM.")
            return AnthropicClient(api_key=anthropic_key, budget=budget)

        groq_key = getattr(config, "groq_api_key", os.getenv("GROQ_API_KEY"))
        if groq_key:
            logger.info("Using Groq for LLM.")
            return GroqClient(api_key=groq_key, budget=budget)

        local_path = getattr(
            config,
            "local_model_path",
            os.getenv("LOCAL_MODEL_PATH", "models/llama-3.1-8b-instruct.gguf"),
        )

        if not os.path.exists(local_path):
            logger.warning(
                "\n" + "=" * 80 + "\n"
                "⚠️  NO API KEYS DETECTED AND LOCAL LLM NOT FOUND! ⚠️\n"
                "RAGBox tried to fall back to a local model, but the weights are missing.\n"
                "To unlock the full power of RAGBox, set ONE of the following environment variables:\n"
                "  - export OPENAI_API_KEY='your-key'\n"
                "  - export ANTHROPIC_API_KEY='your-key'\n"
                "  - export GROQ_API_KEY='your-key'\n"
                f"Alternatively, download a GGUF model and place it at: {local_path}\n"
                + "=" * 80
            )
        else:
            logger.info("Falling back to local LLaMA.")

        return LlamaCppClient(model_path=local_path, budget=budget)


__all__ = [
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "GroqClient",
    "LlamaCppClient",
    "LLMAutoDetector",
]
