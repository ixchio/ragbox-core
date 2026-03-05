"""
Unified LLM client interface for RAGBox.
Abstracts OpenAI, Anthropic, and local models.
"""
import os
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional
from loguru import logger

from ragbox.config.defaults import Settings
from ragbox.utils.cost_tracker import CostCircuitBreaker, CostBudget


class LLMClient(ABC):
    """Abstract Base Class for LLM providers."""

    def __init__(self):
        # Initialize basic cost protection
        self.circuit_breaker = CostCircuitBreaker(CostBudget())

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

        # Protect with circuit breaker using an estimated dummy cost (e.g. $0.001) for now
        # You'd typically extract exact actual prompt tokens or estimate based on prompt len
        estimated_cost = 0.001

        async def operation():
            return await self._agenerate(prompt, **merged_kwargs)

        return await self.circuit_breaker.execute(
            operation,
            estimated_cost=estimated_cost,
            operation_name=self.__class__.__name__,
        )

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

        estimated_cost = 0.005  # higher for structured output estimation

        async def operation():
            return await self._agenerate_structured(prompt, schema, **merged_kwargs)

        return await self.circuit_breaker.execute(
            operation,
            estimated_cost=estimated_cost,
            operation_name=f"{self.__class__.__name__}_structured",
        )

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
        """Public streaming interface with circuit breaker protection."""
        merged_kwargs = {
            "system": system,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 800),
        }
        merged_kwargs.update(kwargs)
        async for chunk in self._astream(prompt, **merged_kwargs):
            yield chunk


class OpenAIClient(LLMClient):
    """OpenAI client wrapper. Also works with any OpenAI-compatible API (Ollama, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        super().__init__()
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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 800),
        )
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
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=kwargs.get("max_tokens", 800),
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error("LLM did not output valid JSON")
            return {}


class AnthropicClient(LLMClient):
    """Anthropic client wrapper."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620") -> None:
        super().__init__()
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

    def __init__(self, model_path: str = "models/llama-3.1-8b-instruct.gguf") -> None:
        super().__init__()
        try:
            from llama_cpp import Llama

            self._model_path = model_path
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

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        super().__init__()
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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 800),
        )
        return response.choices[0].message.content or ""

    async def _astream(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        messages = []
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
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await self._client.chat.completions.create(
            model=kwargs.get("model", self._model),
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
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
        openai_key = getattr(config, "openai_api_key", os.getenv("OPENAI_API_KEY"))
        if openai_key:
            logger.info("Using OpenAI for LLM.")
            return OpenAIClient(api_key=openai_key)

        anthropic_key = getattr(
            config, "anthropic_api_key", os.getenv("ANTHROPIC_API_KEY")
        )
        if anthropic_key:
            logger.info("Using Anthropic for LLM.")
            return AnthropicClient(api_key=anthropic_key)

        groq_key = getattr(config, "groq_api_key", os.getenv("GROQ_API_KEY"))
        if groq_key:
            logger.info("Using Groq for LLM.")
            return GroqClient(api_key=groq_key)

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

        return LlamaCppClient(model_path=local_path)


__all__ = [
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "GroqClient",
    "LlamaCppClient",
    "LLMAutoDetector",
]
