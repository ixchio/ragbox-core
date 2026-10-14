"""
Token tracking, cost estimation, and circuit breaking for API usage.
"""

import time
import os
import re
import threading
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class CostEstimate:
    """Structure holding usage estimation details."""

    def __init__(self, input_tokens: int, output_tokens: int, cost_usd: float):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_cost_usd = cost_usd

    def __str__(self) -> str:
        return (
            f"Approx ${self.total_cost_usd:.4f} "
            f"({self.input_tokens} in / {self.output_tokens} out)"
        )


class CostEstimator:
    """Predicts processing costs before running."""

    # Fallback costs per 1M tokens. Provider pricing changes independently of
    # this package, so every value can be overridden through environment
    # variables (see _pricing_for_model). Unknown remote models fail closed to
    # a conservative default instead of silently being treated as free.
    PRICING = {
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
        "local": {"input": 0.0, "output": 0.0},
    }
    DEFAULT_REMOTE_PRICING = {"input": 5.00, "output": 15.00}

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.encoding: Any = None
        try:
            import tiktoken

            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except Exception:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding:
            return len(self.encoding.encode(text))
        return len(text.split()) * 4 // 3  # naive fallback

    def estimate_embeddings(
        self, texts: List[str], model: str = "text-embedding-3-large"
    ) -> float:
        total_tokens = sum(self.count_tokens(t) for t in texts)
        pricing = self._pricing_for_model(model)
        return (total_tokens / 1_000_000) * pricing["input"]

    @staticmethod
    def _model_env_prefix(model_name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").upper()
        return f"RAGBOX_PRICE_{normalized}"

    def _pricing_for_model(self, model_name: Optional[str] = None) -> Dict[str, float]:
        model = model_name or self.model_name
        prefix = self._model_env_prefix(model)
        model_input = os.getenv(f"{prefix}_INPUT_PER_MILLION")
        model_output = os.getenv(f"{prefix}_OUTPUT_PER_MILLION")
        global_input = os.getenv("RAGBOX_INPUT_COST_PER_MILLION")
        global_output = os.getenv("RAGBOX_OUTPUT_COST_PER_MILLION")

        matched = self.PRICING.get(model)
        if matched is None:
            # Dated provider model names usually retain a known model prefix.
            for known in sorted(self.PRICING, key=len, reverse=True):
                if known != "local" and model.startswith(known):
                    matched = self.PRICING[known]
                    break
        if matched is None:
            matched = (
                self.PRICING["local"]
                if model == "local"
                else self.DEFAULT_REMOTE_PRICING
            )

        return {
            "input": float(model_input or global_input or matched["input"]),
            "output": float(model_output or global_output or matched["output"]),
        }

    def cost_for_tokens(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self._pricing_for_model()
        return (
            max(0, input_tokens) / 1_000_000 * pricing["input"]
            + max(0, output_tokens) / 1_000_000 * pricing["output"]
        )

    def estimate_generation(
        self, prompt: str, approx_output_tokens: int = 500
    ) -> CostEstimate:
        input_tokens = self.count_tokens(prompt)
        cost = self.cost_for_tokens(input_tokens, approx_output_tokens)
        return CostEstimate(input_tokens, approx_output_tokens, cost)


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CostBudget:
    max_daily_cost: float = 10.0  # USD
    max_query_cost: float = 0.50  # USD per query
    max_concurrent_queries: int = 10
    warning_threshold: float = 0.8  # 80% of budget
    failure_threshold: int = 5  # Open after 5 failures
    recovery_timeout: int = 60  # Try again after 60s
    half_open_max_calls: int = 3  # Test with 3 calls


@dataclass
class CostReservation:
    """A worst-case budget reservation held while one provider call is active."""

    operation_name: str
    estimated_cost: float
    active: bool = True


class CostCircuitBreaker:
    """
    Prevents cost overruns and cascading failures.
    Essential for production RAG with expensive LLM calls.
    """

    def __init__(self, budget: CostBudget):
        self.budget = budget
        self.daily_cost = 0.0
        self.reserved_cost = 0.0
        self.daily_cost_reset = time.time()
        self.concurrent_queries = 0
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.success_count = 0
        self.query_costs: Dict[str, float] = {}
        self._lock = threading.RLock()

    def _reset_daily_budget(self) -> None:
        if time.time() - self.daily_cost_reset > 86400:
            self.daily_cost = 0.0
            self.daily_cost_reset = time.time()
            logger.info("Daily cost budget reset")

    def reserve(
        self, estimated_cost: float, operation_name: str = "unknown"
    ) -> CostReservation:
        """Reserve worst-case cost before starting a provider operation."""
        if estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")

        with self._lock:
            self._reset_daily_budget()

            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.budget.recovery_timeout:
                    logger.info("Circuit entering half-open state")
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit open for {operation_name}. "
                        f"Retry after {self.budget.recovery_timeout}s."
                    )

            projected = self.daily_cost + self.reserved_cost + estimated_cost
            if projected > self.budget.max_daily_cost:
                raise BudgetExceeded(
                    f"Daily budget ${self.budget.max_daily_cost:.4f} would be exceeded. "
                    f"Charged: ${self.daily_cost:.4f}; reserved: "
                    f"${self.reserved_cost:.4f}; request: ${estimated_cost:.4f}"
                )

            if estimated_cost > self.budget.max_query_cost:
                raise QueryTooExpensive(
                    f"Query cost ${estimated_cost:.4f} exceeds max "
                    f"${self.budget.max_query_cost:.4f}"
                )

            if self.concurrent_queries >= self.budget.max_concurrent_queries:
                raise TooManyConcurrentQueries(
                    f"Max {self.budget.max_concurrent_queries} concurrent queries"
                )

            self.concurrent_queries += 1
            self.reserved_cost += estimated_cost
            return CostReservation(operation_name, estimated_cost)

    def _release(self, reservation: CostReservation) -> None:
        if not reservation.active:
            return
        reservation.active = False
        self.concurrent_queries = max(0, self.concurrent_queries - 1)
        self.reserved_cost = max(0.0, self.reserved_cost - reservation.estimated_cost)

    def complete(self, reservation: CostReservation, actual_cost: float) -> None:
        """Reconcile a reservation against measured provider usage."""
        if actual_cost < 0:
            raise ValueError("actual_cost must be non-negative")

        with self._lock:
            if not reservation.active:
                return
            self._release(reservation)
            self.daily_cost += actual_cost
            self.query_costs[reservation.operation_name] = (
                self.query_costs.get(reservation.operation_name, 0.0) + actual_cost
            )

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.budget.half_open_max_calls:
                    logger.info("Circuit closed — recovery successful")
                    self.state = CircuitState.CLOSED
            else:
                self.failure_count = max(0, self.failure_count - 1)

            if actual_cost > reservation.estimated_cost + 1e-12:
                logger.warning(
                    "Provider usage cost exceeded the reserved estimate: "
                    f"${actual_cost:.6f} > ${reservation.estimated_cost:.6f}"
                )

            usage_pct = self.daily_cost / max(self.budget.max_daily_cost, 1e-12)
            if usage_pct > self.budget.warning_threshold:
                logger.warning(f"Daily budget at {usage_pct * 100:.1f}%")

    def fail(
        self,
        reservation: CostReservation,
        *,
        count_failure: bool = True,
        actual_cost: float = 0.0,
    ) -> None:
        """Release a reservation, charge partial usage, and record a failure."""
        if actual_cost < 0:
            raise ValueError("actual_cost must be non-negative")
        with self._lock:
            if not reservation.active:
                return
            self._release(reservation)
            if actual_cost:
                self.daily_cost += actual_cost
                self.query_costs[reservation.operation_name] = (
                    self.query_costs.get(reservation.operation_name, 0.0) + actual_cost
                )
            if not count_failure:
                return
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.budget.failure_threshold:
                logger.error(f"Circuit opened due to {self.failure_count} failures")
                self.state = CircuitState.OPEN

    async def execute(
        self,
        operation: Callable,
        estimated_cost: float,
        operation_name: str = "unknown",
        actual_cost_getter: Optional[Callable[[Any], float]] = None,
    ) -> Any:
        """Execute with a reservation, then charge measured or estimated cost."""
        reservation = self.reserve(estimated_cost, operation_name)
        try:
            result = await operation()
            actual_cost = (
                actual_cost_getter(result) if actual_cost_getter else estimated_cost
            )
            self.complete(reservation, actual_cost)
            return result
        except BaseException as exc:
            self.fail(
                reservation,
                count_failure=not isinstance(
                    exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)
                ),
            )
            raise

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "daily_cost": self.daily_cost,
                "reserved_cost": self.reserved_cost,
                "daily_budget": self.budget.max_daily_cost,
                "remaining_budget": max(
                    0.0,
                    self.budget.max_daily_cost - self.daily_cost - self.reserved_cost,
                ),
                "circuit_state": self.state.value,
                "concurrent_queries": self.concurrent_queries,
                "query_breakdown": dict(self.query_costs),
            }


class CircuitBreakerOpen(Exception):
    pass


class BudgetExceeded(Exception):
    pass


class QueryTooExpensive(Exception):
    pass


class TooManyConcurrentQueries(Exception):
    pass
