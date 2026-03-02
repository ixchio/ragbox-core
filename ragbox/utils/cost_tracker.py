"""
Token tracking and cost estimation for API usage.
"""
from typing import List
from loguru import logger

class CostEstimate:
    """Structure holding usage estimation details."""
    def __init__(self, input_tokens: int, output_tokens: int, cost_usd: float):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_cost_usd = cost_usd

    def __str__(self) -> str:
        return f"Approx ${(self.total_cost_usd):.4f} ({self.input_tokens} in / {self.output_tokens} out)"

class CostEstimator:
    """Predicts processing costs before running."""
    
    # Costs per 1M tokens (as of early 2024 pricing models)
    PRICING = {
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "local": {"input": 0.0, "output": 0.0}
    }

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        try:
            import tiktoken
            # Fallback encoding if model not found
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                import tiktoken
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self.encoding = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding:
            return len(self.encoding.encode(text))
        return len(text.split()) * 4 // 3 # naive fallback

    def estimate_embeddings(self, texts: List[str], model: str = "text-embedding-3-large") -> float:
        total_tokens = sum(self.count_tokens(t) for t in texts)
        pricing = self.PRICING.get(model, self.PRICING["local"])
        return (total_tokens / 1_000_000) * pricing["input"]

    def estimate_generation(self, prompt: str, approx_output_tokens: int = 500) -> CostEstimate:
        input_tokens = self.count_tokens(prompt)
        pricing = self.PRICING.get(self.model_name, self.PRICING["local"])
        
        in_cost = (input_tokens / 1_000_000) * pricing["input"]
        out_cost = (approx_output_tokens / 1_000_000) * pricing["output"]
        
        return CostEstimate(input_tokens, approx_output_tokens, in_cost + out_cost)


from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any
import time
from enum import Enum
from loguru import logger

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

@dataclass
class CostBudget:
    max_daily_cost: float = 10.0  # USD
    max_query_cost: float = 0.50  # USD per query
    max_concurrent_queries: int = 10
    warning_threshold: float = 0.8  # 80% of budget
    
    # Circuit breaker settings
    failure_threshold: int = 5      # Open after 5 failures
    recovery_timeout: int = 60      # Try again after 60s
    half_open_max_calls: int = 3    # Test with 3 calls

class CostCircuitBreaker:
    """
    Prevents cost overruns and cascading failures.
    Essential for production RAG with expensive LLM calls.
    """
    
    def __init__(self, budget: CostBudget):
        self.budget = budget
        self.daily_cost = 0.0
        self.daily_cost_reset = time.time()
        self.concurrent_queries = 0
        
        # Circuit breaker state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
        
        # Tracking
        self.query_costs: Dict[str, float] = {}
        
    def _reset_daily_budget(self) -> None:
        """Reset daily budget every 24 hours"""
        if time.time() - self.daily_cost_reset > 86400:
            self.daily_cost = 0.0
            self.daily_cost_reset = time.time()
            logger.info("Daily cost budget reset")
    
    async def execute(
        self,
        operation: Callable,
        estimated_cost: float,
        operation_name: str = "unknown"
    ) -> Optional[Any]:
        """
        Execute operation with cost and circuit protection
        """
        self._reset_daily_budget()
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.budget.recovery_timeout:
                logger.info("Circuit entering half-open state")
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0
                self.success_count = 0
            else:
                raise CircuitBreakerOpen(
                    f"Circuit open for {operation_name}. "
                    f"Retry after {self.budget.recovery_timeout}s"
                )
        
        # Check daily budget
        if self.daily_cost >= self.budget.max_daily_cost:
            raise BudgetExceeded(
                f"Daily budget ${self.budget.max_daily_cost} exceeded. "
                f"Current: ${self.daily_cost:.2f}"
            )
        
        # Check per-query budget
        if estimated_cost > self.budget.max_query_cost:
            raise QueryTooExpensive(
                f"Query cost ${estimated_cost:.2f} exceeds max "
                f"${self.budget.max_query_cost:.2f}"
            )
        
        # Check concurrent limit
        if self.concurrent_queries >= self.budget.max_concurrent_queries:
            raise TooManyConcurrentQueries(
                f"Max {self.budget.max_concurrent_queries} concurrent queries"
            )
        
        # Execute with tracking
        self.concurrent_queries += 1
        start_time = time.time()
        
        try:
            result = await operation()
            
            # Success - update circuit
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.budget.half_open_max_calls:
                    logger.info("Circuit closed - recovery successful")
                    self.state = CircuitState.CLOSED
            else:
                self.failure_count = max(0, self.failure_count - 1)
            
            # Track cost
            actual_cost = estimated_cost  # In reality, calculate from tokens
            self.daily_cost += actual_cost
            self.query_costs[operation_name] = (
                self.query_costs.get(operation_name, 0) + actual_cost
            )
            
            # Warning at 80%
            if self.daily_cost / self.budget.max_daily_cost > self.budget.warning_threshold:
                logger.warning(
                    f"Daily budget at {self.daily_cost/self.budget.max_daily_cost*100:.1f}%"
                )
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.budget.failure_threshold:
                logger.error(f"Circuit opened due to {self.failure_count} failures")
                self.state = CircuitState.OPEN
            
            raise
            
        finally:
            self.concurrent_queries -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "daily_cost": self.daily_cost,
            "daily_budget": self.budget.max_daily_cost,
            "remaining_budget": self.budget.max_daily_cost - self.daily_cost,
            "circuit_state": self.state.value,
            "concurrent_queries": self.concurrent_queries,
            "query_breakdown": self.query_costs
        }

class CircuitBreakerOpen(Exception): pass
class BudgetExceeded(Exception): pass
class QueryTooExpensive(Exception): pass
class TooManyConcurrentQueries(Exception): pass
