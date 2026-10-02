"""Base tool abstraction with schema validation, timeouts, retries, and audit logging."""

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Any, Callable, Dict, Optional, Type
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ToolExecutionResult(BaseModel):
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: str = ""


class BaseTool:
    """Base class for all controlled agent tools."""

    name: str = "base_tool"
    description: str = "Base tool description"
    input_schema: Type[BaseModel] = BaseModel
    timeout: float = 15.0
    max_retries: int = 2
    retry_delay: float = 1.0

    def __init__(self, timeout: Optional[float] = None, max_retries: Optional[int] = None):
        if timeout is not None:
            self.timeout = timeout
        if max_retries is not None:
            self.max_retries = max_retries

    def validate_input(self, **kwargs) -> BaseModel:
        """Validate input arguments against the declared schema."""
        try:
            return self.input_schema(**kwargs)
        except ValidationError as e:
            logger.error(f"Validation error in tool '{self.name}': {e}")
            raise ValueError(f"Invalid parameters for tool '{self.name}': {e}") from e

    async def run(self, **kwargs) -> ToolExecutionResult:
        """Execute the tool with validation, retries, timeout, and execution logging."""
        start_time = time.perf_counter()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Input schema validation
        try:
            validated_args = self.validate_input(**kwargs)
        except ValueError as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                execution_time_ms=elapsed,
                timestamp=now_iso,
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                # Execute with strict timeout
                result = await asyncio.wait_for(
                    self._execute(validated_args),
                    timeout=self.timeout
                )
                elapsed = (time.perf_counter() - start_time) * 1000
                return ToolExecutionResult(
                    tool_name=self.name,
                    success=True,
                    data=result,
                    execution_time_ms=elapsed,
                    timestamp=now_iso,
                )
            except asyncio.TimeoutError as e:
                last_error = TimeoutError(f"Tool '{self.name}' timed out after {self.timeout}s (attempt {attempt})")
                logger.warning(str(last_error))
            except Exception as e:
                last_error = e
                logger.warning(f"Tool '{self.name}' failed on attempt {attempt}: {e}")

            if attempt <= self.max_retries:
                await asyncio.sleep(self.retry_delay * attempt)

        elapsed = (time.perf_counter() - start_time) * 1000
        return ToolExecutionResult(
            tool_name=self.name,
            success=False,
            error=str(last_error) if last_error else "Unknown execution failure",
            execution_time_ms=elapsed,
            timestamp=now_iso,
        )

    async def _execute(self, params: BaseModel) -> Any:
        """Core execution logic to be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement _execute")
