"""Unit tests for controlled Tool Registry, validation, and audit logging."""

import asyncio
import pytest
from pydantic import BaseModel, Field

from backend.app.tools.base import BaseTool
from backend.app.tools.registry import ToolRegistry


class DummyInput(BaseModel):
    query: str = Field(..., min_length=3)
    count: int = Field(default=5, ge=1)


class DummyTool(BaseTool):
    name = "dummy_test_tool"
    description = "A dummy tool for unit testing registry behavior"
    input_schema = DummyInput
    timeout = 1.0
    max_retries = 1

    async def _execute(self, params: DummyInput):
        if params.query == "trigger_timeout":
            await asyncio.sleep(2.0)
        if params.query == "trigger_error":
            raise ValueError("Deliberate failure in dummy tool")
        return {"echo": params.query, "count": params.count}


@pytest.mark.asyncio
async def test_tool_registry_validation_and_execution():
    registry = ToolRegistry()
    registry.register_tool(DummyTool())

    # 1. Valid execution
    res = await registry.execute_tool("dummy_test_tool", query="valid query", count=3)
    assert res.success is True
    assert res.data["echo"] == "valid query"
    assert res.data["count"] == 3
    assert res.execution_time_ms > 0

    # 2. Schema validation failure (query too short)
    bad_res = await registry.execute_tool("dummy_test_tool", query="ab", count=3)
    assert bad_res.success is False
    assert "Invalid parameters" in bad_res.error

    # 3. Timeout enforcement
    timeout_res = await registry.execute_tool("dummy_test_tool", query="trigger_timeout")
    assert timeout_res.success is False
    assert "timed out" in timeout_res.error.lower()

    # 4. Audit logging
    logs = registry.get_audit_logs()
    assert len(logs) == 3
    assert logs[0].tool_name == "dummy_test_tool"
    assert logs[0].success is True
    assert logs[1].success is False
