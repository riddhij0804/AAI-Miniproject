"""Controlled tool registry managing allowed agent tools with schema validation and auditing."""

from datetime import datetime, timezone
import logging
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.tools.base import BaseTool, ToolExecutionResult
from backend.app.tools.web_search import WebSearchTool
from backend.app.tools.webpage_reader import WebpageReaderTool
from backend.app.tools.pdf_reader import PDFReaderTool

logger = logging.getLogger(__name__)


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    timeout_seconds: float
    max_retries: int
    allowed_usage: str


class ToolAuditLog(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    execution_time_ms: float
    timestamp: str
    error: Optional[str] = None


class ToolRegistry:
    """Registry maintaining approved tools and executing them with audit logs."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._audit_logs: List[ToolAuditLog] = []

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool instance in the registry."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: '{tool.name}'")

    def unregister_tool(self, tool_name: str) -> None:
        """Remove a tool from the registry."""
        if tool_name in self._tools:
            del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools with their schemas and configuration."""
        definitions = []
        for name, tool in self._tools.items():
            definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_schema=tool.input_schema.model_json_schema(),
                    timeout_seconds=tool.timeout,
                    max_retries=tool.max_retries,
                    allowed_usage=f"Executable only through controlled registry ({name})",
                )
            )
        return definitions

    async def execute_tool(self, tool_name: str, **kwargs) -> ToolExecutionResult:
        """Execute a registered tool by name with parameter validation and audit logging."""
        tool = self._tools.get(tool_name)
        if not tool:
            err_msg = f"Tool '{tool_name}' is not registered. Allowed tools: {list(self._tools.keys())}"
            logger.error(err_msg)
            result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=err_msg,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._log_audit(tool_name, kwargs, result)
            return result

        result = await tool.run(**kwargs)
        self._log_audit(tool_name, kwargs, result)
        return result

    def _log_audit(self, tool_name: str, args: Dict[str, Any], result: ToolExecutionResult) -> None:
        # Avoid storing sensitive secrets or massive payload bodies in audit log
        sanitized_args = {k: ("***" if "key" in k.lower() else str(v)[:200]) for k, v in args.items()}
        self._audit_logs.append(
            ToolAuditLog(
                tool_name=tool_name,
                arguments=sanitized_args,
                success=result.success,
                execution_time_ms=result.execution_time_ms,
                timestamp=result.timestamp,
                error=result.error,
            )
        )

    def get_audit_logs(self) -> List[ToolAuditLog]:
        return list(self._audit_logs)


def create_default_tool_registry() -> ToolRegistry:
    """Instantiate and populate the standard tool registry with approved tools."""
    registry = ToolRegistry()
    registry.register_tool(WebSearchTool())
    registry.register_tool(WebpageReaderTool())
    registry.register_tool(PDFReaderTool())
    return registry


# Global default registry
default_registry = create_default_tool_registry()
