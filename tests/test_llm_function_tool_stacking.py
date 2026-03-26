"""Tests for stacking `@tool` on top of `@llm_function`."""

from __future__ import annotations

import inspect

from SimpleLLMFunc import llm_function, tool
from SimpleLLMFunc.llm_decorator.utils.tools import process_tools
from SimpleLLMFunc.tool import Tool


def _build_specialist(mock_llm_interface):
    @tool(
        name="specialist_agent",
        description="Delegate focused specialist work",
    )
    @llm_function(llm_interface=mock_llm_interface)
    async def specialist_agent(task: str, constraints: str = "") -> str:
        """Focused specialist agent.

        Args:
            task: The task to solve.
            constraints: Extra constraints from the caller.
        """
        return ""

    return specialist_agent


def test_stacked_llm_function_keeps_tool_metadata(mock_llm_interface) -> None:
    specialist_agent = _build_specialist(mock_llm_interface)
    tool_obj = getattr(specialist_agent, "_tool")

    assert inspect.iscoroutinefunction(specialist_agent)
    assert hasattr(specialist_agent, "_tool")
    assert isinstance(tool_obj, Tool)
    assert tool_obj.name == "specialist_agent"
    assert [parameter.name for parameter in tool_obj.parameters] == [
        "task",
        "constraints",
    ]


def test_process_tools_accepts_stacked_llm_function(mock_llm_interface) -> None:
    specialist_agent = _build_specialist(mock_llm_interface)
    tool_obj = getattr(specialist_agent, "_tool")

    tool_param_for_api, tool_map = process_tools([specialist_agent], "supervisor")
    mapped_run = tool_map["specialist_agent"]

    assert tool_param_for_api is not None
    assert len(tool_param_for_api) == 1
    assert tool_param_for_api[0]["function"]["name"] == "specialist_agent"
    assert "task" in tool_param_for_api[0]["function"]["parameters"]["properties"]
    assert getattr(mapped_run, "__self__", None) is tool_obj
    assert getattr(mapped_run, "__func__", None) is Tool.run
