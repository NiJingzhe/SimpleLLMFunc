"""Tests for stdio streaming decorator."""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from types import SimpleNamespace

from SimpleLLMFunc.hooks.events import (
    CustomEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    LLMChunkArriveEvent,
    ReActEventType,
    ReactEndEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from SimpleLLMFunc.hooks.stream import EventOrigin, EventYield
from SimpleLLMFunc.utils.stdio.decorator import stdio


def _ts() -> datetime:
    return datetime.now()


def _chunk_with_content(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )


def _response_with_content(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="fake-model",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


def test_stdio_streams_user_assistant_tool_and_result() -> None:
    async def fake_agent(message: str, history=None):
        yield EventYield(
            event=LLMCallStartEvent(
                event_type=ReActEventType.LLM_CALL_START,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                messages=list(history or []),
                tools=[],
                llm_kwargs={},
                stream=True,
            )
        )
        yield EventYield(
            event=ToolCallStartEvent(
                event_type=ReActEventType.TOOL_CALL_START,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                tool_name="execute_code",
                tool_call_id="call_1",
                arguments={"code": "print(1)"},
                tool_call={},
            )
        )
        yield EventYield(
            event=CustomEvent(
                event_type=ReActEventType.CUSTOM_EVENT,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                event_name="kernel_stdout",
                data={"text": "1\n"},
                tool_name="execute_code",
                tool_call_id="call_1",
            )
        )
        yield EventYield(
            event=ToolCallEndEvent(
                event_type=ReActEventType.TOOL_CALL_END,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                tool_name="execute_code",
                tool_call_id="call_1",
                arguments={"code": "print(1)"},
                result={"success": True, "stdout": "1\n"},
                execution_time=0.12,
                success=True,
            )
        )
        yield EventYield(
            event=LLMChunkArriveEvent(
                event_type=ReActEventType.LLM_CHUNK_ARRIVE,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                chunk=_chunk_with_content(f"done:{message}"),
                accumulated_content=f"done:{message}",
                chunk_index=0,
            )
        )
        yield EventYield(
            event=LLMCallEndEvent(
                event_type=ReActEventType.LLM_CALL_END,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                response=_response_with_content(f"done:{message}"),
                messages=[],
                tool_calls=[],
                execution_time=0.34,
                usage=None,
            )
        )
        yield EventYield(
            event=ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                final_response=f"done:{message}",
                final_messages=[
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": f"done:{message}"},
                ],
                total_iterations=1,
                total_execution_time=0.5,
                total_tool_calls=1,
                total_llm_calls=1,
                total_token_usage=None,
            )
        )

    stdin = StringIO("hello\n")
    stdout = StringIO()
    stderr = StringIO()

    stdio(input_stream=stdin, output_stream=stdout, error_stream=stderr)(fake_agent)()

    rendered = stdout.getvalue()
    assert '<user_message>"hello"</user_message>' in rendered
    assert '<tool_call scope="main" tool="execute_code" call_id="call_1">' in rendered
    assert '<tool_output call_id="call_1">' in rendered
    assert '<tool_result call_id="call_1" status="success"' in rendered
    assert '<assistant_message call_id="llm_call_' in rendered
    assert "done:hello" in rendered
    assert stderr.getvalue() == ""


def test_stdio_routes_tool_input_through_stdin() -> None:
    async def fake_agent(message: str, history=None):
        yield EventYield(
            event=ToolCallStartEvent(
                event_type=ReActEventType.TOOL_CALL_START,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                tool_name="ask_user",
                tool_call_id="call_input",
                arguments={"prompt": "Name: "},
                tool_call={},
            )
        )
        yield EventYield(
            event=CustomEvent(
                event_type=ReActEventType.CUSTOM_EVENT,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                event_name="kernel_input_request",
                data={"request_id": "req_1", "prompt": "Name: "},
                tool_name="ask_user",
                tool_call_id="call_input",
            )
        )
        yield EventYield(
            event=ToolCallEndEvent(
                event_type=ReActEventType.TOOL_CALL_END,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                tool_name="ask_user",
                tool_call_id="call_input",
                arguments={"prompt": "Name: "},
                result={"accepted": True},
                execution_time=0.05,
                success=True,
            )
        )
        yield EventYield(
            event=LLMCallStartEvent(
                event_type=ReActEventType.LLM_CALL_START,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                messages=[],
                tools=[],
                llm_kwargs={},
                stream=True,
            )
        )
        yield EventYield(
            event=LLMChunkArriveEvent(
                event_type=ReActEventType.LLM_CHUNK_ARRIVE,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                chunk=_chunk_with_content(f"echo:{message}"),
                accumulated_content=f"echo:{message}",
                chunk_index=0,
            )
        )
        yield EventYield(
            event=LLMCallEndEvent(
                event_type=ReActEventType.LLM_CALL_END,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                response=_response_with_content(f"echo:{message}"),
                messages=[],
                tool_calls=[],
                execution_time=0.1,
                usage=None,
            )
        )
        yield EventYield(
            event=ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                final_response=f"echo:{message}",
                final_messages=[
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": f"echo:{message}"},
                ],
                total_iterations=1,
                total_execution_time=0.2,
                total_tool_calls=1,
                total_llm_calls=1,
                total_token_usage=None,
            )
        )

    stdin = StringIO("question\nAlice\n")
    stdout = StringIO()
    stderr = StringIO()

    stdio(input_stream=stdin, output_stream=stdout, error_stream=stderr)(fake_agent)()

    rendered = stdout.getvalue()
    assert '<tool_input_request call_id="call_input" request_id="req_1">' in rendered
    assert (
        '<tool_input_response call_id="call_input" request_id="req_1">"Alice"</tool_input_response>'
        in rendered
    )
    assert '<assistant_message call_id="llm_call_' in rendered
    assert "echo:question" in rendered
    assert stderr.getvalue() == ""


def test_stdio_streams_multiple_fork_scopes() -> None:
    async def fake_agent(message: str, history=None):
        _ = history
        fork1_origin = EventOrigin(
            session_id="session",
            agent_call_id="agent_main",
            event_seq=1,
            fork_id="fork_1",
            fork_depth=1,
            memory_key="fork_1_mem",
        )
        fork2_origin = EventOrigin(
            session_id="session",
            agent_call_id="agent_main",
            event_seq=2,
            fork_id="fork_2",
            fork_depth=1,
            memory_key="fork_2_mem",
        )
        yield EventYield(
            event=LLMCallStartEvent(
                event_type=ReActEventType.LLM_CALL_START,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                messages=[],
                tools=[],
                llm_kwargs={},
                stream=True,
            ),
            origin=fork1_origin,
        )
        yield EventYield(
            event=LLMChunkArriveEvent(
                event_type=ReActEventType.LLM_CHUNK_ARRIVE,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                chunk=_chunk_with_content(f"fork-1:{message}"),
                accumulated_content=f"fork-1:{message}",
                chunk_index=0,
            ),
            origin=fork1_origin,
        )
        yield EventYield(
            event=LLMCallEndEvent(
                event_type=ReActEventType.LLM_CALL_END,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                response=_response_with_content(f"fork-1:{message}"),
                messages=[],
                tool_calls=[],
                execution_time=0.1,
                usage=None,
            ),
            origin=fork1_origin,
        )
        yield EventYield(
            event=LLMCallStartEvent(
                event_type=ReActEventType.LLM_CALL_START,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                messages=[],
                tools=[],
                llm_kwargs={},
                stream=True,
            ),
            origin=fork2_origin,
        )
        yield EventYield(
            event=ToolCallStartEvent(
                event_type=ReActEventType.TOOL_CALL_START,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                tool_name="execute_code",
                tool_call_id="call_fork2",
                arguments={"code": "print('fork2')"},
                tool_call={},
            ),
            origin=fork2_origin,
        )
        yield EventYield(
            event=CustomEvent(
                event_type=ReActEventType.CUSTOM_EVENT,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                event_name="kernel_stdout",
                data={"text": "fork2\n"},
                tool_name="execute_code",
                tool_call_id="call_fork2",
            ),
            origin=fork2_origin,
        )
        yield EventYield(
            event=ToolCallEndEvent(
                event_type=ReActEventType.TOOL_CALL_END,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                tool_name="execute_code",
                tool_call_id="call_fork2",
                arguments={"code": "print('fork2')"},
                result={"stdout": "fork2\n"},
                execution_time=0.05,
                success=True,
            ),
            origin=fork2_origin,
        )
        yield EventYield(
            event=LLMChunkArriveEvent(
                event_type=ReActEventType.LLM_CHUNK_ARRIVE,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                chunk=_chunk_with_content(f"fork-2:{message}"),
                accumulated_content=f"fork-2:{message}",
                chunk_index=0,
            ),
            origin=fork2_origin,
        )
        yield EventYield(
            event=LLMCallEndEvent(
                event_type=ReActEventType.LLM_CALL_END,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                response=_response_with_content(f"fork-2:{message}"),
                messages=[],
                tool_calls=[],
                execution_time=0.2,
                usage=None,
            ),
            origin=fork2_origin,
        )
        yield EventYield(
            event=ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=_ts(),
                trace_id="trace",
                func_name="agent",
                iteration=0,
                final_response=f"done:{message}",
                final_messages=[
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": f"done:{message}"},
                ],
                total_iterations=1,
                total_execution_time=0.5,
                total_tool_calls=1,
                total_llm_calls=2,
                total_token_usage=None,
            )
        )

    stdin = StringIO("batch\n")
    stdout = StringIO()
    stderr = StringIO()

    stdio(input_stream=stdin, output_stream=stdout, error_stream=stderr)(fake_agent)()

    rendered = stdout.getvalue()
    assert (
        '<assistant_message call_id="fork::fork_1" label="Assistant[fork_1]">'
        in rendered
    )
    assert (
        '<assistant_message call_id="fork::fork_2" label="Assistant[fork_2]">'
        in rendered
    )
    assert '<tool_call scope="fork_2" tool="execute_code" call_id="' in rendered
    assert '<tool_output call_id="' in rendered
    assert "fork-1:batch" in rendered
    assert "fork-2:batch" in rendered
    assert stderr.getvalue() == ""
