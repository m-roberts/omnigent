"""
Unit tests for :class:`omnigent.inner.antigravity_executor.AntigravityExecutor`.

The fakes here mirror the real ``google-antigravity==0.1.3`` surface that the
executor depends on: ``Agent.chat`` is async and returns a ``ChatResponse``
exposing ``text`` / ``thoughts`` / ``tool_calls`` / ``usage_metadata``, and
``LocalAgentConfig.tools`` is ``list[Callable]``. They let the mapping logic be
tested without the SDK package or network.
"""

from __future__ import annotations

from typing import Any

import pytest

from omnigent.inner import antigravity_executor as ag
from omnigent.inner.antigravity_executor import AntigravityExecutor, _latest_user_text
from omnigent.inner.executor import (
    ExecutorError,
    ReasoningChunk,
    TextChunk,
    ToolCallRequest,
    TurnComplete,
)

# ── Fakes mirroring the real SDK shapes ─────────────────────────────────


class _FakeToolCall:
    def __init__(self, name: str, args: dict[str, Any], call_id: str | None = None) -> None:
        self.name = name
        self.args = args
        self.id = call_id


class _FakeThought:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeUsage:
    def __init__(self) -> None:
        self.prompt_token_count = 11
        self.candidates_token_count = 7
        self.total_token_count = 18
        self.cached_content_token_count = 2


class _FakeChatResponse:
    """Mirror of ``google.antigravity.types.ChatResponse`` (the bits we read)."""

    def __init__(
        self,
        *,
        text: str = "",
        thoughts: list[Any] | None = None,
        tool_calls: list[Any] | None = None,
        usage: Any = None,
    ) -> None:
        self.text = text
        self.thoughts = thoughts or []
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage


class _FakeAgent:
    def __init__(self, config: Any, response: Any) -> None:
        self.config = config
        self._response = response
        self.closed = False
        self.prompts: list[str] = []

    async def __aenter__(self) -> _FakeAgent:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.closed = True

    async def chat(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return self._response


class _FakeLocalAgentConfig:
    def __init__(
        self,
        *,
        system_instructions: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        tools: Any = None,
    ) -> None:
        self.system_instructions = system_instructions
        self.model = model
        self.api_key = api_key
        self.tools = tools


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, response: Any) -> dict[str, Any]:
    """Patch ``_ensure_antigravity_sdk`` to return a fake module.

    :returns: A dict the test can read to inspect the constructed agent/config.
    """
    captured: dict[str, Any] = {}

    class _FakeModule:
        LocalAgentConfig = _FakeLocalAgentConfig

        @staticmethod
        def Agent(config: Any) -> _FakeAgent:
            agent = _FakeAgent(config, response)
            captured["agent"] = agent
            captured["config"] = config
            return agent

    monkeypatch.setattr(ag, "_ensure_antigravity_sdk", lambda: _FakeModule())
    return captured


async def _drain(
    executor: AntigravityExecutor,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> list[Any]:
    events: list[Any] = []
    async for event in executor.run_turn(messages, tools=tools or [], system_prompt="sys"):
        events.append(event)
    return events


# ── Tests ───────────────────────────────────────────────────────────────


def test_latest_user_text_prefers_last_user_message() -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]
    assert _latest_user_text(messages) == "second"


@pytest.mark.asyncio
async def test_chat_response_maps_all_event_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeChatResponse(
        text="Hello world",
        thoughts=[_FakeThought("pondering")],
        tool_calls=[_FakeToolCall("search", {"q": "x"}, "c1")],
        usage=_FakeUsage(),
    )
    captured = _install_fake_sdk(monkeypatch, response)
    executor = AntigravityExecutor(model="gemini-3-pro", api_key="k")

    events = await _drain(executor, [{"role": "user", "content": "hi", "session_id": "s1"}])

    texts = [e for e in events if isinstance(e, TextChunk)]
    assert [t.text for t in texts] == ["Hello world"]

    reasoning = [e for e in events if isinstance(e, ReasoningChunk)]
    assert len(reasoning) == 1 and reasoning[0].delta == "pondering"

    calls = [e for e in events if isinstance(e, ToolCallRequest)]
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].args == {"q": "x"}
    assert calls[0].metadata == {"call_id": "c1"}

    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert len(completes) == 1
    assert completes[0].response == "Hello world"
    assert completes[0].usage == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "cache_read_input_tokens": 2,
    }

    assert captured["config"].model == "gemini-3-pro"
    assert captured["config"].api_key == "k"
    assert captured["agent"].prompts == ["hi"]


@pytest.mark.asyncio
async def test_text_only_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch, _FakeChatResponse(text="final answer"))
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "q"}])

    texts = [e for e in events if isinstance(e, TextChunk)]
    assert [t.text for t in texts] == ["final answer"]
    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert completes[0].response == "final answer"
    assert completes[0].usage is None


@pytest.mark.asyncio
async def test_missing_sdk_yields_executor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> Any:
        raise ImportError("no google-antigravity")

    monkeypatch.setattr(ag, "_ensure_antigravity_sdk", _raise)
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "q"}])

    assert len(events) == 1
    assert isinstance(events[0], ExecutorError)
    assert "google-antigravity" in events[0].message


@pytest.mark.asyncio
async def test_sys_tools_exposed_as_callables_routing_through_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omnigent tools become callable SDK tools whose calls hit ``_tool_executor``.

    This is what lets an Antigravity agent drive Omnigent's sys / sub-agent
    tools under policy (needed to run Polly / Debby).
    """
    captured = _install_fake_sdk(monkeypatch, _FakeChatResponse(text="done"))
    executor = AntigravityExecutor()

    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_tool_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, args))
        return {"ok": True}

    # The harness ExecutorAdapter assigns this in production; set it directly.
    executor._tool_executor = _fake_tool_executor

    tool_specs = [
        {
            "name": "sys_shell",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        }
    ]

    await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}], tool_specs)

    sdk_tools = captured["config"].tools
    assert sdk_tools is not None and len(sdk_tools) == 1
    sdk_tool = sdk_tools[0]
    # LocalAgentConfig.tools is list[Callable]; the SDK reads __name__/__doc__.
    assert callable(sdk_tool)
    assert sdk_tool.__name__ == "sys_shell"
    assert sdk_tool.__doc__ == "Run a shell command"

    # Invoking the callable (kwargs form) routes back through the bridge.
    assert await sdk_tool(cmd="ls") == {"ok": True}
    # Single-dict argument form also works (SDK arg-shape tolerance).
    assert await sdk_tool({"cmd": "pwd"}) == {"ok": True}
    assert calls == [("sys_shell", {"cmd": "ls"}), ("sys_shell", {"cmd": "pwd"})]


@pytest.mark.asyncio
async def test_no_tool_executor_means_no_sdk_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a tool-executor bridge, no SDK tools are built (agent uses native)."""
    captured = _install_fake_sdk(monkeypatch, _FakeChatResponse(text="done"))
    executor = AntigravityExecutor()  # _tool_executor stays None

    await _drain(
        executor,
        [{"role": "user", "content": "go"}],
        [{"name": "sys_shell", "description": "", "parameters": {}}],
    )

    assert captured["config"].tools is None


@pytest.mark.asyncio
async def test_agent_reused_across_turns_same_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second turn on the same session reuses the cached agent."""
    captured = _install_fake_sdk(monkeypatch, _FakeChatResponse(text="ok"))
    executor = AntigravityExecutor()

    await _drain(executor, [{"role": "user", "content": "one", "session_id": "s1"}])
    first_agent = captured["agent"]
    await _drain(executor, [{"role": "user", "content": "two", "session_id": "s1"}])

    assert captured["agent"] is first_agent
    assert first_agent.prompts == ["one", "two"]
