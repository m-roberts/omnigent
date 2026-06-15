"""AntigravityExecutor: run agents using Google's Antigravity SDK.

This executor wraps the ``google-antigravity`` Python SDK (``pip install
google-antigravity``) as the agent runtime while keeping Omnigent as the
system of record for sessions, policies, and history. It is the SDK-wrap
counterpart to :class:`omnigent.inner.openai_agents_sdk_executor.OpenAIAgentsSDKExecutor`
— pure-Python, in-process, no CLI subprocess — and follows the same
contract: ``handles_tools_internally() -> True`` (the SDK runs its own
agentic loop), per-session agent reuse, and a streaming
:meth:`run_turn` that maps SDK events onto Omnigent
:class:`~omnigent.inner.executor.ExecutorEvent` instances.

Default model is Gemini 3 Pro; the SDK can also drive Claude / GPT-OSS.
Authentication is by Antigravity / Gemini API key (``ANTIGRAVITY_API_KEY``
/ ``GEMINI_API_KEY``), threaded in from the workflow layer as an ``api_key``
override (Vertex AI is the SDK's other native path). See the note below on
why OpenAI-compatible gateway routing is not available through this SDK.

The SDK touchpoints are isolated in :meth:`_open_agent` (build + open the
``Agent``), :meth:`_build_sdk_tools` (expose Omnigent tools as callables),
and :meth:`_map_response` (``ChatResponse`` → events). They were validated
against ``google-antigravity==0.1.3`` (``Agent.chat`` is async and returns
a final ``ChatResponse`` with ``text`` / ``thoughts`` / ``tool_calls`` /
``usage_metadata``; ``LocalAgentConfig.tools`` is ``list[Callable]``) and
duck-typed so they tolerate minor drift across the still-moving v0.1.x
surface. Unit tests stub the SDK module
(``tests/inner/test_antigravity_executor.py``).

.. note::
   ``Agent.chat`` runs the full agentic loop and returns the *final*
   response, so this executor surfaces a completed turn rather than
   token-level deltas. Real-time streaming (via ``response.chunks`` /
   ``agent.conversation``) is a follow-up. Token-level streaming aside,
   tool calls / reasoning / text / usage all map faithfully.

   The SDK authenticates against Gemini (API key) or Vertex AI — it has no
   OpenAI-compatible ``base_url``, so OpenRouter / Databricks gateway
   routing is not available through it. ``base_url_override`` is threaded
   for forward-compatibility but dropped when the installed SDK's
   ``LocalAgentConfig`` doesn't accept it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, TypeAlias

from omnigent.spec.types import RetryPolicy

from .executor import (
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    Message,
    ReasoningChunk,
    TextChunk,
    ToolCallRequest,
    ToolSpec,
    TurnComplete,
)

logger = logging.getLogger(__name__)

# Antigravity's default model — Gemini 3 Pro per the SDK / CLI docs. Used
# when neither the spec nor a provider pins a model. Kept as the bare model
# id the SDK expects; gateway-routed runs pass their own (OpenRouter /
# Databricks) model ids instead.
_ANTIGRAVITY_DEFAULT_MODEL = "gemini-3-pro"

# SDK objects we treat as opaque: the Agent instance, its config, and the
# streamed events are duck-typed by the methods below. Kept as ``Any`` so
# ``google-antigravity`` stays an optional import at type-check time — the
# executor only touches the SDK when actually instantiated.
SDKAgent: TypeAlias = Any  # type: ignore[explicit-any]
SDKResponse: TypeAlias = Any  # type: ignore[explicit-any]
SDKTool: TypeAlias = Any  # type: ignore[explicit-any]
ToolArgs: TypeAlias = dict[str, Any]  # type: ignore[explicit-any]
ToolResult: TypeAlias = Any  # type: ignore[explicit-any]

# Tool-execution callback wired in by the harness :class:`ExecutorAdapter`
# (it assigns ``executor._tool_executor`` when unset). Routes an Omnigent
# tool call ``(name, args)`` back through the Session's tool registry —
# this is how the in-SDK agent reaches Omnigent's sys / sub-agent / MCP
# tools, policies and all.
ToolExecutor: TypeAlias = Callable[  # type: ignore[explicit-any]
    [str, dict[str, Any]], Awaitable[dict[str, Any]]
]


def _ensure_antigravity_sdk() -> ModuleType:
    """Import and return the ``google.antigravity`` module.

    :returns: The imported ``google.antigravity`` module.
    :raises ImportError: If the ``google-antigravity`` package isn't
        installed — surfaced on the first :meth:`run_turn` so an absent
        package is a request-time error, not an app-boot crash.
    """
    try:
        from google import antigravity  # type: ignore[attr-defined]

        return antigravity
    except ImportError as exc:
        raise ImportError(
            "AntigravityExecutor requires the 'google-antigravity' package. "
            "Install it with: pip install google-antigravity (or "
            "pip install 'omnigent[antigravity]')."
        ) from exc


def _latest_user_text(messages: list[Message]) -> str:
    """Extract the newest user-authored text to feed the agent's next turn.

    The Antigravity SDK keeps its own conversation state per agent, so each
    turn only needs the latest user input rather than the full transcript.
    Concatenates the text parts of the last ``user`` message; falls back to
    the last message of any role if no user message is present.

    :param messages: The Omnigent turn message list (role / content dicts).
    :returns: The user input text for this turn, or ``""`` when none.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        return _content_to_text(message.get("content"))
    if messages:
        return _content_to_text(messages[-1].get("content"))
    return ""


def _content_to_text(content: Any) -> str:  # type: ignore[explicit-any]
    """Flatten a message ``content`` value to plain text.

    Handles the three shapes Omnigent messages carry: a bare string, a
    list of content blocks (``{"type": "text"|"input_text", "text": ...}``),
    or some other JSON value (serialized as a last resort).

    :param content: The message ``content`` field.
    :returns: A plain-text rendering of *content*.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return json.dumps(content)


@dataclass
class _AntigravitySessionState:
    """Per-session state for the Antigravity executor.

    :param agent: Cached SDK ``Agent`` instance reused across turns so the
        SDK's own conversation state persists, or ``None`` before the first
        turn opens one.
    :param agent_signature: ``(model, system_prompt, tool_signature)`` key;
        a change forces an agent rebuild (system prompt / tool set changed).
    """

    agent: SDKAgent = None
    agent_signature: tuple[str, str, str] | None = field(default=None)


class AntigravityExecutor(Executor):
    """Execute turns using the Google Antigravity SDK."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url_override: str | None = None,
        gateway_host: str | None = None,
        gateway_auth_command: str | None = None,
        profile: str | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """Create an AntigravityExecutor.

        :param model: Constructor-level model default applied when a per-turn
            :attr:`ExecutorConfig.model` is not set, e.g. ``"gemini-3-pro"``.
            Threaded from ``HARNESS_ANTIGRAVITY_MODEL``. ``None`` falls back
            to :data:`_ANTIGRAVITY_DEFAULT_MODEL`.
        :param api_key: Antigravity / Gemini API key (or an OpenAI-compatible
            gateway key when ``base_url_override`` is set). Threaded from
            ``HARNESS_ANTIGRAVITY_API_KEY``. ``None`` lets the SDK read its
            own ambient ``GEMINI_API_KEY`` / ``ANTIGRAVITY_API_KEY``.
        :param base_url_override: OpenAI-compatible gateway base URL for
            OpenRouter / LiteLLM / Databricks routing. ``None`` uses the
            SDK's native Google endpoint.
        :param gateway_host: Gateway workspace host origin (Databricks path),
            paired with *gateway_auth_command* for dynamic token refresh.
        :param gateway_auth_command: Shell command that prints a bearer token
            (Databricks token refresh). ``None`` uses the static *api_key*.
        :param profile: ``~/.databrickscfg`` profile name for the Databricks
            fallback path. ``None`` skips Databricks credential resolution.
        :param retry_policy: Optional retry policy; reserved for parity with
            the other SDK executors. ``None`` uses defaults.
        """
        self._model_override = model
        self._api_key = api_key
        self._base_url_override = base_url_override
        self._gateway_host = gateway_host
        self._gateway_auth_command = gateway_auth_command
        self._profile = profile
        self._retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self._session_states: dict[str, _AntigravitySessionState] = {}
        # Assigned by the harness ExecutorAdapter when unset; the SDK tools we
        # build for the agent route their invocations back through this so the
        # agent can drive Omnigent's sys / sub-agent / MCP tools under policy.
        self._tool_executor: ToolExecutor | None = None

    def supports_streaming(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    def handles_tools_internally(self) -> bool:
        # The Antigravity SDK runs its own agentic loop and executes its own
        # (and any MCP-provided) tools, so the Session must not re-execute
        # tools on ToolCallRequest — they are informational here.
        return True

    def max_context_tokens(self) -> int | None:
        return None

    def _session_key(self, messages: list[Message]) -> str:
        """Resolve the per-session key from the turn's trailing message."""
        if messages:
            last = messages[-1]
            if last.get("session_id"):
                return str(last["session_id"])
            metadata = last.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("session_id"):
                return str(metadata["session_id"])
        return "default"

    @staticmethod
    def _tool_signature(tools: list[ToolSpec]) -> str:
        """Stable cache key for a tool set (names only — enough to detect change)."""
        names = sorted(str(tool.get("name", "")) for tool in tools)
        return json.dumps(names, separators=(",", ":"))

    async def close_session(self, session_key: str) -> None:
        """Close and drop the SDK agent for *session_key*, if any."""
        state = self._session_states.pop(session_key, None)
        if state is not None and state.agent is not None:
            await self._close_agent(state.agent)

    async def close(self) -> None:
        """Close every live SDK agent."""
        for state in list(self._session_states.values()):
            if state.agent is not None:
                await self._close_agent(state.agent)
        self._session_states.clear()

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        """Run one turn through the Antigravity SDK, streaming events.

        :param messages: The conversation messages for this turn.
        :param tools: Omnigent tool specs exposed to the agent.
        :param system_prompt: The agent's system instructions.
        :param config: Per-turn config; ``config.model`` (e.g. from the
            REPL ``/model`` command) wins over the constructor default.
        :yields: :class:`TextChunk`, :class:`ReasoningChunk`,
            :class:`ToolCallRequest`, and a terminal :class:`TurnComplete`
            (or :class:`ExecutorError` on failure).
        """
        model = (config.model if config and config.model else None) or self._model_override or (
            _ANTIGRAVITY_DEFAULT_MODEL
        )
        session_key = self._session_key(messages)
        prompt = _latest_user_text(messages)

        try:
            agent = await self._ensure_agent(
                session_key,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )
        except ImportError as exc:
            yield ExecutorError(message=str(exc), retryable=False)
            return
        except Exception as exc:
            logger.exception("Antigravity agent construction failed")
            yield ExecutorError(message=f"Antigravity agent setup failed: {exc}", retryable=False)
            return

        try:
            # ``Agent.chat`` runs the full agentic loop (tool calls and all)
            # and returns the final ``ChatResponse``. The SDK exposes a
            # streaming surface too (``response.chunks`` / ``agent.conversation``);
            # token-level streaming is a follow-up — see module docstring.
            response = await agent.chat(prompt)
        except Exception as exc:
            logger.exception("Antigravity turn failed")
            yield ExecutorError(message=f"Antigravity turn failed: {exc}", retryable=True)
            return

        final_text = ""
        usage: dict[str, Any] | None = None
        for event in self._map_response(response):
            if isinstance(event, TextChunk):
                final_text += event.text
            yield event
        usage = self._extract_usage(response)
        yield TurnComplete(response=final_text or None, usage=usage)

    # ── SDK touchpoints (isolated; duck-typed; verified against v0.1.3) ──

    async def _ensure_agent(
        self,
        session_key: str,
        *,
        model: str,
        system_prompt: str,
        tools: list[ToolSpec],
    ) -> SDKAgent:
        """Return a cached SDK agent for *session_key*, rebuilding on change.

        The agent is rebuilt when the model, system prompt, or tool set
        changes (so the SDK sees the current configuration), otherwise the
        existing instance is reused to preserve its conversation state.
        """
        signature = (model, system_prompt, self._tool_signature(tools))
        state = self._session_states.get(session_key)
        if state is not None and state.agent is not None and state.agent_signature == signature:
            return state.agent

        if state is not None and state.agent is not None:
            await self._close_agent(state.agent)

        agent = await self._open_agent(model=model, system_prompt=system_prompt, tools=tools)
        self._session_states[session_key] = _AntigravitySessionState(
            agent=agent, agent_signature=signature
        )
        return agent

    async def _open_agent(
        self,
        *,
        model: str,
        system_prompt: str,
        tools: list[ToolSpec],
    ) -> SDKAgent:
        """Construct and open a ``google.antigravity.Agent``.

        Isolated SDK touchpoint. Builds a ``LocalAgentConfig`` from the
        resolved model / system prompt / credentials / tools and enters the
        agent's async context. Optional config fields (``api_key`` / ``model``
        / ``tools``, and ``base_url`` if a future SDK accepts it) are passed
        only when the installed ``LocalAgentConfig`` accepts them.

        Omnigent's tools (sys shell / file, sub-agent delegation, MCP, …) are
        exposed to the agent as callables (``LocalAgentConfig.tools``) whose
        invocations route back through :attr:`_tool_executor` — the same
        bridge the openai-agents harness uses — so the agent runs them under
        Omnigent's policy and sandbox. This is what lets an Antigravity agent
        act as a Polly / Debby orchestrator or worker.
        """
        antigravity = _ensure_antigravity_sdk()
        config_kwargs: dict[str, Any] = {"system_instructions": system_prompt or None}
        sdk_tools = self._build_sdk_tools(antigravity, tools)
        if sdk_tools:
            config_kwargs["tools"] = sdk_tools
        config = self._build_local_agent_config(antigravity, model=model, kwargs=config_kwargs)
        agent = antigravity.Agent(config)
        # The SDK documents Agent as an async context manager; enter it if so.
        if hasattr(agent, "__aenter__"):
            agent = await agent.__aenter__()
        return agent

    def _build_sdk_tools(
        self,
        antigravity: ModuleType,  # noqa: ARG002 — kept for signature parity / future SDK tool helpers
        tools: list[ToolSpec],
    ) -> list[SDKTool]:
        """Build SDK tools (plain callables) from Omnigent tool specs.

        ``LocalAgentConfig.tools`` is ``list[Callable[..., Any]]`` — the SDK
        introspects each callable's ``__name__`` / ``__doc__`` to build its
        function declaration. Each callable routes its invocation back through
        :attr:`_tool_executor`, so the in-SDK agent reaches Omnigent's tool
        registry (sys shell / file, sub-agents, MCP) under policy. This is
        what lets an Antigravity agent act as a Polly / Debby orchestrator or
        worker.

        Returns ``[]`` when there are no tools or no executor bridge yet (the
        agent then runs with its native + MCP tools only).

        :param antigravity: The imported ``google.antigravity`` module.
        :param tools: Omnigent tool specs (``name`` / ``description`` /
            ``parameters``).
        :returns: A list of named async callables, or ``[]``.
        """
        if not tools or self._tool_executor is None:
            return []
        sdk_tools: list[SDKTool] = []
        for tool in tools:
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                continue
            description = tool.get("description")
            description = description if isinstance(description, str) else ""
            sdk_tools.append(self._make_tool_callable(name, description))
        return sdk_tools

    def _make_tool_callable(
        self, tool_name: str, description: str
    ) -> Callable[..., Awaitable[ToolResult]]:  # type: ignore[explicit-any]
        """Build a named async callable the SDK can register as a tool.

        The callable accepts the SDK's argument shape (keyword args, a single
        dict, or a JSON string) and forwards it to :attr:`_tool_executor`. Its
        ``__name__`` / ``__doc__`` are set so the SDK's function-declaration
        introspection picks up the tool name and description.

        :param tool_name: The Omnigent tool name, e.g. ``"sys_shell"``.
        :param description: Human-readable tool description for the model.
        :returns: An async callable bound to *tool_name*.
        """

        async def _invoke(*args: Any, **kwargs: Any) -> ToolResult:  # type: ignore[explicit-any]
            if self._tool_executor is None:
                return {"error": f"No tool executor for '{tool_name}'"}
            tool_args: dict[str, Any] = {}
            if kwargs:
                tool_args = dict(kwargs)
            elif args and isinstance(args[0], dict):
                tool_args = args[0]
            elif args and isinstance(args[0], str):
                try:
                    parsed = json.loads(args[0])
                    tool_args = parsed if isinstance(parsed, dict) else {"input": args[0]}
                except json.JSONDecodeError:
                    tool_args = {"input": args[0]}
            return await self._tool_executor(tool_name, tool_args)

        # The SDK builds the function declaration from these attributes.
        _invoke.__name__ = tool_name
        _invoke.__qualname__ = tool_name
        _invoke.__doc__ = description or tool_name
        return _invoke

    def _build_local_agent_config(
        self,
        antigravity: ModuleType,
        *,
        model: str,
        kwargs: dict[str, Any],
    ) -> Any:  # type: ignore[explicit-any]
        """Build a ``LocalAgentConfig``, passing only supported optional fields.

        :param antigravity: The imported ``google.antigravity`` module.
        :param model: The resolved model id to pin.
        :param kwargs: Base config kwargs (system instructions, tools).
        :returns: A ``LocalAgentConfig`` instance.
        """
        local_config_cls = antigravity.LocalAgentConfig
        supported = self._config_field_names(local_config_cls)
        candidate: dict[str, Any] = dict(kwargs)
        candidate["model"] = model
        if self._api_key:
            candidate["api_key"] = self._api_key
        if self._base_url_override:
            candidate["base_url"] = self._base_url_override
        # Drop any field the installed SDK doesn't accept rather than crash.
        filtered = {
            key: value
            for key, value in candidate.items()
            if supported is None or key in supported
        }
        return local_config_cls(**filtered)

    @staticmethod
    def _config_field_names(config_cls: Any) -> set[str] | None:  # type: ignore[explicit-any]
        """Best-effort set of accepted ``LocalAgentConfig`` field names.

        Inspects the constructor signature so unsupported kwargs are dropped
        before instantiation. Returns ``None`` when the signature can't be
        introspected (``**kwargs`` constructor), in which case the caller
        passes every candidate field through.
        """
        import inspect

        try:
            params = inspect.signature(config_cls).parameters
        except (TypeError, ValueError):
            return None
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return None
        return {name for name in params if name != "self"}

    def _map_response(self, response: SDKResponse) -> list[ExecutorEvent]:
        """Map a finished ``ChatResponse`` to Omnigent events.

        The SDK's ``ChatResponse`` exposes ``thoughts`` (reasoning),
        ``tool_calls`` (the calls the agent made during its loop —
        informational, since ``handles_tools_internally()`` is ``True``), and
        ``text`` (the final answer). Order: reasoning → tool calls → text.

        :param response: The ``ChatResponse`` returned by ``agent.chat``.
        :returns: The ordered list of events to yield (excluding the terminal
            :class:`TurnComplete`, which the caller emits).
        """
        events: list[ExecutorEvent] = []
        for thought in getattr(response, "thoughts", None) or []:
            text = getattr(thought, "text", None)
            if isinstance(text, str) and text:
                events.append(ReasoningChunk(delta=text, event_type="reasoning_text"))
        for call in getattr(response, "tool_calls", None) or []:
            tool_call = self._map_tool_call(call)
            if tool_call is not None:
                events.append(tool_call)
        text = self._response_text(response)
        if text:
            events.append(TextChunk(text=text))
        return events

    @staticmethod
    def _map_tool_call(call: Any) -> ToolCallRequest | None:  # type: ignore[explicit-any]
        """Map an SDK ``ToolCall`` to a :class:`ToolCallRequest`.

        ``ToolCall`` carries ``name`` (a ``BuiltinTools`` enum or ``str``),
        ``args`` (dict), and an optional ``id``.
        """
        raw_name = getattr(call, "name", None)
        # ``name`` may be a ``BuiltinTools`` enum; ``.value`` or ``str()`` both
        # yield the wire name.
        name = getattr(raw_name, "value", raw_name)
        if not isinstance(name, str) or not name:
            return None
        raw_args = getattr(call, "args", None)
        args: ToolArgs = raw_args if isinstance(raw_args, dict) else {}
        call_id = getattr(call, "id", None)
        metadata = {"call_id": call_id} if call_id else {}
        return ToolCallRequest(name=name, args=args, metadata=metadata)

    @staticmethod
    def _response_text(response: SDKResponse) -> str:
        """Return the final assistant text from a ``ChatResponse``."""
        value = getattr(response, "text", None)
        # ``text`` is a property on ChatResponse; tolerate a callable form too.
        if callable(value):
            value = value()
        return str(value) if value else ""

    @staticmethod
    def _extract_usage(response: SDKResponse) -> dict[str, Any] | None:
        """Map ``ChatResponse.usage_metadata`` to Omnigent's usage dict shape.

        :param response: The finished ``ChatResponse``.
        :returns: A usage dict (``input_tokens`` / ``output_tokens`` /
            ``total_tokens``), or ``None`` when the SDK reports no usage.
        """
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return None
        usage: dict[str, Any] = {}
        prompt_tokens = getattr(meta, "prompt_token_count", None)
        output_tokens = getattr(meta, "candidates_token_count", None)
        total_tokens = getattr(meta, "total_token_count", None)
        cached = getattr(meta, "cached_content_token_count", None)
        if prompt_tokens is not None:
            usage["input_tokens"] = prompt_tokens
        if output_tokens is not None:
            usage["output_tokens"] = output_tokens
        if total_tokens is not None:
            usage["total_tokens"] = total_tokens
        if cached is not None:
            usage["cache_read_input_tokens"] = cached
        return usage or None

    @staticmethod
    async def _close_agent(agent: SDKAgent) -> None:
        """Best-effort close of an SDK agent's async context."""
        closer = getattr(agent, "__aexit__", None)
        if closer is not None:
            try:
                await closer(None, None, None)
            except Exception:  # noqa: BLE001 — agent teardown is best-effort
                logger.debug("Antigravity agent close failed", exc_info=True)
            return
        aclose = getattr(agent, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 — agent teardown is best-effort
                logger.debug("Antigravity agent aclose failed", exc_info=True)
