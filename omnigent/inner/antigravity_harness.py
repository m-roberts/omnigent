"""``harness: antigravity`` wrap.

Thin module exposing :func:`create_app` — the entrypoint the shared
:mod:`omnigent.runtime.harnesses._runner` invokes after the parent
process resolves ``"antigravity"`` to this module via
:data:`omnigent.runtime.harnesses._HARNESS_MODULES`.

Internally, instantiates
:class:`omnigent.runtime.harnesses._executor_adapter.ExecutorAdapter`
around a :class:`omnigent.inner.antigravity_executor.AntigravityExecutor`
configured from env vars the parent process sets before spawning. Mirrors
the openai-agents wrap (``openai_agents_sdk_harness.py``); see the
claude-sdk module's docstring for the v1 config-flow rationale (env vars
vs per-request).

Like the OpenAI-Agents SDK wrap, Antigravity is a pure-Python SDK harness:
no CLI binary, no sandbox subprocess, and the model is a simple
constructor override from the spawn env.

Env vars read at startup:

- ``HARNESS_ANTIGRAVITY_MODEL``: model identifier the inner executor pins
  for every turn, e.g. ``"gemini-3-pro"``. Constructor-level override —
  wins over the per-turn ``request.model`` (which carries the agent NAME,
  not an LLM identifier under the harness contract). ``None`` falls back
  to the executor's built-in default.
- ``HARNESS_ANTIGRAVITY_API_KEY``: direct Antigravity / Gemini API key
  (``ANTIGRAVITY_API_KEY`` / ``GEMINI_API_KEY``), or an OpenAI-compatible
  gateway key when ``HARNESS_ANTIGRAVITY_GATEWAY_BASE_URL`` is also set.
  Written when the agent spec declares ``executor.auth: {type: api_key,
  …}`` or a generic key/gateway provider resolves. Takes precedence over
  the SDK's ambient credential lookup.
- ``HARNESS_ANTIGRAVITY_GATEWAY_BASE_URL``: OpenAI-compatible gateway base
  URL for OpenRouter / LiteLLM / Databricks routing, e.g.
  ``"https://openrouter.ai/api/v1"``.
- ``HARNESS_ANTIGRAVITY_GATEWAY_HOST``: gateway workspace host origin
  (Databricks path), paired with the auth command for token refresh.
- ``HARNESS_ANTIGRAVITY_GATEWAY_AUTH_COMMAND``: shell command that prints
  a bearer token (Databricks token refresh), used instead of a static key.
- ``HARNESS_ANTIGRAVITY_DATABRICKS_PROFILE``: ``~/.databrickscfg`` profile
  name for the Databricks fallback path.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from omnigent.inner.antigravity_executor import AntigravityExecutor
from omnigent.inner.executor import Executor
from omnigent.runtime.harnesses._executor_adapter import ExecutorAdapter

_logger = logging.getLogger(__name__)

# Env-var keys the wrap reads at executor construction time. See the module
# docstring for semantics. Centralizing as constants so misconfigurations
# surface as a single grep target.
_ENV_MODEL = "HARNESS_ANTIGRAVITY_MODEL"
_ENV_API_KEY = "HARNESS_ANTIGRAVITY_API_KEY"
_ENV_GATEWAY_BASE_URL = "HARNESS_ANTIGRAVITY_GATEWAY_BASE_URL"
_ENV_GATEWAY_HOST = "HARNESS_ANTIGRAVITY_GATEWAY_HOST"
_ENV_GATEWAY_AUTH_COMMAND = "HARNESS_ANTIGRAVITY_GATEWAY_AUTH_COMMAND"
_ENV_DATABRICKS_PROFILE = "HARNESS_ANTIGRAVITY_DATABRICKS_PROFILE"


def _build_antigravity_executor() -> Executor:
    """Construct an :class:`AntigravityExecutor` from env-var config.

    Called lazily by the :class:`ExecutorAdapter` on the first turn, so an
    absent ``google-antigravity`` package surfaces as a request-time error
    rather than an app-boot crash.

    :returns: A configured :class:`AntigravityExecutor` instance.
    """
    return AntigravityExecutor(
        model=os.environ.get(_ENV_MODEL) or None,
        api_key=os.environ.get(_ENV_API_KEY) or None,
        base_url_override=os.environ.get(_ENV_GATEWAY_BASE_URL) or None,
        gateway_host=os.environ.get(_ENV_GATEWAY_HOST) or None,
        gateway_auth_command=os.environ.get(_ENV_GATEWAY_AUTH_COMMAND) or None,
        profile=os.environ.get(_ENV_DATABRICKS_PROFILE) or None,
    )


def create_app() -> FastAPI:
    """Build the antigravity harness's FastAPI app.

    Required entry point per the harness contract — the runner imports this
    module (resolved from
    :data:`omnigent.runtime.harnesses._HARNESS_MODULES`) and invokes
    ``create_app()`` to get the app it serves. The wrapped
    :class:`AntigravityExecutor` is constructed lazily on the first turn.

    :returns: The FastAPI app from :class:`ExecutorAdapter`'s
        :meth:`build` method.
    """
    adapter = ExecutorAdapter(executor_factory=_build_antigravity_executor)
    return adapter.build()
