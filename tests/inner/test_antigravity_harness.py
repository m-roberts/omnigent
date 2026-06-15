"""
Tests for the ``harness: antigravity`` wrap shape.

Mirror of the openai-agents wrap tests — verifies the wrap module has the
same shape (registry entry, FastAPI app routes, env-var-driven lazy
executor construction). Does NOT exercise the real ``google-antigravity``
SDK; the inner :class:`AntigravityExecutor.__init__` is mocked so the
tests pass without the package installed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from omnigent.inner import antigravity_harness
from omnigent.runtime.harnesses import _HARNESS_MODULES


def test_harness_module_registered_in_module_registry() -> None:
    """``"antigravity"`` resolves to the harness module path.

    Without this entry, the runner subprocess can't find the wrap when the
    parent tries to spawn it for an ``executor.harness == "antigravity"``
    spec.
    """
    assert _HARNESS_MODULES.get("antigravity") == "omnigent.inner.antigravity_harness"


def test_create_app_returns_fastapi_with_required_routes() -> None:
    """``create_app()`` returns a FastAPI app exposing the harness API.

    The :class:`AntigravityExecutor` is constructed lazily on the first
    turn (not at app build time), so this passes without
    ``google-antigravity`` installed.
    """
    app = antigravity_harness.create_app()
    # The harness API routes are mounted via a lazily-included router, so the
    # OpenAPI schema is the reliable surface to assert against.
    paths = set(app.openapi().get("paths", {}).keys())
    assert "/health" in paths
    assert "/v1/sessions/{conversation_id}/events" in paths


def test_executor_factory_threads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``HARNESS_ANTIGRAVITY_*`` env vars thread into the executor ctor.

    Locks in the canonical env-var contract the spawn-env builder
    (``_build_antigravity_spawn_env`` in workflow.py) emits.
    """
    monkeypatch.setenv("HARNESS_ANTIGRAVITY_MODEL", "gemini-3-pro")
    monkeypatch.setenv("HARNESS_ANTIGRAVITY_API_KEY", "ag-test-key")
    monkeypatch.setenv("HARNESS_ANTIGRAVITY_GATEWAY_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("HARNESS_ANTIGRAVITY_GATEWAY_HOST", "https://openrouter.ai")
    monkeypatch.setenv("HARNESS_ANTIGRAVITY_GATEWAY_AUTH_COMMAND", "printf token")
    monkeypatch.setenv("HARNESS_ANTIGRAVITY_DATABRICKS_PROFILE", "my-profile")

    captured: dict[str, Any] = {}

    def _fake_init(self: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    with patch(
        "omnigent.inner.antigravity_harness.AntigravityExecutor.__init__",
        _fake_init,
    ):
        antigravity_harness._build_antigravity_executor()

    assert captured["model"] == "gemini-3-pro"
    assert captured["api_key"] == "ag-test-key"
    assert captured["base_url_override"] == "https://openrouter.ai/api/v1"
    assert captured["gateway_host"] == "https://openrouter.ai"
    assert captured["gateway_auth_command"] == "printf token"
    assert captured["profile"] == "my-profile"


def test_executor_factory_defaults_to_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env vars resolve to ``None`` (SDK falls back to ambient creds)."""
    for var in (
        "HARNESS_ANTIGRAVITY_MODEL",
        "HARNESS_ANTIGRAVITY_API_KEY",
        "HARNESS_ANTIGRAVITY_GATEWAY_BASE_URL",
        "HARNESS_ANTIGRAVITY_GATEWAY_HOST",
        "HARNESS_ANTIGRAVITY_GATEWAY_AUTH_COMMAND",
        "HARNESS_ANTIGRAVITY_DATABRICKS_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)

    captured: dict[str, Any] = {}

    def _fake_init(self: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    with patch(
        "omnigent.inner.antigravity_harness.AntigravityExecutor.__init__",
        _fake_init,
    ):
        antigravity_harness._build_antigravity_executor()

    assert captured["model"] is None
    assert captured["api_key"] is None
    assert captured["base_url_override"] is None
    assert captured["profile"] is None
