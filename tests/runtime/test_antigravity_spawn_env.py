"""
Tests for ``_build_antigravity_spawn_env`` in
``omnigent/runtime/workflow.py``.

The spawn-env builder maps ``spec.executor`` fields to
``HARNESS_ANTIGRAVITY_*`` env vars the antigravity harness wrap reads at
first-turn time. Mirrors ``test_openai_agents_sdk_spawn_env.py``.

Unit test — no subprocess spawn, no real httpx.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.runtime.workflow import _build_antigravity_spawn_env
from omnigent.spec.types import (
    AgentSpec,
    ApiKeyAuth,
    DatabricksAuth,
    ExecutorSpec,
    LLMConfig,
)


@pytest.fixture(autouse=True)
def _isolate_global_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point OMNIGENT_CONFIG_HOME at an empty temp dir so the developer's
    real ``~/.omnigent/config.yaml`` doesn't leak into these tests."""
    monkeypatch.setenv("OMNIGENT_CONFIG_HOME", str(tmp_path))


def _make_spec(
    *,
    model: str | None = "gemini-3-pro",
    profile: str | None = None,
    auth: ApiKeyAuth | DatabricksAuth | None = None,
) -> AgentSpec:
    """Build a minimal antigravity :class:`AgentSpec` for spawn-env tests."""
    config: dict[str, object] = {"harness": "antigravity"}
    if model is not None:
        config["model"] = model
    if profile is not None:
        config["profile"] = profile
    return AgentSpec(
        spec_version=1,
        name="test-antigravity",
        instructions="You are a test agent.",
        executor=ExecutorSpec(type="omnigent", config=config, model=model, auth=auth),
        llm=LLMConfig(model=model) if model is not None else None,
    )


def test_model_threads_into_env_var() -> None:
    """``executor.model`` is encoded into ``HARNESS_ANTIGRAVITY_MODEL``."""
    env = _build_antigravity_spawn_env(_make_spec(model="gemini-3-pro"))
    assert env["HARNESS_ANTIGRAVITY_MODEL"] == "gemini-3-pro"


def test_no_model_omits_env_var() -> None:
    """A spec with no model omits ``HARNESS_ANTIGRAVITY_MODEL`` entirely."""
    env = _build_antigravity_spawn_env(_make_spec(model=None))
    assert "HARNESS_ANTIGRAVITY_MODEL" not in env


def test_api_key_auth_threads_key_and_base_url() -> None:
    """``ApiKeyAuth`` sets the API key and (when present) the gateway base URL."""
    env = _build_antigravity_spawn_env(
        _make_spec(
            model="gemini-3-pro",
            auth=ApiKeyAuth(api_key="ag-secret", base_url="https://openrouter.ai/api/v1"),
        )
    )
    assert env["HARNESS_ANTIGRAVITY_API_KEY"] == "ag-secret"
    assert env["HARNESS_ANTIGRAVITY_GATEWAY_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_databricks_auth_threads_profile() -> None:
    """``DatabricksAuth`` sets ``HARNESS_ANTIGRAVITY_DATABRICKS_PROFILE``."""
    env = _build_antigravity_spawn_env(
        _make_spec(model="databricks-claude-sonnet-4-6", auth=DatabricksAuth(profile="dev"))
    )
    assert env["HARNESS_ANTIGRAVITY_DATABRICKS_PROFILE"] == "dev"


def test_legacy_profile_threads_into_env_var() -> None:
    """An explicit ``executor.config['profile']`` sets the Databricks profile."""
    env = _build_antigravity_spawn_env(_make_spec(model="gemini-3-pro", profile="my-profile"))
    assert env["HARNESS_ANTIGRAVITY_DATABRICKS_PROFILE"] == "my-profile"


def test_databricks_model_prefix_auto_routes_default_profile() -> None:
    """A ``databricks-`` model with no auth falls back to the DEFAULT profile."""
    env = _build_antigravity_spawn_env(_make_spec(model="databricks-gpt-5-5", profile=None))
    assert env["HARNESS_ANTIGRAVITY_DATABRICKS_PROFILE"] == "DEFAULT"


def test_no_auth_non_databricks_model_is_minimal() -> None:
    """A plain Gemini model with no auth yields only the model var.

    The wrap then falls back to the SDK's ambient
    ``GEMINI_API_KEY`` / ``ANTIGRAVITY_API_KEY``.
    """
    env = _build_antigravity_spawn_env(_make_spec(model="gemini-3-pro", profile=None))
    assert env == {"HARNESS_ANTIGRAVITY_MODEL": "gemini-3-pro"}
