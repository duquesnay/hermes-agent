"""Tests for agent.transports.acp_client_session — ACP session adapter.

Tests cover session lifecycle (ensure_started, close), turn execution,
streaming delta projection, should_retire policy on crash/timeout, and
server-request handling (permission decline, fs/terminal decline).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional
from unittest.mock import MagicMock, call

import pytest

from agent.transports.acp_client import ACPClientError
from agent.transports.acp_client_session import (
    ACPClientSession,
    TurnResult,
    _coerce_user_input,
    _extract_text_from_update,
    _is_tool_iteration,
)


# ---------------------------------------------------------------------------
# Helpers — mock ACPClient
# ---------------------------------------------------------------------------


def _make_session(
    *,
    command: str = "fake-acp",
    args=None,
    on_delta=None,
    client_mock: Optional[MagicMock] = None,
) -> tuple[ACPClientSession, MagicMock]:
    """Return an ACPClientSession with a mock ACPClient injected."""
    if client_mock is None:
        client_mock = MagicMock()
        client_mock.is_alive.return_value = True
        client_mock.initialize.return_value = {"protocolVersion": 1}
        client_mock.request.return_value = {}
        client_mock.take_notification.return_value = None
        client_mock.take_server_request.return_value = None
        client_mock.stderr_tail.return_value = []

    session = ACPClientSession(
        command=command,
        args=args,
        on_delta=on_delta,
        client_factory=lambda **kw: client_mock,
    )
    return session, client_mock


# ---------------------------------------------------------------------------
# Tests: ensure_started / session lifecycle
# ---------------------------------------------------------------------------


class TestEnsureStarted:
    def test_ensure_started_initializes_and_creates_session(self):
        """ensure_started() calls initialize then session/new, stores session_id."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [
            {"sessionId": "sess-abc-123"},  # session/new
        ]

        sid = session.ensure_started(cwd="/tmp")
        assert sid == "sess-abc-123"
        assert session._session_id == "sess-abc-123"

        # initialize was called once
        mock_client.initialize.assert_called_once()
        # session/new was called with correct cwd
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "session/new"
        assert call_args[0][1]["cwd"] == "/tmp"

    def test_ensure_started_idempotent(self):
        """ensure_started() called twice returns same session_id."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "sess-001"}

        sid1 = session.ensure_started(cwd="/tmp")
        sid2 = session.ensure_started(cwd="/other")
        assert sid1 == sid2 == "sess-001"
        # initialize and session/new called only once
        assert mock_client.initialize.call_count == 1
        assert mock_client.request.call_count == 1

    def test_ensure_started_raises_on_missing_session_id(self):
        """ensure_started() raises ACPClientError if no sessionId in response."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {}  # no sessionId

        with pytest.raises(ACPClientError) as exc_info:
            session.ensure_started()
        assert "sessionId" in str(exc_info.value)
        assert session._session_id is None

    def test_ensure_started_error_sets_should_retire(self):
        """run_turn() → ensure_started() failure sets should_retire=True."""
        session, mock_client = _make_session()
        mock_client.initialize.side_effect = ACPClientError(
            code=-32603, message="initialize failed"
        )

        result = session.run_turn("hello")
        assert result.should_retire is True
        assert result.error is not None
        assert "startup" in result.error.lower()


# ---------------------------------------------------------------------------
# Tests: close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_sends_session_close_and_closes_client(self):
        """close() calls session/close then client.close()."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [
            {"sessionId": "sess-xyz"},  # session/new
            {},  # session/close
        ]
        session.ensure_started()
        session.close()

        # session/close request was made
        close_call = mock_client.request.call_args_list[-1]
        assert close_call[0][0] == "session/close"
        assert close_call[0][1]["sessionId"] == "sess-xyz"
        # client.close() was called
        mock_client.close.assert_called_once()

    def test_close_idempotent(self):
        """close() called twice does not raise."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "sess-x"}
        session.ensure_started()
        session.close()
        session.close()  # must not raise

    def test_context_manager_calls_close(self):
        """ACPClientSession used as context manager calls close() on exit."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [{"sessionId": "s1"}, {}]
        with session:
            session.ensure_started()
        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: run_turn — happy path
# ---------------------------------------------------------------------------


class TestRunTurn:
    def _setup_happy_session(self):
        """Return session + mock configured for a successful prompt turn."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [
            {"sessionId": "sess-happy"},  # session/new
            {"stopReason": "end_turn"},   # session/prompt
        ]
        return session, mock_client

    def test_run_turn_sends_session_prompt(self):
        """run_turn() sends session/prompt with the user text."""
        session, mock_client = self._setup_happy_session()
        # No streaming notifications
        mock_client.take_notification.side_effect = [
            None,  # polled once, returns None
            None,  # second poll → triggers req_thread to finish
        ]

        result = session.run_turn("hello world", cwd="/tmp")
        # Check session/prompt was called
        prompt_call = None
        for c in mock_client.request.call_args_list:
            if c[0][0] == "session/prompt":
                prompt_call = c
                break
        assert prompt_call is not None
        assert prompt_call[0][1]["sessionId"] == "sess-happy"
        assert prompt_call[0][1]["prompt"][0]["text"] == "hello world"

    def test_run_turn_collects_text_from_streaming_chunks(self):
        """Text chunks from session/update notifications are assembled."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [
            {"sessionId": "sess-stream"},  # session/new
        ]

        deltas_received = []

        def on_delta(text):
            deltas_received.append(text)

        session2 = ACPClientSession(
            command="fake",
            on_delta=on_delta,
            client_factory=lambda **kw: mock_client,
        )

        # Notifications: two text chunks, then None to stop
        # The session/prompt result arrives through request()
        notes_iter = iter([
            {
                "method": "session/update",
                "params": {
                    "sessionId": "sess-stream",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "Hello "},
                    },
                },
            },
            {
                "method": "session/update",
                "params": {
                    "sessionId": "sess-stream",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "world!"},
                    },
                },
            },
            None,
        ])

        def take_notif(timeout=0.0):
            try:
                return next(notes_iter)
            except StopIteration:
                return None

        mock_client.take_notification.side_effect = take_notif
        mock_client.request.return_value = {"sessionId": "sess-stream"}

        # Override request to return promptResponse after chunks
        call_count = [0]
        def req_side_effect(method, params=None, timeout=30):
            call_count[0] += 1
            if method == "session/new":
                return {"sessionId": "sess-stream"}
            if method == "session/prompt":
                # Small sleep to let notification drain happen first
                time.sleep(0.05)
                return {"stopReason": "end_turn"}
            return {}

        mock_client.request.side_effect = req_side_effect

        result = session2.run_turn("test", cwd="/tmp")
        assert "Hello " in result.final_text
        assert "world!" in result.final_text
        assert "Hello " in deltas_received
        assert "world!" in deltas_received

    def test_run_turn_projects_message_into_messages(self):
        """A final text turn is projected into projected_messages."""
        session, mock_client = _make_session()

        def req_side_effect(method, params=None, timeout=30):
            if method == "session/new":
                return {"sessionId": "sess-proj"}
            if method == "session/prompt":
                time.sleep(0.02)
                return {"stopReason": "end_turn"}
            return {}

        mock_client.request.side_effect = req_side_effect

        # Push one text chunk via notification
        notes = [
            {
                "method": "session/update",
                "params": {
                    "sessionId": "sess-proj",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "Answer here."},
                    },
                },
            },
            None,
        ]
        notes_iter = iter(notes)
        mock_client.take_notification.side_effect = lambda timeout=0.0: next(notes_iter, None)

        result = session.run_turn("question")
        assert len(result.projected_messages) == 1
        assert result.projected_messages[0]["role"] == "assistant"
        assert result.projected_messages[0]["content"] == "Answer here."


# ---------------------------------------------------------------------------
# Tests: should_retire policy
# ---------------------------------------------------------------------------


class TestShouldRetire:
    def test_subprocess_crash_sets_should_retire(self):
        """When the process exits unexpectedly, should_retire=True."""
        session, mock_client = _make_session()

        call_count = [0]
        def req_side_effect(method, params=None, timeout=30):
            call_count[0] += 1
            if method == "session/new":
                return {"sessionId": "sess-crash"}
            if method == "session/prompt":
                # Simulate blocking while process dies
                time.sleep(0.1)
                raise RuntimeError("stdin closed unexpectedly")
            return {}

        mock_client.request.side_effect = req_side_effect
        # Process dies after first poll
        alive_iter = iter([True, True, False, False])
        mock_client.is_alive.side_effect = lambda: next(alive_iter, False)

        result = session.run_turn("hello")
        assert result.should_retire is True
        assert result.error is not None

    def test_session_prompt_acp_error_sets_should_retire_for_negative_code(self):
        """ACPClientError with negative code (system error) → should_retire."""
        session, mock_client = _make_session()

        def req_side_effect(method, params=None, timeout=30):
            if method == "session/new":
                return {"sessionId": "sess-err"}
            if method == "session/prompt":
                time.sleep(0.02)
                raise ACPClientError(code=-32603, message="internal error")
            return {}

        mock_client.request.side_effect = req_side_effect

        result = session.run_turn("hello")
        assert result.error is not None
        assert "session/prompt failed" in result.error
        assert result.should_retire is True

    def test_session_prompt_timeout_sets_should_retire(self):
        """TimeoutError from session/prompt sets should_retire."""
        session, mock_client = _make_session()

        def req_side_effect(method, params=None, timeout=30):
            if method == "session/new":
                return {"sessionId": "sess-timeout"}
            if method == "session/prompt":
                raise TimeoutError("ACP method timed out")
            return {}

        mock_client.request.side_effect = req_side_effect

        result = session.run_turn("hello")
        assert result.should_retire is True
        assert result.error is not None


# ---------------------------------------------------------------------------
# Tests: server request handling
# ---------------------------------------------------------------------------


class TestServerRequestHandling:
    def test_permission_request_declined(self):
        """Permission requests from the server are declined (granted: False)."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [
            {"sessionId": "sess-perm"},  # session/new
        ]

        # Simulate the internal _handle_server_request call
        session.ensure_started()
        req = {"id": 42, "method": "session/request_permission", "params": {"permissionId": "exec"}}
        session._handle_server_request(req)

        mock_client.respond.assert_called_once_with(42, {"granted": False})

    def test_fs_write_declined_with_error(self):
        """fs/write_text_file is declined with respond_error."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "sess-fs"}
        session.ensure_started()

        req = {"id": 7, "method": "fs/write_text_file", "params": {"path": "/etc/passwd", "content": "bad"}}
        session._handle_server_request(req)

        mock_client.respond_error.assert_called_once()
        call_args = mock_client.respond_error.call_args
        # respond_error(rid, code=..., message=...) — rid is positional
        assert call_args[0][0] == 7
        assert call_args[1]["code"] == -32601  # method not supported

    def test_unknown_server_request_declined_with_error(self):
        """Unknown server requests receive respond_error."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "sess-unk"}
        session.ensure_started()

        req = {"id": 99, "method": "some/unknown_method", "params": {}}
        session._handle_server_request(req)
        mock_client.respond_error.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_text_from_text_chunk(self):
        params = {
            "sessionId": "s",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "hello"},
            },
        }
        assert _extract_text_from_update(params) == "hello"

    def test_extract_text_from_non_text_chunk_returns_empty(self):
        params = {
            "sessionId": "s",
            "update": {
                "sessionUpdate": "tool_call_update",
                "content": {"type": "image"},
            },
        }
        assert _extract_text_from_update(params) == ""

    def test_is_tool_iteration_for_tool_call_update(self):
        params = {"update": {"sessionUpdate": "tool_call_update"}}
        assert _is_tool_iteration(params) is True

    def test_is_tool_iteration_for_agent_message_returns_false(self):
        params = {"update": {"sessionUpdate": "agent_message_chunk"}}
        assert _is_tool_iteration(params) is False

    def test_coerce_user_input_string(self):
        assert _coerce_user_input("hello") == "hello"

    def test_coerce_user_input_list_of_text_blocks(self):
        result = _coerce_user_input([{"type": "text", "text": "hello"}])
        assert result == "hello"

    def test_coerce_user_input_image_block_replaced(self):
        result = _coerce_user_input([{"type": "image"}])
        assert "[image attached]" in result

    def test_coerce_user_input_none(self):
        assert _coerce_user_input(None) == ""

    def test_coerce_user_input_integer(self):
        assert _coerce_user_input(42) == "42"
