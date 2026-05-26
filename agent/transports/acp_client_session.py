"""Session adapter for ACP client runtime.

Owns one ACP session per Hermes session. Drives ``session/new`` + ``session/prompt``,
consumes streaming ``session/update`` notifications (AgentMessageChunk), handles
server-initiated requests (fs/*, terminal/*, permission — declined by default in v1),
and returns a TurnResult that acp_runtime.run_acp_client_turn() can splice into
the ``messages`` list.

Lifecycle:
    session = ACPClientSession(command="claude-agent-acp")
    session.ensure_started(cwd="/home/x/proj")      # spawns + initialize + session/new
    result = session.run_turn("hello")               # blocks until session/prompt returns
    # result.final_text          → assistant text returned to caller
    # result.projected_messages  → list of {role, content} for messages list
    # result.tool_iterations     → count of tool-shaped update events (skill nudge)
    # result.should_retire       → True if session wedged (timeout, crash)
    session.close()                                  # session/close + subprocess teardown

Threading model: single-threaded from the caller's perspective.
The underlying ACPClient owns its own reader threads but exposes
blocking-with-timeout queues that this adapter polls in a loop.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.transports.acp_client import ACPClient, ACPClientError

logger = logging.getLogger(__name__)

# ACP wire method names (from acp.meta)
_METHOD_INITIALIZE = "initialize"
_METHOD_SESSION_NEW = "session/new"
_METHOD_SESSION_PROMPT = "session/prompt"
_METHOD_SESSION_CLOSE = "session/close"
_METHOD_SESSION_CANCEL = "session/cancel"
_METHOD_SESSION_UPDATE = "session/update"

# Server-initiated (client-side) methods we receive
_METHOD_FS_READ = "fs/read_text_file"
_METHOD_FS_WRITE = "fs/write_text_file"
_METHOD_PERMISSION = "session/request_permission"
_METHOD_TERMINAL_CREATE = "terminal/create"
_METHOD_TERMINAL_OUTPUT = "terminal/output"
_METHOD_TERMINAL_RELEASE = "terminal/release"
_METHOD_TERMINAL_WAIT = "terminal/wait_for_exit"
_METHOD_TERMINAL_KILL = "terminal/kill"

# ACP session/update discriminator values for streaming chunks
_UPDATE_AGENT_MESSAGE = "agent_message_chunk"
_UPDATE_AGENT_THOUGHT = "agent_thought_chunk"
_UPDATE_TOOL_CALL = "tool_call_update"
_UPDATE_TOOL_CALL_START = "tool_call"

# How many trailing stderr lines to show in error messages
_STDERR_TAIL_LINES = 12


@dataclass
class TurnResult:
    """Result of one user→assistant turn through an ACP-compliant agent."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    # True when the session is wedged (timeout, crash, bad response).
    # The caller should retire and re-create the session on the next turn.
    should_retire: bool = False


def _extract_text_from_update(params: dict) -> str:
    """Extract plain text from an ACP session/update notification params.

    ``session/update`` params carry:
      { "sessionId": "...", "update": { "sessionUpdate": "agent_message_chunk",
                                        "content": { "type": "text", "text": "..." } } }
    Returns the text string, or "" if not a text chunk.
    """
    update = params.get("update") or {}
    content = update.get("content") or {}
    if isinstance(content, dict) and content.get("type") == "text":
        return content.get("text") or ""
    return ""


def _is_tool_iteration(params: dict) -> bool:
    """Return True if the update represents a tool call completion."""
    update = params.get("update") or {}
    kind = update.get("sessionUpdate") or update.get("session_update") or ""
    return kind in {_UPDATE_TOOL_CALL, _UPDATE_TOOL_CALL_START}


class ACPClientSession:
    """One ACP session per Hermes session, lifetime owned by AIAgent.

    Not thread-safe — one caller drives it at a time, matching how AIAgent's
    run_conversation() loop is structured today.
    """

    def __init__(
        self,
        *,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        on_delta: Optional[Callable[[str], None]] = None,
        client_factory: Optional[Callable[..., ACPClient]] = None,
    ) -> None:
        """
        Args:
            command: ACP agent binary to spawn (e.g. "claude-agent-acp").
            args: Additional arguments to pass to the command.
            env: Extra environment variables for the subprocess.
            on_delta: Optional callback invoked with each text delta during streaming.
                      Bridges to Hermes' ``_fire_stream_delta`` for live output.
            client_factory: Inject a custom ACPClient constructor for testing.
        """
        self._command = command
        self._args = list(args or [])
        self._env = env
        self._on_delta = on_delta
        self._client_factory = client_factory or ACPClient

        self._client: Optional[ACPClient] = None
        self._session_id: Optional[str] = None
        self._closed = False

    # ---------- lifecycle ----------

    def ensure_started(self, cwd: Optional[str] = None) -> str:
        """Spawn the subprocess, do the initialize handshake, and start a
        session. Returns the ACP session_id. Idempotent — repeated calls
        return the same session_id."""
        if self._session_id is not None:
            return self._session_id
        if self._client is None:
            self._client = self._client_factory(
                command=self._command,
                args=self._args,
                env=self._env,
            )
        self._client.initialize(
            client_name="hermes",
            client_version=_get_hermes_version(),
        )
        result = self._client.request(
            _METHOD_SESSION_NEW,
            # mcpServers is required by the ACP schema (NewSessionRequest); send
            # an empty list — Hermes handles its own MCP surface, not the agent.
            {"cwd": cwd or os.getcwd(), "mcpServers": []},
            timeout=15,
        )
        session_id = result.get("sessionId") or result.get("session_id") or ""
        if not session_id:
            raise ACPClientError(
                code=-32603,
                message=(
                    "ACP session/new returned no sessionId "
                    f"(payload keys: {sorted(result.keys())})"
                ),
            )
        self._session_id = session_id
        logger.info(
            "ACP client session started: id=%s command=%r cwd=%s",
            self._session_id[:8],
            self._command,
            cwd or os.getcwd(),
        )
        return self._session_id

    def close(self) -> None:
        """Send session/close and tear down the subprocess."""
        if self._closed:
            return
        self._closed = True
        if self._client is not None and self._session_id is not None:
            try:
                self._client.request(
                    _METHOD_SESSION_CLOSE,
                    {"sessionId": self._session_id},
                    timeout=5,
                )
            except Exception:
                pass  # best-effort
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._session_id = None

    def __enter__(self) -> "ACPClientSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        cwd: Optional[str] = None,
        turn_timeout: float = 600.0,
        notification_poll_timeout: float = 0.25,
    ) -> TurnResult:
        """Send a user message and block until session/prompt returns.

        Streams session/update notifications to on_delta as they arrive.
        Projects streamed content into projected_messages so memory/skill
        review keep working.

        Returns a TurnResult. Sets should_retire=True on crash/timeout.
        """
        result = TurnResult()
        # Ensure session is open (lazy start on first turn)
        try:
            self.ensure_started(cwd=cwd)
        except (ACPClientError, TimeoutError, RuntimeError) as exc:
            result.error = f"ACP client session startup failed: {exc}"
            result.should_retire = True
            return result

        assert self._client is not None and self._session_id is not None

        user_text = _coerce_user_input(user_input)

        # Build ACP prompt request
        prompt_params = {
            "sessionId": self._session_id,
            "prompt": [{"type": "text", "text": user_text}],
        }

        # session/prompt is a request that blocks until the agent returns
        # PromptResponse. While waiting, the agent sends session/update
        # notifications which arrive in the _notifications queue.
        # We poll both in a deadline loop.
        deadline = time.monotonic() + turn_timeout
        text_chunks: list[str] = []

        # Send session/prompt in a background thread so we can drain
        # notifications concurrently. The result arrives via a shared dict.
        _response: dict = {}
        _error: list = []  # [exc] if the request raised

        def _do_request() -> None:
            try:
                r = self._client.request(
                    _METHOD_SESSION_PROMPT,
                    prompt_params,
                    timeout=turn_timeout,
                )
                _response["result"] = r
            except (ACPClientError, TimeoutError, RuntimeError) as exc:
                _error.append(exc)

        req_thread = threading.Thread(target=_do_request, daemon=True)
        req_thread.start()

        def _process_notification(note: dict) -> None:
            """Apply a single session/update notification to result + text_chunks.

            Factored out so the same logic runs during the live drain loop and
            the post-join tail-drain (notifications that arrived between the
            last loop poll and req_thread completion would otherwise be lost).
            """
            if note.get("method") != _METHOD_SESSION_UPDATE:
                return
            params = note.get("params") or {}
            delta = _extract_text_from_update(params)
            if delta:
                text_chunks.append(delta)
                if self._on_delta is not None:
                    try:
                        self._on_delta(delta)
                    except Exception:
                        logger.debug("on_delta callback raised", exc_info=True)
            if _is_tool_iteration(params):
                result.tool_iterations += 1

        # Drain notifications while waiting for the prompt response.
        # session/prompt blocks for the entire turn; req_thread sends it while
        # this loop concurrently drains session/update chunks.
        # _send_lock on ACPClient ensures the two threads don't interleave
        # writes to the same BufferedWriter (see ACPClient._send).
        while req_thread.is_alive() and time.monotonic() < deadline:
            if not (self._client and self._client.is_alive()):
                result.error = self._format_error("ACP agent subprocess exited unexpectedly")
                result.should_retire = True
                break

            # Handle server-initiated requests (fs/*, permission, terminal/*)
            sreq = self._client.take_server_request(timeout=0)
            if sreq is not None:
                self._handle_server_request(sreq)
                continue

            # Drain streaming notifications (session/update)
            note = self._client.take_notification(timeout=notification_poll_timeout)
            if note is None:
                continue
            _process_notification(note)

        req_thread.join(timeout=2.0)

        # Tail-drain: consume notifications that were parsed by the reader
        # thread between the last loop poll and req_thread completing. These
        # would be silently dropped without this drain — short responses that
        # fit in the first chunks are the most likely to be affected.
        if self._client is not None:
            while True:
                note = self._client.take_notification(timeout=0)
                if note is None:
                    break
                _process_notification(note)

        if _error:
            exc = _error[0]
            result.error = f"ACP session/prompt failed: {exc}"
            if isinstance(exc, TimeoutError) or (
                isinstance(exc, ACPClientError) and exc.code < 0
            ):
                result.should_retire = True
            return result

        if "result" not in _response and not result.should_retire:
            # Deadline hit without response
            result.error = f"ACP turn timed out after {turn_timeout}s"
            result.should_retire = True
            result.interrupted = True
            return result

        if result.should_retire:
            return result

        # Assemble final text from streamed chunks. If chunks are empty,
        # look for text in the PromptResponse itself (some implementations
        # may put content there instead of streaming).
        prompt_result = _response.get("result") or {}
        assembled = "".join(text_chunks)
        if not assembled:
            # Fallback: look for content in the PromptResponse
            for block in (prompt_result.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "text":
                    assembled += block.get("text") or ""

        result.final_text = assembled

        # Project into messages so curator/memory/skill review can see the turn.
        if assembled:
            result.projected_messages.append(
                {"role": "assistant", "content": assembled}
            )

        return result

    # ---------- internals ----------

    def _handle_server_request(self, req: dict) -> None:
        """Handle server-initiated requests from the ACP agent.

        In v1 we decline all server-initiated requests (fs/*, terminal/*,
        permission) because Hermes controls those surfaces itself. A future
        iteration could bridge these to Hermes' tools.
        """
        if self._client is None:
            return
        method = req.get("method", "")
        rid = req.get("id")

        if method == _METHOD_PERMISSION:
            # Decline permission escalation — user controls Hermes' permission
            # model through Hermes' own config.
            self._client.respond(rid, {"granted": False})
        elif method in {
            _METHOD_FS_READ, _METHOD_FS_WRITE,
            _METHOD_TERMINAL_CREATE, _METHOD_TERMINAL_OUTPUT,
            _METHOD_TERMINAL_RELEASE, _METHOD_TERMINAL_WAIT,
            _METHOD_TERMINAL_KILL,
        }:
            # Decline fs/terminal proxying — Hermes drives its own tool
            # executor. ACP agents that need fs/terminal ops should spawn
            # their own processes.
            logger.debug("ACP client: declining server request %r (not proxied in v1)", method)
            self._client.respond_error(
                rid,
                code=-32601,
                message=f"Method not supported by Hermes ACP client v1: {method}",
            )
        else:
            logger.warning("ACP client: unknown server request %r", method)
            self._client.respond_error(
                rid,
                code=-32601,
                message=f"Unknown method: {method}",
            )

    def _format_error(self, prefix: str) -> str:
        """Build a user-facing error string, appending stderr tail when available."""
        if self._client is None:
            return prefix
        try:
            tail = self._client.stderr_tail(_STDERR_TAIL_LINES)
        except Exception:
            return prefix
        if not tail:
            return prefix
        joined = "\n".join(line.rstrip() for line in tail if line)
        if not joined.strip():
            return prefix
        return f"{prefix}\nACP agent stderr (last {len(tail)} lines):\n{joined}"


def _coerce_user_input(user_input: Any) -> str:
    """Collapse Hermes/OpenAI rich content into plain text for ACP session/prompt."""
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                if item is not None:
                    parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item_type in {"image", "image_url", "input_image"}:
                parts.append("[image attached]")
        text = "\n\n".join(p for p in parts if p).strip()
        return text or "What do you see in this image?"
    return "" if user_input is None else str(user_input)


def _get_hermes_version() -> str:
    """Best-effort Hermes version string for ACP initialize."""
    try:
        from importlib.metadata import version
        return version("hermes-agent")
    except Exception:  # pragma: no cover
        return "0.0.0"
