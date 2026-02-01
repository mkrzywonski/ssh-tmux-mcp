#!/usr/bin/env python3
import argparse
import base64
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union


MAX_MESSAGE_BYTES = 2 * 1024 * 1024
MAX_READ_BYTES = 10 * 1024 * 1024
MAX_WRITE_B64 = 15 * 1024 * 1024
MAX_PATCH_BYTES = 1 * 1024 * 1024
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_LOG_BYTES = 25 * 1024 * 1024
SERVER_PROTOCOL_VERSION = "2024-11-05"
DEBUG = os.getenv("SSH_TMUX_MCP_DEBUG", "0") not in ("0", "", "false", "False")
FRAMING = "lsp"

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b[()][AB012]|\x1b\[\?[0-9;]*[hlm]')


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)


def _log(message: str) -> None:
    if not DEBUG:
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[ssh-tmux-mcp {ts}] {message}", file=sys.stderr, flush=True)


class MCPError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ControlMasterGone(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _read_exact(n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _negotiate_protocol(requested: Optional[str]) -> str:
    if not requested:
        return SERVER_PROTOCOL_VERSION
    if not (len(requested) == 10 and requested[4] == "-" and requested[7] == "-"):
        return SERVER_PROTOCOL_VERSION
    if not requested.replace("-", "").isdigit():
        return SERVER_PROTOCOL_VERSION
    return min(requested, SERVER_PROTOCOL_VERSION)


def read_message() -> Optional[Dict[str, Any]]:
    # LSP-style headers: Content-Length: N\r\n\r\n<json>
    headers = {}
    line = sys.stdin.buffer.readline()
    _log(f"read_message: first line={line!r}")
    if not line:
        return None
    # Skip any stray blank lines
    while line in (b"\r\n", b"\n"):
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        _log(f"read_message: skipping blank, next line={line!r}")
    if line.lstrip().startswith(b"{"):
        global FRAMING
        FRAMING = "json"
        _log("read_message: detected json line without headers")
        if len(line) > MAX_MESSAGE_BYTES:
            raise ValueError("incoming message too large")
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            raise ValueError("invalid json message")
    while line not in (b"\r\n", b"\n"):
        parts = line.decode("ascii", errors="ignore").split(":", 1)
        if len(parts) == 2:
            headers[parts[0].strip().lower()] = parts[1].strip()
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        _log(f"read_message: header line={line!r}")
    if "content-length" not in headers:
        return None
    length = int(headers["content-length"])
    if length > MAX_MESSAGE_BYTES:
        raise ValueError("incoming message too large")
    body = _read_exact(length)
    if not body:
        return None
    _log(f"read_message: payload length={len(body)}")
    return json.loads(body.decode("utf-8"))


def send_message(payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    if FRAMING == "json":
        sys.stdout.buffer.write(data + b"\n")
        sys.stdout.buffer.flush()
        _log(f"send_message: json bytes={len(data)}")
        return
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    _log(f"send_message: lsp bytes={len(data)}")


def _run_cmd(cmd: List[str], input_bytes: Optional[bytes] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    _log(f"run_cmd: {cmd!r} input_bytes={len(input_bytes) if input_bytes is not None else 0} timeout={timeout}")
    try:
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise MCPError(f"command timed out after {timeout}s: {cmd[0]}")
    _log(f"run_cmd: code={proc.returncode} stdout={len(proc.stdout)} stderr={len(proc.stderr)}")
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace"), proc.stderr.decode("utf-8", errors="replace")


def _read_socket_info(socket_path: str) -> Dict[str, str]:
    info_path = f"{socket_path}.info"
    data: Dict[str, str] = {}
    try:
        with open(info_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()
    except FileNotFoundError:
        _log(f"socket info not found: {info_path}")
        raise MCPError(f"socket info not found: {info_path}")
    _log(f"socket info loaded: {data}")
    return data


def _parse_unified_diff(patch_text: str) -> List[Tuple[int, List[str]]]:
    hunks: List[Tuple[int, List[str]]] = []
    lines = patch_text.splitlines()
    current_hunk: Optional[List[str]] = None
    current_start: Optional[int] = None
    saw_file = False

    for line in lines:
        if line.startswith("diff ") or line.startswith("index "):
            continue
        if line.startswith("--- "):
            saw_file = True
            current_hunk = None
            current_start = None
            continue
        if line.startswith("+++ "):
            saw_file = True
            current_hunk = None
            current_start = None
            continue
        if line.startswith("@@ "):
            parts = line.split(" ")
            if len(parts) < 3 or not parts[1].startswith("-"):
                raise MCPError("invalid hunk header")
            old_span = parts[1][1:]
            old_start = old_span.split(",")[0]
            try:
                current_start = int(old_start)
            except ValueError as exc:
                raise MCPError("invalid hunk header") from exc
            current_hunk = []
            hunks.append((current_start, current_hunk))
            continue
        if line.startswith("\\ No newline at end of file"):
            continue
        if current_hunk is not None:
            current_hunk.append(line)
        elif line.strip() and not saw_file:
            raise MCPError("patch missing file headers")

    if not hunks:
        raise MCPError("no hunks found in patch")
    return hunks


def _apply_unified_diff(original_text: str, patch_text: str) -> str:
    hunks = _parse_unified_diff(patch_text)
    original_lines = original_text.splitlines(keepends=True)
    out_lines: List[str] = []
    idx = 0

    for start, hunk_lines in hunks:
        target = max(0, start - 1)
        if target < idx or target > len(original_lines):
            raise MCPError("hunk position out of range")
        out_lines.extend(original_lines[idx:target])
        idx = target
        for hline in hunk_lines:
            if not hline:
                continue
            tag = hline[0]
            text = hline[1:]
            if tag == " ":
                if idx >= len(original_lines):
                    raise MCPError("context extends beyond file")
                if original_lines[idx].rstrip("\n") != text:
                    raise MCPError("context mismatch while applying patch")
                out_lines.append(original_lines[idx])
                idx += 1
            elif tag == "-":
                if idx >= len(original_lines):
                    raise MCPError("deletion extends beyond file")
                if original_lines[idx].rstrip("\n") != text:
                    raise MCPError("deletion mismatch while applying patch")
                idx += 1
            elif tag == "+":
                out_lines.append(text + "\n")
            else:
                raise MCPError(f"unexpected patch line: {hline}")

    out_lines.extend(original_lines[idx:])
    return "".join(out_lines)


class TmuxClientBase:
    """Shared tmux session logic for both SSH and local modes.

    Subclasses must implement:
      _run_tmux(tmux_shell_cmd) -> str  — run a full tmux command string
      _read_stream_from_offset(log_path, offset, max_bytes) -> dict
      _get_log_path() -> str
      _ensure_tmux_session(session) -> str
    """

    def __init__(self, max_log_bytes: int = DEFAULT_MAX_LOG_BYTES, prompt_timeout_ms: int = 30000) -> None:
        self._log_offsets: Dict[str, int] = {}
        self.max_log_bytes = max_log_bytes
        self._prompt_token = os.urandom(4).hex()
        self._prompt_marker = f"__AISESS_{self._prompt_token}_"
        self._prompt_re = re.compile(r"__AISESS_" + self._prompt_token + r"_(\d+)__")
        self._prompt_initialized = False
        self.prompt_timeout_ms = prompt_timeout_ms

    def _run_tmux(self, tmux_shell_cmd: str) -> str:
        raise NotImplementedError

    def _read_stream_from_offset(self, log_path: str, offset: int, max_bytes: int) -> Dict[str, Any]:
        raise NotImplementedError

    def _get_log_path(self) -> str:
        raise NotImplementedError

    def _ensure_tmux_session(self, session: Optional[str]) -> str:
        raise NotImplementedError

    def _truncate_log_if_needed(self, log_path: str) -> None:
        """Truncate the log file in place if it exceeds max_log_bytes."""
        raise NotImplementedError

    def tmux_log_purge(self) -> Dict[str, Any]:
        """Purge the tmux log file entirely (e.g. to remove sensitive output)."""
        raise NotImplementedError

    def _initialize_prompt(self, quoted_session: str) -> None:
        """Export a custom PS1 containing our session token. Called once per session."""
        ps1 = f"__AISESS_{self._prompt_token}_$?__ \\\\$ "
        export_cmd = shlex.quote(f"export PS1='{ps1}'")
        self._run_tmux(f"tmux send-keys -t {quoted_session} -- {export_cmd} C-m")
        # Poll capture-pane until marker appears (up to 5s)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            time.sleep(0.1)
            out = self._run_tmux(f"tmux capture-pane -p -t {quoted_session}")
            for line in reversed(out.splitlines()):
                line = line.strip()
                if line and self._prompt_re.search(line):
                    self._prompt_initialized = True
                    return
        raise MCPError("Failed to initialize PS1 prompt marker (timed out)")

    def _check_shell_ready(self, quoted_session: str) -> bool:
        """Check if the last non-empty line in the pane matches our prompt marker."""
        out = self._run_tmux(f"tmux capture-pane -p -t {quoted_session}")
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line:
                return bool(self._prompt_re.search(line))
        return False

    def _wait_for_shell_ready(self, quoted_session: str, timeout_ms: int) -> Tuple[bool, str]:
        """Wait for shell prompt to appear. Returns (ready, pane_content)."""
        if self._check_shell_ready(quoted_session):
            return True, ""
        # Send Enter to trigger a fresh prompt
        self._run_tmux(f"tmux send-keys -t {quoted_session} '' C-m")
        time.sleep(0.2)
        deadline = time.time() + timeout_ms / 1000.0
        while time.time() < deadline:
            out = self._run_tmux(f"tmux capture-pane -p -t {quoted_session}")
            for line in reversed(out.splitlines()):
                line = line.strip()
                if line:
                    if self._prompt_re.search(line):
                        return True, out
                    break
            time.sleep(0.1)
        out = self._run_tmux(f"tmux capture-pane -p -t {quoted_session}")
        return False, out

    def _poll_log_for_marker(self, log_path: str, offset: int, timeout_ms: int, max_bytes: int) -> Dict[str, Any]:
        """Poll the log file from offset until our prompt marker appears."""
        deadline = time.time() + timeout_ms / 1000.0
        content = ""
        current_offset = offset
        last_size = offset
        timed_out = False

        while True:
            remaining = max_bytes - len(content)
            if remaining <= 0:
                break
            result = self._read_stream_from_offset(log_path, current_offset, min(remaining, 8192))
            current_offset = result["offset"]
            last_size = result.get("size", last_size)
            if result["content"]:
                content += result["content"]
                match = self._prompt_re.search(content)
                if match:
                    # Found marker — extract exit code, truncate content at marker line
                    exit_code = int(match.group(1))
                    # Cut everything from the marker line onward
                    marker_line_start = content.rfind("\n", 0, match.start())
                    if marker_line_start == -1:
                        cut_content = ""
                    else:
                        cut_content = content[:marker_line_start]
                    # Clean up
                    cut_content = _strip_ansi(cut_content)
                    cleaned_lines = [
                        line for line in cut_content.splitlines()
                        if self._prompt_marker not in line
                    ]
                    self._log_offsets[log_path] = current_offset
                    if last_size > self.max_log_bytes:
                        self._truncate_log_if_needed(log_path)
                    return {
                        "content": "\n".join(cleaned_lines).strip(),
                        "exit_code": exit_code,
                        "timed_out": False,
                    }
            if time.time() >= deadline:
                timed_out = True
                break
            time.sleep(0.1)

        # Timed out — return what we have
        content = _strip_ansi(content)
        cleaned_lines = [
            line for line in content.splitlines()
            if self._prompt_marker not in line
        ]
        self._log_offsets[log_path] = current_offset
        if last_size > self.max_log_bytes:
            self._truncate_log_if_needed(log_path)
        return {
            "content": "\n".join(cleaned_lines).strip(),
            "exit_code": None,
            "timed_out": timed_out,
        }

    def tmux_capture(self, session: Optional[str], lines: Optional[int]) -> Dict[str, Any]:
        sess = self._ensure_tmux_session(session)
        quoted = shlex.quote(sess)
        if lines is not None:
            out = self._run_tmux(f"tmux capture-pane -p -t {quoted} -S -{int(lines)}")
        else:
            out = self._run_tmux(f"tmux capture-pane -p -t {quoted}")
        return {"content": out}

    def tmux_capture_scrollback(self, session: Optional[str], max_lines: Optional[int]) -> Dict[str, Any]:
        sess = self._ensure_tmux_session(session)
        quoted = shlex.quote(sess)
        if max_lines is not None:
            out = self._run_tmux(f"tmux capture-pane -p -t {quoted} -S -{int(max_lines)}")
        else:
            out = self._run_tmux(f"tmux capture-pane -p -t {quoted} -S -")
        return {"content": out}

    def tmux_send(self, session: Optional[str], keys: str, enter: bool) -> Dict[str, Any]:
        sess = self._ensure_tmux_session(session)
        quoted = shlex.quote(sess)
        key_arg = shlex.quote(keys)
        if enter:
            self._run_tmux(f"tmux send-keys -t {quoted} -- {key_arg} C-m")
        else:
            self._run_tmux(f"tmux send-keys -t {quoted} -- {key_arg}")
        return {"ok": True}

    def tmux_run(self, session: Optional[str], command: str, timeout_ms: Optional[int] = None) -> Dict[str, Any]:
        sess = self._ensure_tmux_session(session)
        quoted = shlex.quote(sess)
        key_arg = shlex.quote(command)
        timeout = timeout_ms or self.prompt_timeout_ms

        if not self._prompt_initialized:
            self._initialize_prompt(quoted)

        # Pre-check: wait for shell to be ready
        ready, pane = self._wait_for_shell_ready(quoted, timeout)
        if not ready:
            raise MCPError(f"Shell not ready (no prompt detected). Pane:\n{pane[-500:]}")

        # Record log offset, send command
        log_path = self._get_log_path()
        start = self._read_stream_from_offset(log_path, 0, 0)["size"]
        self._run_tmux(f"tmux send-keys -t {quoted} -- {key_arg} C-m")

        # Wait for completion
        result = self._poll_log_for_marker(log_path, start, timeout, max_bytes=65536)
        return result

    def tmux_stream_read(self, session: Optional[str], max_bytes: int, reset: bool) -> Dict[str, Any]:
        self._ensure_tmux_session(session)
        log_path = self._get_log_path()
        offset = 0 if reset else self._log_offsets.get(log_path, 0)
        result = self._read_stream_from_offset(log_path, offset, max_bytes)
        self._log_offsets[log_path] = result["offset"]
        if result.get("size", 0) > self.max_log_bytes:
            self._truncate_log_if_needed(log_path)
        return result

    def tmux_cwd(self, session: Optional[str]) -> Dict[str, Any]:
        sess = self._ensure_tmux_session(session)
        quoted = shlex.quote(sess)
        out = self._run_tmux(f"tmux display-message -p -t {quoted} '#{{pane_current_path}}'").strip()
        return {"cwd": out}


class SSHClient(TmuxClientBase):
    def __init__(self, host: Optional[str], socket_path: str, root: Optional[str], tmux_session: Optional[str], timeout: int = DEFAULT_TIMEOUT, max_log_bytes: int = DEFAULT_MAX_LOG_BYTES, prompt_timeout_ms: int = 30000):
        super().__init__(max_log_bytes=max_log_bytes, prompt_timeout_ms=prompt_timeout_ms)
        self.host = host
        self.socket_path = socket_path
        self.root = posixpath.normpath(root) if root is not None else None
        self.tmux_session = tmux_session
        self.timeout = timeout
        self.info_path = f"{socket_path}.info"
        self.log_path: Optional[str] = None

    def _refresh_info(self) -> None:
        info = _read_socket_info(self.socket_path)
        host = info.get("host")
        tmux = info.get("tmux")
        log_path = info.get("log_path")
        if host:
            self.host = host
        if tmux:
            self.tmux_session = tmux
        if log_path:
            self.log_path = log_path

    def _check_socket(self) -> None:
        if self.host is None or self.tmux_session is None:
            self._refresh_info()
        if not os.path.exists(self.socket_path):
            raise ControlMasterGone(f"SSH control socket not found: {self.socket_path}")
        if self.host is None:
            raise MCPError(f"session info not found: {self.info_path}")
        code, _out, _err = _run_cmd(["ssh", "-S", self.socket_path, "-O", "check", self.host], timeout=10)
        if code != 0:
            raise ControlMasterGone("SSH ControlMaster not responding")

    def _ensure_tmux_session(self, session: Optional[str]) -> str:
        if session:
            return session
        if self.tmux_session is None:
            self._refresh_info()
        if self.tmux_session is None:
            raise MCPError(f"session info not found: {self.info_path}")
        return self.tmux_session

    def _get_log_path(self) -> str:
        if self.log_path is None:
            try:
                self._refresh_info()
            except MCPError:
                pass
        if self.log_path is None:
            self.log_path = "/tmp/ai-ssh/tmux.log"
        return self.log_path

    def _run_tmux(self, tmux_shell_cmd: str) -> str:
        return self.ssh(tmux_shell_cmd)

    def _read_stream_from_offset(self, log_path: str, offset: int, max_bytes: int) -> Dict[str, Any]:
        quoted = shlex.quote(log_path)
        size_out = self.ssh(f"stat -Lc '%s' -- {quoted}").strip()
        try:
            size = int(size_out)
        except ValueError:
            raise MCPError("unexpected log stat output")
        if size < offset:
            offset = 0
        to_read = min(max_bytes, max(0, size - offset))
        if to_read == 0:
            return {"content": "", "offset": offset, "size": size, "truncated": False}
        cmd = f"tail -c +{int(offset) + 1} -- {quoted} | head -c {int(to_read)}"
        out = self.ssh(cmd)
        new_offset = offset + to_read
        return {
            "content": out,
            "offset": new_offset,
            "size": size,
            "truncated": (size - offset) > max_bytes,
        }

    def _ssh_base(self) -> List[str]:
        return [
            "ssh",
            "-S",
            self.socket_path,
            "-o",
            "BatchMode=yes",
            self.host,
            "--",
        ]

    def _normalize_path(self, path: str) -> str:
        if self.root is None:
            return path
        root = self.root
        target = posixpath.normpath(posixpath.join(root, path.lstrip("/")))
        if posixpath.commonpath([root, target]) != root:
            raise MCPError("Path escapes configured root")
        return target

    def ssh(self, remote_cmd: str, input_bytes: Optional[bytes] = None, timeout: Optional[int] = None) -> str:
        self._check_socket()
        cmd = self._ssh_base() + [remote_cmd]
        code, out, err = _run_cmd(cmd, input_bytes=input_bytes, timeout=timeout or self.timeout)
        if code != 0:
            raise MCPError(err.strip() or f"ssh command failed: {remote_cmd}")
        return out

    def read_file(self, path: str, offset: Optional[int], length: Optional[int]) -> Dict[str, Any]:
        path = self._normalize_path(path)
        quoted = shlex.quote(path)
        if length is None:
            size_out = self.ssh(f"stat -Lc '%s' -- {quoted}").strip()
            try:
                size = int(size_out)
            except ValueError:
                raise MCPError("unexpected stat output")
            if size > MAX_READ_BYTES:
                raise MCPError(f"read exceeds {MAX_READ_BYTES} bytes; use offset/length")
        if offset is not None or length is not None:
            off = offset or 0
            cnt = "" if length is None else f" count={int(length)}"
            head_arg = f" | head -c {int(length)}" if length is not None else ""
            cmd = f"tail -c +{int(off) + 1} -- {quoted}{head_arg} | base64 -w 0"
        else:
            cmd = f"base64 -w 0 -- {quoted}"
        out = self.ssh(cmd)
        return {"encoding": "base64", "content": out}

    def write_file(self, path: str, content_b64: str) -> Dict[str, Any]:
        if len(content_b64) > MAX_WRITE_B64:
            raise MCPError(f"write exceeds {MAX_WRITE_B64} base64 bytes")
        path = self._normalize_path(path)
        quoted = shlex.quote(path)
        dir_quoted = shlex.quote(path.rsplit("/", 1)[0] if "/" in path else ".")
        cmd = f"__tmp=$(mktemp -p {dir_quoted}) && base64 -d > \"$__tmp\" && mv \"$__tmp\" {quoted}"
        self.ssh(cmd, input_bytes=content_b64.encode("utf-8"))
        return {"ok": True}

    def patch_file(self, path: str, patch_text: str, backup: bool, backup_suffix: str) -> Dict[str, Any]:
        if len(patch_text.encode("utf-8")) > MAX_PATCH_BYTES:
            raise MCPError(f"patch exceeds {MAX_PATCH_BYTES} bytes")
        path = self._normalize_path(path)
        quoted = shlex.quote(path)
        original = self.read_file(path, None, None)
        content_b64 = original.get("content", "")
        original_text = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        patched_text = _apply_unified_diff(original_text, patch_text)
        if backup:
            backup_path = path + backup_suffix
            self.ssh(f"cp -- {quoted} {shlex.quote(backup_path)}")
        patched_b64 = base64.b64encode(patched_text.encode("utf-8")).decode("ascii")
        self.write_file(path, patched_b64)
        return {"ok": True, "backup": backup}

    def list_dir(self, path: str) -> Dict[str, Any]:
        path = self._normalize_path(path)
        quoted = shlex.quote(path)
        cmd = (
            "find -P "
            + quoted
            + " -maxdepth 1 -mindepth 1 -printf '%y\t%s\t%T@\t%p\n'"
        )
        out = self.ssh(cmd)
        entries: List[Dict[str, Any]] = []
        for line in out.splitlines():
            parts = line.split("\t", 3)
            if len(parts) != 4:
                continue
            ftype, size, mtime, fullpath = parts
            entries.append(
                {
                    "type": ftype,
                    "size": int(size),
                    "mtime": float(mtime),
                    "path": fullpath,
                }
            )
        return {"entries": entries}

    def stat(self, path: str) -> Dict[str, Any]:
        path = self._normalize_path(path)
        quoted = shlex.quote(path)
        cmd = f"stat -Lc '%F\t%s\t%Y\t%a\t%u\t%g\t%n' -- {quoted}"
        out = self.ssh(cmd).strip()
        parts = out.split("\t", 6)
        if len(parts) < 7:
            raise MCPError("unexpected stat output")
        ftype, size, mtime, mode, uid, gid, name = parts
        return {
            "type": ftype,
            "size": int(size),
            "mtime": int(mtime),
            "mode": mode,
            "uid": int(uid),
            "gid": int(gid),
            "path": name,
        }

    def mkdir(self, path: str, parents: bool) -> Dict[str, Any]:
        path = self._normalize_path(path)
        quoted = shlex.quote(path)
        cmd = f"mkdir {'-p ' if parents else ''}{quoted}"
        self.ssh(cmd)
        return {"ok": True}

    def rm(self, path: str, recursive: bool) -> Dict[str, Any]:
        path = self._normalize_path(path)
        quoted = shlex.quote(path)
        cmd = f"rm {'-rf ' if recursive else ''}{quoted}"
        self.ssh(cmd)
        return {"ok": True}

    def rename(self, src: str, dst: str) -> Dict[str, Any]:
        src = self._normalize_path(src)
        dst = self._normalize_path(dst)
        cmd = f"mv {shlex.quote(src)} {shlex.quote(dst)}"
        self.ssh(cmd)
        return {"ok": True}

    def ssh_exec(self, command: str) -> Dict[str, Any]:
        out = self.ssh(command)
        return {"content": out}

    def _truncate_log_if_needed(self, log_path: str) -> None:
        quoted = shlex.quote(log_path)
        size_out = self.ssh(f"stat -Lc '%s' -- {quoted}").strip()
        try:
            size = int(size_out)
        except ValueError:
            return
        if size <= self.max_log_bytes:
            return
        dir_quoted = shlex.quote(posixpath.dirname(log_path) or ".")
        cmd = (
            f"__tmp=$(mktemp -p {dir_quoted}) && "
            f"tail -c {int(self.max_log_bytes)} -- {quoted} > \"$__tmp\" && "
            f"mv \"$__tmp\" {quoted}"
        )
        self.ssh(cmd)
        self._log_offsets[log_path] = min(self.max_log_bytes, size)

    def tmux_log_purge(self) -> Dict[str, Any]:
        log_path = self._get_log_path()
        quoted = shlex.quote(log_path)
        self.ssh(f": > {quoted}")
        self._log_offsets[log_path] = 0
        return {"ok": True, "purged": log_path}


class LocalClient(TmuxClientBase):
    def __init__(self, tmux_session: Optional[str], log_path: Optional[str], timeout: int = DEFAULT_TIMEOUT, max_log_bytes: int = DEFAULT_MAX_LOG_BYTES, prompt_timeout_ms: int = 30000) -> None:
        super().__init__(max_log_bytes=max_log_bytes, prompt_timeout_ms=prompt_timeout_ms)
        self.tmux_session = tmux_session or "shared-ai"
        self.log_path = log_path or "/tmp/ai-ssh/tmux.log"
        self.timeout = timeout

    def _run_local(self, cmd: List[str], timeout: Optional[int] = None) -> Tuple[int, str, str]:
        return _run_cmd(cmd, timeout=timeout or self.timeout)

    def _ensure_tmux_session(self, session: Optional[str]) -> str:
        return session or self.tmux_session

    def _get_log_path(self) -> str:
        return self.log_path

    def _run_tmux(self, tmux_shell_cmd: str) -> str:
        # Run a full tmux command string locally via shell to preserve quoting.
        code, out, err = self._run_local(["bash", "-c", tmux_shell_cmd])
        if code != 0:
            raise MCPError(err.strip() or "tmux command failed")
        return out

    def _read_stream_from_offset(self, log_path: str, offset: int, max_bytes: int) -> Dict[str, Any]:
        try:
            size = os.path.getsize(log_path)
        except FileNotFoundError:
            return {"content": "", "offset": 0, "size": 0, "truncated": False}
        if size < offset:
            offset = 0
        to_read = min(max_bytes, max(0, size - offset))
        if to_read == 0:
            return {"content": "", "offset": offset, "size": size, "truncated": False}
        with open(log_path, "rb") as handle:
            handle.seek(offset)
            chunk = handle.read(to_read)
        return {
            "content": chunk.decode("utf-8", errors="replace"),
            "offset": offset + to_read,
            "size": size,
            "truncated": (size - offset) > max_bytes,
        }

    def _truncate_log_if_needed(self, log_path: str) -> None:
        try:
            size = os.path.getsize(log_path)
        except OSError:
            return
        if size <= self.max_log_bytes:
            return
        keep = min(self.max_log_bytes, size)
        try:
            with open(log_path, "rb") as handle:
                handle.seek(size - keep)
                tail = handle.read(keep)
            with open(log_path, "wb") as handle:
                handle.write(tail)
        except OSError:
            return
        self._log_offsets[log_path] = keep

    def tmux_log_purge(self) -> Dict[str, Any]:
        log_path = self._get_log_path()
        try:
            os.truncate(log_path, 0)
        except OSError as e:
            raise MCPError(f"failed to purge log: {e}")
        self._log_offsets[log_path] = 0
        return {"ok": True, "purged": log_path}


TOOLS = [
    {
        "name": "sftp_list",
        "description": "List directory entries via SSH (find -maxdepth 1).",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "sftp_read",
        "description": "Read a file and return base64 content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "length": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "sftp_write",
        "description": "Write a file from base64 content.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "sftp_patch",
        "description": "Apply a unified diff to a file and write the result back.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "patch": {"type": "string"},
                "backup": {"type": "boolean"},
                "backup_suffix": {"type": "string"},
            },
            "required": ["path", "patch"],
        },
    },
    {
        "name": "sftp_stat",
        "description": "Stat a file/directory.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "sftp_mkdir",
        "description": "Create a directory.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "parents": {"type": "boolean"}},
            "required": ["path"],
        },
    },
    {
        "name": "sftp_rm",
        "description": "Remove a file or directory.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
            "required": ["path"],
        },
    },
    {
        "name": "sftp_rename",
        "description": "Rename/move a file or directory.",
        "inputSchema": {
            "type": "object",
            "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
            "required": ["src", "dst"],
        },
    },
    {
        "name": "tmux_capture",
        "description": "Capture tmux pane content.",
        "inputSchema": {
            "type": "object",
            "properties": {"session": {"type": "string"}, "lines": {"type": "integer"}},
        },
    },
    {
        "name": "tmux_capture_scrollback",
        "description": "Capture tmux scrollback content.",
        "inputSchema": {
            "type": "object",
            "properties": {"session": {"type": "string"}, "max_lines": {"type": "integer"}},
        },
    },
    {
        "name": "tmux_send",
        "description": "Send keys to a tmux session. For interactive use only; prefer tmux_run for executing commands.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": {"type": "string"},
                "keys": {"type": "string"},
                "enter": {"type": "boolean"},
            },
            "required": ["keys"],
        },
    },
    {
        "name": "tmux_run",
        "description": "Run a command in the tmux session and return its complete output. Waits for the shell to be ready, executes the command, and blocks until completion or timeout. Returns output and exit code. This is the primary tool for executing commands.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": {"type": "string"},
                "command": {"type": "string"},
                "timeout_ms": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "tmux_stream_read",
        "description": "Read new tmux output from the session log.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": {"type": "string"},
                "max_bytes": {"type": "integer"},
                "reset": {"type": "boolean"},
            },
        },
    },
    {
        "name": "tmux_cwd",
        "description": "Get current working directory of tmux pane.",
        "inputSchema": {
            "type": "object",
            "properties": {"session": {"type": "string"}},
        },
    },
    {
        "name": "tmux_log_purge",
        "description": "Purge the tmux log file entirely. Use when the log contains sensitive data (secrets, credentials) that should be deleted.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "ssh_exec",
        "description": "Run a non-interactive command over SSH (no tmux) and return output.",
        "inputSchema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

TMUX_ONLY_TOOLS = [
    tool for tool in TOOLS
    if tool["name"].startswith("tmux_")
]


SSH_ONLY_TOOLS = {"sftp_list", "sftp_read", "sftp_write", "sftp_patch", "sftp_stat", "sftp_mkdir", "sftp_rm", "sftp_rename", "ssh_exec"}


def handle_call(client: Union[SSHClient, LocalClient], name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(client, LocalClient) and name in SSH_ONLY_TOOLS:
        raise MCPError(f"tool not available in local mode: {name}")
    if name == "sftp_list":
        return client.list_dir(args["path"])
    if name == "sftp_read":
        return client.read_file(args["path"], args.get("offset"), args.get("length"))
    if name == "sftp_write":
        return client.write_file(args["path"], args["content"])
    if name == "sftp_patch":
        backup = bool(args.get("backup", False))
        backup_suffix = str(args.get("backup_suffix", ".bak"))
        return client.patch_file(args["path"], args["patch"], backup, backup_suffix)
    if name == "sftp_stat":
        return client.stat(args["path"])
    if name == "sftp_mkdir":
        return client.mkdir(args["path"], bool(args.get("parents")))
    if name == "sftp_rm":
        return client.rm(args["path"], bool(args.get("recursive")))
    if name == "sftp_rename":
        return client.rename(args["src"], args["dst"])
    if name == "tmux_capture":
        return client.tmux_capture(args.get("session"), args.get("lines"))
    if name == "tmux_capture_scrollback":
        return client.tmux_capture_scrollback(args.get("session"), args.get("max_lines"))
    if name == "tmux_send":
        return client.tmux_send(args.get("session"), args["keys"], bool(args.get("enter", True)))
    if name == "tmux_run":
        timeout_ms = args.get("timeout_ms")
        if timeout_ms is not None:
            timeout_ms = int(timeout_ms)
        return client.tmux_run(args.get("session"), args["command"], timeout_ms)
    if name == "tmux_stream_read":
        max_bytes = int(args.get("max_bytes", 8192))
        reset = bool(args.get("reset", False))
        return client.tmux_stream_read(args.get("session"), max_bytes, reset)
    if name == "tmux_cwd":
        return client.tmux_cwd(args.get("session"))
    if name == "tmux_log_purge":
        return client.tmux_log_purge()
    if name == "ssh_exec":
        return client.ssh_exec(args["command"])
    raise MCPError(f"Unknown tool: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SSH/tmux MCP server (stdio).")
    parser.add_argument("--host", default=None, help="user@host for SSH (optional if socket info exists)")
    parser.add_argument("--socket", default=None, help="path to SSH control socket")
    parser.add_argument("--tmux-session", default=None, help="tmux session name")
    parser.add_argument("--root", default=None, help="optional root path restriction")
    parser.add_argument("--local", action="store_true", help="run in local tmux-only mode")
    parser.add_argument("--log-path", default=None, help="log path for local tmux capture")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"SSH/command timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--max-log-size", type=int, default=DEFAULT_MAX_LOG_BYTES, help=f"max tmux log size in bytes before truncation (default: {DEFAULT_MAX_LOG_BYTES})")
    parser.add_argument("--prompt-timeout-ms", type=int, default=30000, help="default timeout for tmux_run in milliseconds (default: 30000)")
    args = parser.parse_args()

    _log(
        "starting server with args: "
        f"host={args.host} socket={args.socket} tmux={args.tmux_session} root={args.root} "
        f"local={args.local} log_path={args.log_path} timeout={args.timeout}"
    )
    host = args.host
    tmux_session = args.tmux_session
    if tmux_session is None:
        tmux_session = "shared-ai"

    if not args.local and not args.socket:
        raise SystemExit("--socket is required unless --local is set")

    _log(f"resolved session: host={host} tmux={tmux_session}")
    if args.local:
        client = LocalClient(tmux_session, args.log_path, timeout=args.timeout, max_log_bytes=args.max_log_size, prompt_timeout_ms=args.prompt_timeout_ms)
        tools = TMUX_ONLY_TOOLS
        # Ensure local tmux session exists and pipe-pane is enabled.
        code, _out, _err = _run_cmd(["tmux", "has-session", "-t", tmux_session])
        if code != 0:
            _run_cmd(["tmux", "new-session", "-d", "-s", tmux_session])
        os.makedirs(os.path.dirname(client.log_path), exist_ok=True)
        with open(client.log_path, "a", encoding="utf-8"):
            pass
        # Close any existing pipe first (pipe-pane toggles, so a restart would turn it off)
        _run_cmd(["tmux", "pipe-pane", "-t", tmux_session])
        _run_cmd(["tmux", "pipe-pane", "-o", "-t", tmux_session, f"cat >> {client.log_path}"])
        # Truncate log on startup if oversized
        try:
            size = os.path.getsize(client.log_path)
            if size > args.max_log_size:
                _log(f"startup: log {client.log_path} is {size} bytes, truncating")
                with open(client.log_path, "w"):
                    pass
        except OSError:
            pass
    else:
        client = SSHClient(host, args.socket, args.root, tmux_session, timeout=args.timeout, max_log_bytes=args.max_log_size, prompt_timeout_ms=args.prompt_timeout_ms)
        tools = TOOLS
        # Avoid SSH calls on startup so initialize can complete quickly.

    while True:
        try:
            msg = read_message()
        except ValueError as exc:
            _log(f"protocol error: {exc}")
            print(f"Protocol error: {exc}", file=sys.stderr)
            break
        if msg is None:
            _log("read_message: EOF")
            break
        if "method" not in msg:
            _log(f"message missing method: {msg!r}")
            continue
        method = msg.get("method")
        msg_id = msg.get("id")
        _log(f"dispatch method={method} id={msg_id}")

        if method == "initialize":
            params = msg.get("params", {}) if isinstance(msg.get("params", {}), dict) else {}
            negotiated = _negotiate_protocol(params.get("protocolVersion"))
            result = {
                "protocolVersion": negotiated,
                "serverInfo": {"name": "ssh-tmux-mcp", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            }
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": result})
            continue

        if method in ("listTools", "tools/list"):
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}})
            continue

        if method in ("callTool", "tools/call"):
            params = msg.get("params", {})
            name = params.get("name")
            args_in = params.get("arguments", {})
            try:
                output = handle_call(client, name, args_in)
                send_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(output, ensure_ascii=True)}]},
                })
            except ControlMasterGone as e:
                send_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32000, "message": e.message},
                })
            except MCPError as e:
                send_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32000, "message": e.message},
                })
            continue

        if method in ("resources/list", "resources/templates/list"):
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": {"resources": []}})
            continue

        if method in ("prompts/list",):
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": {"prompts": []}})
            continue

        if method == "ping":
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": {}})
            continue

        if method == "shutdown":
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": None})
            break

        if method in ("notifications/initialized", "initialized"):
            _log("received initialized notification")
            continue

        # Other notifications can be ignored


if __name__ == "__main__":
    main()
