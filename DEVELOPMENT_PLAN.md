# Development Plan

Tracked improvements for ssh-tmux-mcp, ordered by priority.

## Issue 5: No timeout on SSH subprocess calls

**Status:** Done

**Description:** `_run_cmd` calls `subprocess.run` with no `timeout` parameter. A hung SSH connection will block the MCP server indefinitely, making it unresponsive to the client.

**Changes:**
- `_run_cmd` accepts `timeout` param (default `DEFAULT_TIMEOUT = 30`s), catches `subprocess.TimeoutExpired` and raises `MCPError`.
- `SSHClient` and `LocalClient` accept `timeout` in constructors, propagate to all subprocess calls.
- `SSHClient.ssh()` accepts per-call `timeout` override.
- `_check_socket` uses hardcoded 10s timeout for quick health checks.
- `--timeout` CLI arg configures the default at startup.
- Tested against both local tmux and live SSH session. Server recovers cleanly after a timeout.

---

## Issue 8: write_file is vulnerable to partial writes

**Status:** Done

**Description:** `write_file` runs `base64 -d > file`, which truncates the target immediately. If the SSH connection drops mid-transfer, the file is left corrupted or empty with no way to recover.

**Changes:**
- `write_file` now uses `mktemp -p <dir>` to create a temp file in the same directory as the target, writes to it via `base64 -d`, then atomically `mv`s it into place.
- `patch_file` inherits the fix since it calls `write_file` internally.
- Tested over live SSH (15/15 pass): basic write, overwrite, nested paths, empty files, binary content, patch integration, no stale temp files, and clean error on nonexistent directory.

---

## Issue 6: Unnecessary SSH roundtrips for tmux operations

**Status:** Done

**Description:** `tmux_send` makes two SSH calls when `enter=True` (one for keys, one for `C-m`). `tmux_run` makes three. `tmux_run_sentinel` makes four. Each roundtrip adds latency over the network.

**Changes:**
- `tmux_send`: combined keys + `C-m` into single `send-keys` call (2 → 1 SSH call).
- `tmux_run`: combined send-keys, delay, and capture into single SSH command (3 → 1 SSH call).
- `tmux_run_stream`: combined send-keys + enter (2 → 1 SSH call).
- `tmux_run_sentinel`: combined command + enter + marker + enter (4 → 1 SSH call, plus 1 for log offset).
- Tested over live SSH (13/13 pass): send with/without enter, run, run_stream, sentinel with success and failure exit codes.

---

## Issue 2: handle_call doesn't work with LocalClient

**Status:** Done

**Description:** `handle_call` is typed to accept `SSHClient` but `main()` passes `LocalClient` in local mode. If a client sends an `sftp_*` or `ssh_exec` tool call in local mode, it crashes with `AttributeError` instead of returning a clean error.

**Changes:**
- Added `SSH_ONLY_TOOLS` set listing all sftp/ssh_exec tool names.
- `handle_call` now checks `isinstance(client, LocalClient)` and raises `MCPError("tool not available in local mode: ...")` for SSH-only tools.
- Updated type annotation to `Union[SSHClient, LocalClient]`.
- Tested in local mode (15/15 pass): all 9 SSH-only tools return clean errors, tmux tools still work, and `tools/list` correctly omits SSH-only tools.

---

## Issue 1: Code duplication between SSHClient and LocalClient

**Status:** Done

**Description:** `tmux_run_sentinel` (~90 lines), `_find_sentinel_line`, `tmux_run_stream`, `tmux_stream_read`, and `_read_stream_from_offset` are nearly identical in both classes. This makes bug fixes and feature changes error-prone.

**Changes:**
- Extracted `_find_sentinel_line` to a module-level function.
- Created `TmuxClientBase` with all shared tmux methods: `tmux_capture`, `tmux_capture_scrollback`, `tmux_send`, `tmux_run`, `tmux_stream_read`, `tmux_run_stream`, `tmux_run_sentinel`, `tmux_cwd`.
- Subclasses implement 4 primitives: `_run_tmux`, `_read_stream_from_offset`, `_get_log_path`, `_ensure_tmux_session`.
- `SSHClient` overrides `tmux_run` and `_send_keys_with_sentinel` to combine multiple tmux commands into single SSH calls (preserving Issue 6 optimization).
- `LocalClient._run_tmux` uses `bash -c` to preserve shell quoting of the full tmux command string.
- Net reduction of 75 lines despite all other feature additions.
- All 46 tests pass across SSH, local, atomic write, and timeout test suites — no regressions.

---

## Issue 4: dd bs=1 is very slow for stream reads over SSH

**Status:** Done

**Description:** `_read_stream_from_offset` in `SSHClient` uses `dd if=... bs=1 skip=N count=M` which reads one byte at a time. This is extremely slow for larger reads over SSH.

**Changes:**
- Replaced `dd bs=1 skip=N count=M` with `tail -c +N | head -c M` in both `_read_stream_from_offset` and `read_file`.
- GNU coreutils `tail -c +N` is 1-indexed, so offset is passed as `offset + 1`.
- Tested over live SSH (33/33 pass): full reads, partial reads with offset/length, offset past EOF, binary content, stream reads, and sentinel polling.

---

## Issue 3: _normalize_path uses os.path for remote paths

**Status:** Done

**Description:** `_normalize_path` uses `os.path.normpath` and `os.path.commonpath`, which follow the local OS conventions. If the MCP server runs on Windows targeting a Linux remote, path handling would break.

**Changes:**
- Replaced `os.path.normpath`, `os.path.join`, `os.path.commonpath` with `posixpath` equivalents in both `__init__` (root normalization) and `_normalize_path`.
- Verified posixpath correctly handles normal paths, traversal attempts, and leading slashes.
- All existing tests pass — no regressions.

---

## Issue 9: Sentinel token not fully cleaned from output

**Status:** Done

**Description:** The sentinel cleanup only removes lines containing both `token` and `printf`. If the shell echoes the command differently (e.g. zsh right-prompt, multiline PS1), the printf marker line could leak into the returned output.

**Changes:**
- Simplified the cleanup filter: now removes any line containing the token string, regardless of whether `printf` also appears.
- Tested over live SSH (11/11 pass): simple commands, multi-line output, and special characters all produce clean output with no sentinel leakage.

---

## Issue 7: Log file grows without bound

**Status:** Done

**Description:** Both local and SSH modes append to `tmux.log` via `pipe-pane` with no rotation or truncation. Long-running sessions will consume arbitrary disk space.

**Changes:**
- Implemented truncation that keeps the last `max_log_size` bytes for both SSH and local modes.
- Truncation is applied lazily during tmux log reads/stream operations (no SSH on startup).
- Updated docs to mention `--max-log-size` behavior.

**Testing:**
- Local: created a 100-byte log with `max_log_size=32`, truncated to the last 32 bytes, verified contents match tail.
- SSH: manual validation pending (run a long log session, confirm size caps and tail is preserved).

---

## Issue 10: ai-ssh cleanup tears down ControlMaster on normal detach

**Status:** Not planned

**Description:** `cleanup` runs unconditionally at the end of `ai-ssh`, which kills the ControlMaster after detaching from tmux. This means the MCP server loses its SSH connection whenever the user detaches.

**Decision:** Keep current behavior. Exiting tmux should tear down the ControlMaster and session cleanly.

---

## Issue 11: Missing notifications/initialized handling

**Status:** Pending

**Description:** The MCP spec expects the server to accept `notifications/initialized` after the initialize handshake. The server silently drops it, which works but makes debugging harder.

**Plan:**
- Add an explicit case for `notifications/initialized` (and the bare `initialized` variant) that logs receipt and continues.
- No response is needed since it's a notification, not a request.

---

## Issue 12: Protocol version is pinned

**Status:** Pending

**Description:** The server always responds with `protocolVersion: "2024-11-05"` regardless of what the client requests. As the MCP spec evolves, this could cause compatibility issues.

**Plan:**
- Read the client's requested `protocolVersion` from the `initialize` params.
- Respond with the minimum of the client's requested version and the server's supported version.
- For now, keep `2024-11-05` as the maximum supported version and update as needed.
