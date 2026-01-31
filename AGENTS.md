# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Bash + Python toolset:
- `ai-ssh`: Shell entrypoint that establishes the SSH ControlMaster and tmux session.
- `mcp_server.py`: MCP server that exposes filesystem + tmux tools over the SSH socket.
- `README.md`: Usage notes and setup steps.
There are no separate test or asset directories.

## Build, Test, and Development Commands
No build step is required.
- Run the session setup: `./ai-ssh user@host` (creates the SSH ControlMaster and tmux session).
- Run the MCP server directly (if needed):  
  `python3 /home/mike/ai/ssh-tmux-mcp/mcp_server.py --host user@host --socket /tmp/ssh-remote.sock --tmux-session shared-ai`
- Lint/format: no tooling configured. Keep changes minimal and readable.

## Coding Style & Naming Conventions
- Bash: POSIX-ish with `set -euo pipefail`, double-quoted variables, `snake_case` function names.
- Python: standard library only; keep functions small and explicit. Use `snake_case`.
- Keep output messages short and actionable; prefer `Error: ...` for failures.

## Testing Guidelines
There is no automated test suite. Validate changes manually:
- Start a session via `./ai-ssh user@host`.
- Confirm tmux attaches and MCP tools can read/write within the remote host.
- End the session by exiting tmux and verify the socket is gone.

## Workflow Tips
- Prefer `sftp_*` tools for file edits (read/modify/write) to avoid shell quoting issues.
- Use tmux tools for commands, build output, and interactive steps (sudo prompts, long-running tasks).
- Use `sftp_patch` for unified diffs when possible; fall back to read/modify/write if patching fails.

## Commit & Pull Request Guidelines
No strict commit message convention is documented. Use clear, imperative messages (e.g., "Harden path checks").  
PRs should include:
- A short summary of changes.
- Manual verification steps.
- Any security or behavior impacts.

## Security & Configuration Tips
- The MCP server trusts the SSH ControlMaster socket; it should only be used for intended sessions.
- If you change socket or tmux defaults, update both `ai-ssh` and your Codex MCP server config.
- Tmux log output is capped to `--max-log-size` bytes; older output is truncated to keep the last N bytes.
