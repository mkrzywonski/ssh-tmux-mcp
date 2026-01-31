# ssh-tmux-mcp

Local MCP server that uses an existing SSH ControlMaster socket to provide:
- Remote file access (list/read/write/stat/mkdir/rm/rename)
- Shared terminal access via tmux (capture/send/cwd)

This keeps authentication on your side. The MCP server only works while the SSH
control socket exists.

## Requirements
- Linux (NixOS/WSL fine)
- `ssh` on local machine
- `tmux` on the remote host
- Coreutils on the remote host (`base64`, `find`, `stat`, `dd`, `mkdir`, `rm`, `mv`)

## Start a session

1) Start the SSH ControlMaster and tmux session:

```bash
./ai-ssh user@host
```

2) Configure Codex to start the MCP server (example). The socket lives under
`/run/user/<uid>/ai-ssh/ssh-remote.sock` and `ai-ssh` writes a `.info` file so
the server can infer `host` and `tmux`:

```toml
[mcp_servers.ssh_tmux]
command = "python3"
args = ["/home/mike/ai/ssh-tmux-mcp/mcp_server.py", "--socket", "/run/user/1000/ai-ssh/ssh-remote.sock"]
```

3) Restart Codex so it starts the MCP server, then use the `ssh_tmux` tools.

When you exit the tmux session, the ControlMaster is closed automatically.

Helper:
- `./tmux-capture-recent [lines]` prints the last N lines from the shared tmux session.

## Local mode (tmux-only)

If you want the same collaborative tmux workflow locally (no SSH), run the server
with `--local`. It will create the tmux session if missing and enable a log pipe.

Example config:

```toml
[mcp_servers.tmux_local]
command = "python3"
args = ["/home/mike/ai/ssh-tmux-mcp/mcp_server.py", "--local", "--tmux-session", "shared-ai"]
```

## Tools

- `sftp_list` { path }
- `sftp_read` { path, offset?, length? } -> base64 content
- `sftp_write` { path, content } (base64)
- `sftp_patch` { path, patch, backup?, backup_suffix? }
- `sftp_stat` { path }
- `sftp_mkdir` { path, parents? }
- `sftp_rm` { path, recursive? }
- `sftp_rename` { src, dst }
- `tmux_capture` { session?, lines? }
- `tmux_capture_scrollback` { session?, max_lines? }
- `tmux_send` { session?, keys, enter? }
- `tmux_run` { session?, command, lines?, delay_ms? } -> recent output
- `tmux_stream_read` { session?, max_bytes?, reset? } -> new output since last read
- `tmux_run_stream` { session?, command, max_bytes?, delay_ms? } -> new output from log
- `tmux_run_sentinel` { session?, command, max_bytes?, delay_ms?, timeout_ms?, poll_ms? } -> output up to sentinel + exit code
- `tmux_cwd` { session? }
- `ssh_exec` { command } -> output (non-interactive, no tmux)

## Notes

- `sftp_*` tools are implemented via SSH commands (coreutils), not the SFTP subsystem.
- `sftp_read`/`sftp_write` use base64 to be binary-safe.
- `sftp_patch` applies unified diffs locally, then writes back; it does not require `patch` on the remote.
- If you want to restrict access, pass `--root /path` to constrain file ops.
- Tmux log output is capped to `--max-log-size` bytes; older output is truncated to keep the last N bytes.

## Workflow Recommendations

- Prefer `sftp_*` tools for file edits (read/modify/write) to avoid shell quoting issues.
- Use tmux tools for commands, build output, and interactive steps (sudo prompts, long-running tasks).
- Use `ssh_exec` for quick, non-interactive commands when you don't need shared tmux visibility.
