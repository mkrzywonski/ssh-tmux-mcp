#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from typing import Any, Dict


def send(proc: subprocess.Popen, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    proc.stdin.write(data + b"\n")
    proc.stdin.flush()


def recv(proc: subprocess.Popen) -> Dict[str, Any]:
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("no response from server")
    return json.loads(line.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple MCP client for ssh-tmux-mcp.")
    parser.add_argument("--socket", required=True, help="path to SSH control socket")
    parser.add_argument("--host", help="user@host for SSH")
    parser.add_argument("--tmux-session", help="tmux session name")
    parser.add_argument("--tool", required=True, help="tool name to call")
    parser.add_argument("--args", default="{}", help="JSON arguments for the tool")
    args = parser.parse_args()

    cmd = ["python3", "/home/mike/ai/ssh-tmux-mcp/mcp_server.py", "--socket", args.socket]
    if args.host:
        cmd += ["--host", args.host]
    if args.tmux_session:
        cmd += ["--tmux-session", args.tmux_session]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    recv(proc)
    send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    recv(proc)

    tool_args = json.loads(args.args)
    send(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": args.tool, "arguments": tool_args}})
    reply = recv(proc)
    print(json.dumps(reply, indent=2))

    proc.terminate()


if __name__ == "__main__":
    main()
