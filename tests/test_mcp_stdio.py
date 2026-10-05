"""Protocol-level test: start doceval-mcp and talk to it over stdio.

test_mcp_server.py calls the tool functions directly, which proves the scoring
but not that the server starts. This drives the real process with raw JSON-RPC,
which is the same on every MCP SDK version, so CI can run it against the oldest
supported release and the newest one.
"""

import json
import subprocess
import sys

import pytest

pytest.importorskip("mcp")

# The oldest protocol revision; servers on any SDK version still accept it.
PROTOCOL_VERSION = "2024-11-05"


class _Session:
    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "doceval.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.next_id = 1

    def send(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            msg["id"] = self.next_id
            self.next_id += 1
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("server exited: " + self.proc.stderr.read())
            reply = json.loads(line)
            if reply.get("id") == msg["id"]:
                return reply

    def close(self):
        self.proc.stdin.close()
        self.proc.terminate()
        self.proc.wait(timeout=10)


@pytest.fixture
def session():
    s = _Session()
    reply = s.send(
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "doceval-test", "version": "0"},
        },
    )
    assert "result" in reply, reply
    s.send("notifications/initialized", notify=True)
    yield s
    s.close()


def test_server_lists_both_tools(session):
    reply = session.send("tools/list", {})
    names = {tool["name"] for tool in reply["result"]["tools"]}
    assert names == {"score_extraction", "run_eval"}


def test_server_scores_over_the_protocol(session):
    reply = session.send(
        "tools/call",
        {
            "name": "score_extraction",
            "arguments": {
                "expected": {"vendor": "Acme Corp", "total": "100.00"},
                "actual": {"vendor": "Acme Corp", "total": "999.00"},
            },
        },
    )
    result = reply["result"]
    assert not result.get("isError"), result
    payload = json.loads(result["content"][0]["text"])
    assert payload["fields_correct"] == 1
    assert payload["fields_total"] == 2
