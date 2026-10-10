import argparse
import json
import sys
from typing import Any

from agent.loop import run_agent
from agent.provider import get_provider
from agent.tracing import init_tracing, shutdown_tracing


def format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Human-readable listing of each tool call's name, args and result."""
    if not tool_calls:
        return "Tool calls: none"
    lines = [f"Tool calls ({len(tool_calls)}):"]
    for i, tc in enumerate(tool_calls, 1):
        args = ", ".join(f"{k}={json.dumps(v)}" for k, v in tc["args"].items())
        lines.append(f"  {i}. {tc['name']}({args})")
        result = json.dumps(tc["result"], indent=2).replace("\n", "\n       ")
        lines.append(f"     -> {result}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent")
    parser.add_argument("message", help="customer message")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="also print each tool call and its result (to stderr) after the reply")
    try:
        args = parser.parse_args(argv[1:])
    except SystemExit as e:
        return int(e.code or 0)
    init_tracing()  # before the client is created so it can be wrapped
    try:
        result = run_agent(args.message, get_provider())
        print(json.dumps(result.to_dict(), indent=2), flush=True)
        if args.verbose:
            # stderr so stdout stays parseable JSON
            print(format_tool_calls(result.tool_calls), file=sys.stderr)
        return 0
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
