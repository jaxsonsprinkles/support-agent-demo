import json
import sys

from agent.loop import run_agent
from agent.provider import get_provider
from agent.tracing import init_tracing, shutdown_tracing


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('usage: python -m agent "customer message"', file=sys.stderr)
        return 2
    init_tracing()  # before the client is created so it can be wrapped
    try:
        result = run_agent(argv[1], get_provider())
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
