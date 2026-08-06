"""Minimal one-turn entrypoint — use the runtime Session, run once, print.

The Session owns build + tracing + stream + flush, so this stops re-implementing
``setup_observability() → build_agent() → agent.run() → shutdown_observability()``
by hand. A ``with`` block guarantees spans flush before the short-lived process
exits (a bare ``run_sync`` can drop them).
"""

from __future__ import annotations

from paf.runtime import Session


def main() -> None:
    with Session("assistant") as s:
        print(f"Agent: {s.name}", flush=True)
        result = s.run_sync("Say hello in one sentence.")
        print(f"\n{result}")


if __name__ == "__main__":
    main()
