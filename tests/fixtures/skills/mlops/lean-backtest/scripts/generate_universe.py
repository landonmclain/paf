#!/usr/bin/env python3
"""Regenerate the MES future-contract universe for the backtest window."""


def universe(window_start: str, window_end: str) -> list[str]:
    # Truncated for the fixture — real impl pulls from the local future DB.
    return ["MESZ26", "MESH27"]


if __name__ == "__main__":
    print("\n".join(universe("2026-01-01", "2026-03-01")))
