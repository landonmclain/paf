---
name: lean-backtest
description: Run a vectorized LEAN backtest over local minute bars and pull the resulting performance metrics.
tags: [lean, backtest, quantconnect, futures, mes]
---

# Skill: lean-backtest

Run a local LEAN backtest against a published strategy and pull its performance
artifact. Load the mesh/reference notes before wiring up a full run.

Steps:
1. Load the reference on bar-level debug logging (see `references/`).
2. Use the helper script to regenerate the future universe (see `scripts/`).
3. Execute the backtest through the LEAN CLI local workflow.
