---
name: vectorized-backtest
description: Build a standalone Python vectorized backtesting framework to evaluate a trading strategy.
tags: [backtest, trading, strategy, vectorized, pandas]
---

# Skill: vectorized-backtest

Build a fast, vectorized backtest for a trading strategy.

1. Load the price/bar data (CSV or the data store).
2. Compute the strategy's signal vector over the full series.
3. Derive positions and PnL using vectorized numpy/pandas ops — no per-bar loop.
4. Return summary stats: total return, Sharpe, max drawdown, win rate.

Keep the framework dependency-free where possible; prefer pure pandas/numpy.
