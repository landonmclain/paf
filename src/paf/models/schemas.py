"""Shared Pydantic schemas — the typed contract layer.

One file for M1; split by domain as it grows (not internal-vs-domain).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Greeting(BaseModel):
    """Structured output of the example agent."""

    message: str = Field(description="The greeting message produced by the agent.")
    to: str = Field(description="The name of the person greeted.")
    polite: bool = Field(description="Whether the message is polite/kind.")
