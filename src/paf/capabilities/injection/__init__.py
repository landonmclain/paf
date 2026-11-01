"""paf-owned prompt-injection defense (``InjectionGuard``).

Rehomed from the harness ``PromptInjectionDefender`` so paf owns the decision
policy. The classifier is a swappable *scorer* (``_scorer.py``); what happens to
a result it flags is the *policy* (``_policy.py``), declared per tool in the spec;
the capability (``_capability.py``) applies one to the other and traces every
decision.

The implemented actions are ``off``, ``observe`` and ``block``. ``flag`` and
``redact``, marking untrusted content, and tightening gating after flagged
content are reserved names: the policy parser rejects them with a clear error
until they are implemented, so a spec cannot silently ask for protection it does
not get.
"""

from paf.capabilities.injection._capability import InjectionGuard
from paf.capabilities.injection._policy import (
    ACTIONS,
    IMPLEMENTED_ACTIONS,
    InjectionPolicy,
    parse_injection_policy,
)
from paf.capabilities.injection._scorer import InjectionScorer, Score, StackoneScorer

__all__ = [
    "ACTIONS",
    "IMPLEMENTED_ACTIONS",
    "InjectionGuard",
    "InjectionPolicy",
    "InjectionScorer",
    "Score",
    "StackoneScorer",
    "parse_injection_policy",
]
