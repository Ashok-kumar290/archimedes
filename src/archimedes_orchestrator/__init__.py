"""Archimedes orchestrator: the integration layer that turns isolated expert
lobes into a mind.

v1 scope — the *decomposer* and *executor*: break a multi-step problem into
single-operation sub-questions, each solved by one lobe forward pass, chaining
the outputs. The lobe does all arithmetic; the orchestrator only decides which
sub-questions to ask and in what order.
"""
from .decomposer import Plan, Ref, Step, decompose, execute

__all__ = ["Plan", "Ref", "Step", "decompose", "execute"]
