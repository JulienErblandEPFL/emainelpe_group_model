"""
Group-model adapter merging.

This subpackage takes the four locked-spec LoRA adapters (math, general_knowledge,
multilingual, safety) and composes them into a single group adapter for the
CS-552 leaderboard. It reads the team-locked spec from ``../lora.yaml`` and
the chat template from ``../chat_template.jinja`` — never duplicate those
values inside this package.

This is not training code. It is not teammate-facing. Group-merge work is
Julien's responsibility for team Émainèlpé (g65).
"""

__version__ = "0.1.0-skeleton"
