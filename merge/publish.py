"""
Upload the merged group adapter to ``cs-552-2026-emainelpe/group_model``.

Wraps ``huggingface_hub.HfApi.upload_folder`` with the project-specific
guardrails:

- The repo ID must match the leaderboard contract exactly (CI looks up models
  by slug — typos forfeit the milestone).
- ``adapter_config.json``, ``adapter_model.safetensors``,
  ``chat_template.jinja``, and ``generation_config.json`` must all be present
  in ``adapter_dir`` before upload.
- ``HF_TOKEN`` must be readable from the env if no token is passed.

To be implemented in Stage 5.
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi  # noqa: F401  (used by future implementation)


def publish_adapter(
    adapter_dir: Path,
    repo_id: str,
    *,
    hf_token: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    create_if_missing: bool = True,
) -> str:
    """
    Upload an adapter directory to HF Hub. Returns the commit URL.

    Args:
        adapter_dir: Path to a complete, PEFT-loadable adapter directory.
        repo_id: Target HF repo, expected to be
            ``"cs-552-2026-emainelpe/group_model"`` for the leaderboard.
        hf_token: HF write-token. If None, falls back to ``$HF_TOKEN``.
        private: Whether to create the repo as private. The CS-552 CI
            requires public repos — keep False unless testing.
        commit_message: Optional override. Default includes a UTC timestamp
            and the merge method name.
        create_if_missing: If True, creates the repo when it doesn't exist.

    Returns:
        Commit URL on HF Hub (the value ``HfApi.upload_folder`` returns).

    Raises:
        FileNotFoundError: if a required artifact is missing from ``adapter_dir``.
        huggingface_hub.errors.HfHubHTTPError: on auth/permission errors.
        ValueError: if ``private=True`` and ``repo_id`` looks like the
            leaderboard slug (defensive guard against accidental private push).
    """
    raise NotImplementedError("Stage 5")
