import os
from typing import NamedTuple


class CometCredentials(NamedTuple):
    api_key: str | None
    project_name: str | None
    workspace: str | None


def _load_private_tokens():
    try:
        import private_tokens
    except ImportError:
        return None
    return private_tokens


def get_comet_credentials(
    project_name: str | None = None,
    workspace: str | None = None,
) -> CometCredentials:
    tokens = _load_private_tokens()

    api_key = None
    resolved_project = project_name
    resolved_workspace = workspace

    if tokens is not None:
        api_key = getattr(tokens, "COMET_API_KEY", None) or None
        if not resolved_project:
            resolved_project = getattr(tokens, "COMET_PROJECT_NAME", None) or None
        if not resolved_workspace:
            resolved_workspace = getattr(tokens, "COMET_WORKSPACE", None) or None

    if not api_key:
        api_key = os.environ.get("COMET_API_KEY") or None
    if not resolved_project:
        resolved_project = os.environ.get("COMET_PROJECT_NAME") or None
    if not resolved_workspace:
        resolved_workspace = os.environ.get("COMET_WORKSPACE") or None

    return CometCredentials(
        api_key=api_key,
        project_name=resolved_project,
        workspace=resolved_workspace,
    )
