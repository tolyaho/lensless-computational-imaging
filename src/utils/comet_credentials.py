import os
from typing import NamedTuple


class CometCredentials(NamedTuple):
    api_key: str | None
    project_name: str | None
    workspace: str | None


def get_comet_credentials(
    project_name: str | None = None,
    workspace: str | None = None,
) -> CometCredentials:
    api_key = os.environ.get("COMET_API_KEY") or None
    resolved_project = project_name or os.environ.get("COMET_PROJECT_NAME") or None
    resolved_workspace = workspace or os.environ.get("COMET_WORKSPACE") or None

    if not api_key:
        try:
            import private_tokens

            api_key = getattr(private_tokens, "COMET_API_KEY", None) or None
            if not resolved_project:
                resolved_project = (
                    getattr(private_tokens, "COMET_PROJECT_NAME", None) or None
                )
            if not resolved_workspace:
                resolved_workspace = (
                    getattr(private_tokens, "COMET_WORKSPACE", None) or None
                )
        except ImportError:
            pass

    return CometCredentials(
        api_key=api_key,
        project_name=resolved_project,
        workspace=resolved_workspace,
    )
