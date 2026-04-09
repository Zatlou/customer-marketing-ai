from __future__ import annotations

import json
from pathlib import Path


def project_root() -> Path:
    """Locate the repository root from either the root or src/ directory."""
    candidates = [Path.cwd(), Path.cwd().parent]
    for candidate in candidates:
        if (candidate / "data" / "raw" / "marketing_campaign.csv").exists():
            return candidate.resolve()

    return Path.cwd().resolve()


def resolve_path(path_like: str | Path) -> Path:
    """Resolve project-relative paths declared in params.yaml."""
    path = Path(path_like)
    if path.is_absolute():
        return path

    return project_root() / path


def load_params(path: str | Path | None = None) -> dict:
    """Load params from params.yaml.

    The params file is stored as JSON-compatible YAML so it can be parsed
    with the standard library when PyYAML is not installed.
    """
    params_path = resolve_path(path or "params.yaml")
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found at: {params_path}")

    raw_text = params_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return json.loads(raw_text)

    return yaml.safe_load(raw_text)
