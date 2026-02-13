from pathlib import Path

from zenml import step


def _candidate_model_dirs() -> list[Path]:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[6] if len(this_file.parents) > 6 else this_file.parent
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "trained_models" / "groupclassifier",
        repo_root / "trained_models" / "groupclassifier",
        Path("/workspace/trained_models/groupclassifier"),
    ]
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


@step(enable_cache=False)
def load_group_classifier_model(
    model_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    cfg = {}
    if isinstance(pipeline_config, dict) and isinstance(pipeline_config.get("model_loader"), dict):
        cfg = dict(pipeline_config["model_loader"])

    selected = model_path or cfg.get("model_path")
    if selected is None:
        candidates: list[Path] = []
        searched: list[str] = []
        for model_dir in _candidate_model_dirs():
            searched.append(str(model_dir))
            if not model_dir.exists():
                continue
            candidates.extend(model_dir.glob("*_torchscript.pt"))
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                "No torchscript models found. Searched directories: "
                + ", ".join(searched)
            )
        resolved = candidates[0].resolve()
    else:
        resolved = Path(selected).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model not found: {resolved}")

    return {"model_path": str(resolved)}
