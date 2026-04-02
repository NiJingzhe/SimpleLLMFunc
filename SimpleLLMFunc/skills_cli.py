from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


SKILL_KIND_TO_FOLDER = {
    "usage": "simplellmfunc",
    "useage": "simplellmfunc",
    "developer": "simplellmfunc-developer",
    "dev": "simplellmfunc-developer",
}


def _resolve_packaged_skills_root() -> Path:
    package_dir = Path(__file__).resolve().parent
    candidates = [
        package_dir.parent / "skills",
        package_dir / "skills",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find packaged skills directory. Reinstall SimpleLLMFunc with packaged skill data."
    )


def normalize_skill_kind(kind: str) -> str:
    normalized = kind.strip().lower()
    if normalized not in SKILL_KIND_TO_FOLDER:
        supported = ", ".join(sorted(SKILL_KIND_TO_FOLDER))
        raise ValueError(
            f"Unsupported skill kind '{kind}'. Expected one of: {supported}"
        )
    return normalized


def export_skill(kind: str, target_root: str | Path, *, force: bool = False) -> Path:
    normalized_kind = normalize_skill_kind(kind)
    skill_folder = SKILL_KIND_TO_FOLDER[normalized_kind]

    skills_root = _resolve_packaged_skills_root()
    source_dir = skills_root / skill_folder
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Packaged skill folder not found: {source_dir}")

    target_root_path = Path(target_root).expanduser()
    target_root_path.mkdir(parents=True, exist_ok=True)
    if not target_root_path.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_root_path}")

    destination_dir = target_root_path / skill_folder
    if destination_dir.exists():
        if not force:
            raise FileExistsError(
                f"Destination already exists: {destination_dir}. Use --force to overwrite."
            )
        if destination_dir.is_dir():
            shutil.rmtree(destination_dir)
        else:
            destination_dir.unlink()

    shutil.copytree(source_dir, destination_dir)
    return destination_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export packaged SimpleLLMFunc Agent Skills to a target directory."
    )
    parser.add_argument(
        "kind",
        help="Skill kind to export: usage/useage or developer/dev",
    )
    parser.add_argument(
        "target_path",
        help="Parent directory that will receive the exported skill folder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination skill folder if it already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        destination = export_skill(args.kind, args.target_path, force=args.force)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
