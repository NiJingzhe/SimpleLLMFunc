from __future__ import annotations

from pathlib import Path

import pytest

from SimpleLLMFunc.skills_cli import export_skill, main


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def fake_skills_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    skills_root = tmp_path / "skills-src"
    _write(skills_root / "simplellmfunc" / "SKILL.md", "usage skill")
    _write(
        skills_root / "simplellmfunc" / "reference" / "configuration.md",
        "usage config",
    )
    _write(skills_root / "simplellmfunc-developer" / "SKILL.md", "developer skill")

    monkeypatch.setattr(
        "SimpleLLMFunc.skills_cli._resolve_packaged_skills_root",
        lambda: skills_root,
    )
    return skills_root


def test_export_usage_skill_creates_expected_folder(
    fake_skills_root: Path, tmp_path: Path
) -> None:
    destination = export_skill("usage", tmp_path)

    assert destination == tmp_path / "simplellmfunc"
    assert (destination / "SKILL.md").read_text(encoding="utf-8") == "usage skill"
    assert (destination / "reference" / "configuration.md").exists()


def test_export_developer_skill_uses_expected_name(
    fake_skills_root: Path,
    tmp_path: Path,
) -> None:
    destination = export_skill("developer", tmp_path)

    assert destination == tmp_path / "simplellmfunc-developer"
    assert (destination / "SKILL.md").read_text(encoding="utf-8") == "developer skill"


def test_export_supports_useage_alias(fake_skills_root: Path, tmp_path: Path) -> None:
    destination = export_skill("useage", tmp_path)

    assert destination == tmp_path / "simplellmfunc"


def test_export_requires_force_when_destination_exists(
    fake_skills_root: Path,
    tmp_path: Path,
) -> None:
    export_skill("usage", tmp_path)

    with pytest.raises(FileExistsError):
        export_skill("usage", tmp_path)

    destination = export_skill("usage", tmp_path, force=True)
    assert destination == tmp_path / "simplellmfunc"


def test_main_prints_exported_path(
    fake_skills_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["usage", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(tmp_path / "simplellmfunc") in captured.out
