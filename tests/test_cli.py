from __future__ import annotations

import sys

import pytest

from deepcrew.cli.main import main


def test_version_exits_zero_and_prints_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["deepcrew", "--version"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "deepcrew-ai" in captured.out


def test_help_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["deepcrew", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()


def test_no_command_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["deepcrew"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


def test_invalid_subcommand_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["deepcrew", "bogus"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


def test_agents_list_missing_config_flag_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["deepcrew", "agents", "list"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


def test_agents_list_prints_table(monkeypatch, capsys, tmp_path):
    config = tmp_path / "workflow.yaml"
    config.write_text(
        "agents:\n  - name: bot\n    model: openai/gpt-4o-mini\nworkflow: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["deepcrew", "agents", "list", "--config", str(config)])
    main()
    captured = capsys.readouterr()
    assert "bot" in captured.out
    assert "openai/gpt-4o-mini" in captured.out
