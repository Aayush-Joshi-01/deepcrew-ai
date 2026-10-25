"""Run the memory example scripts end to end against a real model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).parent.parent.parent / "examples"

pytestmark = pytest.mark.live(
    feature="examples_memory",
    watch=[
        "examples/episodic_memory.py",
        "examples/procedural_memory.py",
        "src/deepcrew/runner.py",
        "src/deepcrew/episodic_memory.py",
        "src/deepcrew/procedural_memory.py",
    ],
)


def _load(name: str):
    path = _EXAMPLES / name
    spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["episodic_memory.py", "procedural_memory.py"])
@pytest.mark.asyncio
async def test_example_runs(name: str, capsys):
    module = _load(name)
    await module.main()
    out = capsys.readouterr().out
    assert out.strip(), f"{name} produced no output"
