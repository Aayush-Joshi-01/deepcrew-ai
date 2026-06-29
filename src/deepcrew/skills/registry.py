from __future__ import annotations

from typing import ClassVar

from .base import Skill


class SkillRegistry:
    """Class-level registry for named skills."""

    _registry: ClassVar[dict[str, Skill]] = {}

    @classmethod
    def register(cls, skill: Skill) -> None:
        cls._registry[skill.name] = skill

    @classmethod
    def get(cls, name: str) -> Skill | None:
        return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> list[Skill]:
        return list(cls._registry.values())

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
