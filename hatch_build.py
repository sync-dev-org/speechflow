"""Restrict source distributions to the configured public file selection."""

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        allowed_paths = [Path(path) for path in self.build_config.target_config["only-include"]]
        build_data["force_include"] = {
            source: destination
            for source, destination in build_data["force_include"].items()
            if any(Path(destination) == allowed or allowed in Path(destination).parents for allowed in allowed_paths)
        }
