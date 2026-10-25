from __future__ import annotations

import argparse
import asyncio


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("logging")
    group.add_argument(
        "--log-level",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging threshold (default: quiet — warnings and errors only)",
    )
    group.add_argument(
        "--log-profile",
        choices=["dev", "prod", "verbose", "quiet"],
        default=None,
        help="Logging profile preset",
    )
    group.add_argument("--log-json", action="store_true", help="Emit logs as JSON lines")
    group.add_argument(
        "-v", "--verbose", action="store_true", help="Shortcut for --log-profile verbose"
    )
    group.add_argument("-q", "--quiet", action="store_true", help="Only log errors")


def _configure_from_args(args: argparse.Namespace) -> None:
    from ..logging_config import configure_logging

    overrides: dict[str, object] = {}
    profile: str = "quiet"
    if args.verbose:
        profile = "verbose"
    elif args.log_profile:
        profile = args.log_profile
    if args.quiet:
        overrides["level"] = "ERROR"
    if args.log_level:
        overrides["level"] = args.log_level
    if args.log_json:
        overrides["format"] = "json"
    configure_logging(profile, **overrides)


def main() -> None:
    from .. import __version__

    parser = argparse.ArgumentParser(
        prog="deepcrew",
        description="deepcrew-ai — multi-agent AI library CLI",
    )
    parser.add_argument("--version", action="version", version=f"deepcrew-ai {__version__}")
    _add_logging_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    # deepcrew run workflow.yaml [--input "..."] [--no-stream]
    run_p = sub.add_parser("run", help="Run a declarative workflow YAML file")
    run_p.add_argument("file", help="Path to workflow YAML")
    run_p.add_argument("--input", "-i", default=None, help="Initial input text")
    run_p.add_argument("--no-stream", action="store_true", help="Print final output only")

    # deepcrew agents list --config workflow.yaml
    agents_p = sub.add_parser("agents", help="Agent management commands")
    agents_sub = agents_p.add_subparsers(dest="agents_cmd", required=True)
    list_p = agents_sub.add_parser("list", help="List agents in a config file")
    list_p.add_argument("--config", "-c", required=True, help="Path to workflow YAML")

    args = parser.parse_args()
    _configure_from_args(args)

    if args.command == "run":
        from .run_cmd import run_workflow_file

        asyncio.run(run_workflow_file(args.file, args.input, not args.no_stream))
    elif args.command == "agents" and args.agents_cmd == "list":
        from .agents_cmd import list_agents

        list_agents(args.config)


if __name__ == "__main__":
    main()
