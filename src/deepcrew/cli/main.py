from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    from .. import __version__

    parser = argparse.ArgumentParser(
        prog="deepcrew",
        description="deepcrew-ai — multi-agent AI library CLI",
    )
    parser.add_argument("--version", action="version", version=f"deepcrew-ai {__version__}")
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

    if args.command == "run":
        from .run_cmd import run_workflow_file

        asyncio.run(run_workflow_file(args.file, args.input, not args.no_stream))
    elif args.command == "agents" and args.agents_cmd == "list":
        from .agents_cmd import list_agents

        list_agents(args.config)


if __name__ == "__main__":
    main()
