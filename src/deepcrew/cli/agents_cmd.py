from __future__ import annotations


def list_agents(config_path: str) -> None:
    """Load a YAML config file and print a formatted table of agents."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("pyyaml is required. Install with: pip install deepcrew-ai") from exc

    from .yaml_schema import WorkflowYAML

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    spec = WorkflowYAML.model_validate(raw)

    col_w = [20, 30, 8]
    header = f"{'NAME':<{col_w[0]}}  {'MODEL':<{col_w[1]}}  {'TOOLS':<{col_w[2]}}"
    sep = "-" * (sum(col_w) + 4)
    print(header)
    print(sep)
    for a in spec.agents:
        print(f"{a.name:<{col_w[0]}}  {a.model:<{col_w[1]}}  {len(a.tools):<{col_w[2]}}")
