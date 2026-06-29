from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Generator, Literal


@dataclass
class ObservabilityConfig:
    """OpenTelemetry configuration for deepcrew-ai."""

    otel_endpoint: str | None = None
    service_name: str = "deepcrew"
    enabled: bool = True
    export_format: Literal["grpc", "http"] = "grpc"


def get_tracer(config: ObservabilityConfig) -> Any:
    """Return an OpenTelemetry Tracer. Raises ImportError if SDK not installed."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        raise ImportError(
            "OpenTelemetry SDK is not installed. "
            "Install it with: pip install deepcrew-ai[otel]"
        )

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)

    if config.otel_endpoint:
        try:
            if config.export_format == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
            exporter = OTLPSpanExporter(endpoint=config.otel_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            raise ImportError(
                "OpenTelemetry OTLP exporter not installed. "
                "Install it with: pip install deepcrew-ai[otel]"
            )

    trace.set_tracer_provider(provider)
    return trace.get_tracer(config.service_name)


@contextmanager
def agent_span(
    config: ObservabilityConfig | None, model: str, agent_id: str
) -> Generator[Any, None, None]:
    """Context manager wrapping the full agent execution in an OTel span."""
    if config is None or not config.enabled:
        with nullcontext():
            yield
        return

    tracer = get_tracer(config)
    with tracer.start_as_current_span(f"agent.run") as span:
        span.set_attribute("agent.id", agent_id)
        span.set_attribute("agent.model", model)
        yield span


@contextmanager
def llm_span(
    config: ObservabilityConfig | None, model: str, agent_id: str
) -> Generator[Any, None, None]:
    """Context manager wrapping a single LLM API call in an OTel span."""
    if config is None or not config.enabled:
        with nullcontext():
            yield
        return

    tracer = get_tracer(config)
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("agent.id", agent_id)
        yield span


@contextmanager
def tool_span(
    config: ObservabilityConfig | None, tool_name: str, agent_id: str
) -> Generator[Any, None, None]:
    """Context manager wrapping tool execution in an OTel span."""
    if config is None or not config.enabled:
        with nullcontext():
            yield
        return

    tracer = get_tracer(config)
    with tracer.start_as_current_span("tool.call") as span:
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("agent.id", agent_id)
        yield span


@contextmanager
def workflow_step_span(
    config: ObservabilityConfig | None, step_name: str
) -> Generator[Any, None, None]:
    """Context manager wrapping a workflow DAG node execution in an OTel span."""
    if config is None or not config.enabled:
        with nullcontext():
            yield
        return

    tracer = get_tracer(config)
    with tracer.start_as_current_span("workflow.step") as span:
        span.set_attribute("step.name", step_name)
        yield span
