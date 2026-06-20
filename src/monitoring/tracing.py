"""OpenTelemetry tracing configuration.

Provides a consistent tracer for the application.  When OTEL is disabled
(default for development), spans are created but not exported.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

_configured = False


def configure_tracing(
    service_name: str = "knowledge-base",
    enabled: bool = False,
    otlp_endpoint: str | None = None,
) -> None:
    """Configure OpenTelemetry tracing.

    Args:
        service_name: Service name attached to all spans.
        enabled: If False, tracing is a no-op (spans not exported).
        otlp_endpoint: OTLP gRPC endpoint (e.g. http://localhost:4317).
    """
    global _configured
    if _configured:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if enabled and otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except ImportError:
            # Fall back to console exporter if OTLP package not available.
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer(name: str) -> trace.Tracer:
    """Return an OpenTelemetry tracer for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        OpenTelemetry Tracer instance.

    Example:
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("embed_batch"):
        ...     vectors = await provider.embed_batch(texts)
    """
    return trace.get_tracer(name)
