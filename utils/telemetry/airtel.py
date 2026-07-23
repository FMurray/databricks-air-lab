"""airtel — OTEL telemetry for Databricks AIR workloads via Zerobus OTLP.

Copy this file next to your training code and call `airtel.init(service_name=...)` once at
startup. It wires: stdlib-logging → OTLP logs, a meter for custom metrics, per-GPU NVML gauges
(graceful no-op off-GPU), requester identity (resource attrs + baggage → per-record attrs).

Env contract (set via the workload YAML — see the air-otel-telemetry skill):
  WORKSPACE_ID, WORKSPACE_URL, ZEROBUS_REGION       endpoint + token minting
  OTEL_LOGS_TABLE, OTEL_METRICS_TABLE               pre-created UC Delta tables
  DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET    SP creds (from workload secrets)
  ZEROBUS_TOKEN                                     optional pre-minted token (testing only)
  AIR_TEAM, SUBMITTED_BY                            optional identity overrides

Hard-won constraints honored here:
  - gRPC only; one exporter per signal, each with its own table header
  - tokens minted per-table with resource=...zerobusDirectWriteApi + authorization_details;
    anything else is silently dropped by the edge (grpc-status 0 on auth failure!)
  - tokens live 1h: fine for jobs shorter than that; longer jobs need a collector sidecar
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

import requests
from opentelemetry import baggage, context as otel_context, metrics
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource


def _endpoint() -> str:
    return (
        f"https://{os.environ['WORKSPACE_ID']}.zerobus."
        f"{os.environ['ZEROBUS_REGION']}.cloud.databricks.com:443"
    )


def mint_token(table: str) -> str:
    """Zerobus-audience SP OAuth token, downscoped to one target table (1h lifetime)."""
    if os.environ.get("ZEROBUS_TOKEN"):
        return os.environ["ZEROBUS_TOKEN"]
    catalog, schema, _ = table.split(".")
    authz = [
        {"type": "unity_catalog_privileges", "privileges": ["USE CATALOG"],
         "object_type": "CATALOG", "object_full_path": catalog},
        {"type": "unity_catalog_privileges", "privileges": ["USE SCHEMA"],
         "object_type": "SCHEMA", "object_full_path": f"{catalog}.{schema}"},
        {"type": "unity_catalog_privileges", "privileges": ["SELECT", "MODIFY"],
         "object_type": "TABLE", "object_full_path": table},
    ]
    resp = requests.post(
        f"{os.environ['WORKSPACE_URL']}/oidc/v1/token",
        auth=(os.environ["DATABRICKS_CLIENT_ID"], os.environ["DATABRICKS_CLIENT_SECRET"]),
        data={
            "grant_type": "client_credentials",
            "scope": "all-apis",
            "resource": f"api://databricks/workspaces/{os.environ['WORKSPACE_ID']}/zerobusDirectWriteApi",
            "authorization_details": json.dumps(authz),
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _headers(table: str):
    # gRPC metadata keys must be lowercase
    return (
        ("authorization", f"Bearer {mint_token(table)}"),
        ("x-databricks-zerobus-table-name", table),
    )


def harvest_identity() -> dict:
    """Requester + correlation IDs. air.requester is DERIVED from HYPERPARAMETERS_PATH
    (present only when the workload YAML has a parameters: block). Self-reported —
    for chargeback-grade truth join air.mlflow_run_id to system.lakeflow.job_run_timeline."""
    ctx = {}
    m = re.search(r"/Workspace/Users/([^/]+)/", os.environ.get("HYPERPARAMETERS_PATH", ""))
    if m:
        ctx["air.requester"] = m.group(1)
    if os.environ.get("SUBMITTED_BY"):
        ctx["air.requester"] = os.environ["SUBMITTED_BY"]
    if os.environ.get("AIR_TEAM"):
        ctx["air.team"] = os.environ["AIR_TEAM"]
    for attr, env in [("air.mlflow_run_id", "MLFLOW_RUN_ID"),
                      ("air.mlflow_experiment_id", "MLFLOW_EXPERIMENT_ID"),
                      ("air.workspace_id", "WORKSPACE_ID")]:
        if os.environ.get(env):
            ctx[attr] = os.environ[env]
    return ctx


class _BaggageFilter(logging.Filter):
    """Stamp OTEL baggage onto every stdlib log record, so identity set in the client
    context reaches all logs — including from libraries. LoggingHandler maps the extra
    record fields to OTEL log attributes."""

    def filter(self, record):
        for k, v in baggage.get_all().items():
            setattr(record, k, v)
        return True


def _register_gpu_metrics(meter) -> None:
    try:
        import pynvml
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                   for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception as e:
        logging.getLogger(__name__).warning("NVML unavailable, skipping GPU metrics: %s", e)
        return

    import pynvml as p
    from opentelemetry.metrics import Observation

    def observe(fn):
        def callback(_options):
            out = []
            for i, h in enumerate(handles):
                try:
                    out.append(Observation(fn(h), {"gpu": str(i)}))
                except Exception:
                    pass
            return out
        return [callback]

    meter.create_observable_gauge("gpu.utilization.percent",
        callbacks=observe(lambda h: p.nvmlDeviceGetUtilizationRates(h).gpu))
    meter.create_observable_gauge("gpu.memory.used.bytes",
        callbacks=observe(lambda h: p.nvmlDeviceGetMemoryInfo(h).used))
    meter.create_observable_gauge("gpu.memory.total.bytes",
        callbacks=observe(lambda h: p.nvmlDeviceGetMemoryInfo(h).total))
    meter.create_observable_gauge("gpu.power.watts",
        callbacks=observe(lambda h: p.nvmlDeviceGetPowerUsage(h) / 1000.0))
    meter.create_observable_gauge("gpu.temperature.celsius",
        callbacks=observe(lambda h: p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU)))


@dataclass
class Telemetry:
    logger_provider: LoggerProvider
    meter_provider: MeterProvider
    meter: object
    loss_gauge: object
    step_counter: object

    def shutdown(self) -> None:
        """Flush and shut down. Call before process exit or trailing batches are lost."""
        self.logger_provider.force_flush()
        self.logger_provider.shutdown()
        self.meter_provider.force_flush()
        self.meter_provider.shutdown()


def init(service_name: str, export_interval_seconds: int = 10) -> Telemetry:
    identity = harvest_identity()
    resource = Resource.create({
        "service.name": service_name,
        "air.node_rank": os.environ.get("POD_RANK", "0"),
        "air.world_size": os.environ.get("WORLD_SIZE", "1"),
        **identity,
    })
    for k, v in identity.items():
        otel_context.attach(baggage.set_baggage(k, v))

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(
        OTLPLogExporter(endpoint=_endpoint(),
                        headers=_headers(os.environ["OTEL_LOGS_TABLE"]))))
    set_logger_provider(logger_provider)
    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    handler.addFilter(_BaggageFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()])

    meter_provider = MeterProvider(resource=resource, metric_readers=[
        PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=_endpoint(),
                               headers=_headers(os.environ["OTEL_METRICS_TABLE"])),
            export_interval_millis=export_interval_seconds * 1000)])
    metrics.set_meter_provider(meter_provider)

    meter = metrics.get_meter(service_name)
    _register_gpu_metrics(meter)
    return Telemetry(
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        meter=meter,
        loss_gauge=meter.create_gauge("train.loss"),
        step_counter=meter.create_counter("train.steps"),
    )
