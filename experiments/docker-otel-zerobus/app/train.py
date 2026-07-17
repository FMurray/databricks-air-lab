"""Simulated training loop that ships event logs + metrics as OTEL over OTLP/gRPC to Zerobus.

Env contract (set via workload YAML env_variables/secrets):
  WORKSPACE_ID, WORKSPACE_URL, ZEROBUS_REGION      -- endpoint construction
  OTEL_LOGS_TABLE, OTEL_METRICS_TABLE              -- <catalog>.<schema>.<table>, pre-created
  DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET   -- SP creds (secrets), used to mint bearer tokens
  ZEROBUS_TOKEN                                    -- optional pre-minted token override (skips minting)
  EPOCHS, STEPS_PER_EPOCH                          -- loop size (optional)

Zerobus OTLP constraints honored here (hard-won, see NOTES.md):
  - gRPC only (proto.grpc exporter, not http)
  - one table per request -> one exporter per signal, each with its own table header
  - the bearer token MUST be minted with resource=api://databricks/workspaces/<id>/zerobusDirectWriteApi
    and authorization_details for THAT table; a plain all-apis token is rejected -- and the edge
    reports rejections as grpc-status 0 (success!), so a wrong token means silent data loss
  - tokens expire in 1h -> long runs need an OTEL Collector sidecar with oauth2 refresh
"""

import json
import logging
import math
import os
import random
import time

import requests
from opentelemetry import metrics
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource


def mint_token(table: str) -> str:
    """Zerobus-audience SP OAuth token, downscoped to one target table.

    ZEROBUS_TOKEN env overrides (single shared token, e.g. pre-minted for a smoke test).
    """
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


def zerobus_headers(table: str):
    # gRPC metadata keys must be lowercase
    return (
        ("authorization", f"Bearer {mint_token(table)}"),
        ("x-databricks-zerobus-table-name", table),
    )


def setup_telemetry():
    endpoint = (
        f"https://{os.environ['WORKSPACE_ID']}.zerobus."
        f"{os.environ['ZEROBUS_REGION']}.cloud.databricks.com:443"
    )
    resource = Resource.create(
        {
            "service.name": "air-otel-smoke",
            # AIR-injected context so rows are attributable to the workload
            "air.node_rank": os.environ.get("POD_RANK", "0"),
            "air.world_size": os.environ.get("WORLD_SIZE", "1"),
            "air.mlflow_run_id": os.environ.get("MLFLOW_RUN_ID", ""),
        }
    )

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(
                endpoint=endpoint,
                headers=zerobus_headers(os.environ["OTEL_LOGS_TABLE"]),
            )
        )
    )
    set_logger_provider(logger_provider)
    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()])

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=endpoint,
                    headers=zerobus_headers(os.environ["OTEL_METRICS_TABLE"]),
                ),
                export_interval_millis=10_000,
            )
        ],
    )
    metrics.set_meter_provider(meter_provider)
    return logger_provider, meter_provider


def main():
    logger_provider, meter_provider = setup_telemetry()
    log = logging.getLogger("train")
    meter = metrics.get_meter("air-otel-smoke")
    loss_gauge = meter.create_gauge("train.loss")
    step_counter = meter.create_counter("train.steps")

    epochs = int(os.environ.get("EPOCHS", "3"))
    steps = int(os.environ.get("STEPS_PER_EPOCH", "20"))
    log.info("training started", extra={"epochs": epochs, "steps_per_epoch": steps})

    global_step = 0
    for epoch in range(epochs):
        t0 = time.time()
        for _ in range(steps):
            global_step += 1
            loss = 2.0 * math.exp(-0.01 * global_step) + random.uniform(0, 0.05)
            loss_gauge.set(loss, {"epoch": epoch})
            step_counter.add(1)
            time.sleep(0.2)  # stand-in for a training step
        log.info(
            "epoch complete",
            extra={"epoch": epoch, "loss": round(loss, 4), "secs": round(time.time() - t0, 1)},
        )

    log.info("training finished", extra={"global_steps": global_step})
    # flush before the container exits or the last batches are lost
    logger_provider.force_flush()
    logger_provider.shutdown()
    meter_provider.force_flush()
    meter_provider.shutdown()


if __name__ == "__main__":
    main()
