#!/usr/bin/env python3
"""Zerobus OTLP auth probe. ALWAYS run this before wiring telemetry into a real workload:
the edge reports auth failures as grpc-status 0, so a training run can "export successfully"
forever while writing nothing. This probe sends one synchronous log record and reports the
real outcome; --raw additionally shows the edge's x-databricks-reason-phrase header, which
OTLP clients cannot see.

Env: WORKSPACE_ID, ZEROBUS_REGION, OTEL_LOGS_TABLE, and either ZEROBUS_TOKEN or
     WORKSPACE_URL + DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET (mints per the recipe).

Exit codes: 0 = row accepted by real handler; 1 = rejected/unreachable (details printed).
Requires: opentelemetry-exporter-otlp-proto-grpc (for the default mode); curl (for --raw).
"""

import json
import os
import subprocess
import sys
import time


def token() -> str:
    if os.environ.get("ZEROBUS_TOKEN"):
        return os.environ["ZEROBUS_TOKEN"]
    import requests
    table = os.environ["OTEL_LOGS_TABLE"]
    catalog, schema, _ = table.split(".")
    authz = [
        {"type": "unity_catalog_privileges", "privileges": ["USE CATALOG"],
         "object_type": "CATALOG", "object_full_path": catalog},
        {"type": "unity_catalog_privileges", "privileges": ["USE SCHEMA"],
         "object_type": "SCHEMA", "object_full_path": f"{catalog}.{schema}"},
        {"type": "unity_catalog_privileges", "privileges": ["SELECT", "MODIFY"],
         "object_type": "TABLE", "object_full_path": table},
    ]
    r = requests.post(
        f"{os.environ['WORKSPACE_URL']}/oidc/v1/token",
        auth=(os.environ["DATABRICKS_CLIENT_ID"], os.environ["DATABRICKS_CLIENT_SECRET"]),
        data={"grant_type": "client_credentials", "scope": "all-apis",
              "resource": f"api://databricks/workspaces/{os.environ['WORKSPACE_ID']}/zerobusDirectWriteApi",
              "authorization_details": json.dumps(authz)},
        timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def raw_probe(host: str, tok: str, table: str) -> int:
    """HTTP/2 curl with empty body: healthy auth => grpc-status 13 'Missing request message'.
    Any x-databricks-reason-phrase header = edge rejected the request (regardless of grpc-status)."""
    out = subprocess.run(
        ["curl", "-s", "--http2", "-D", "-", "-o", "/dev/null", "--max-time", "15", "-X", "POST",
         f"https://{host}/opentelemetry.proto.collector.logs.v1.LogsService/Export",
         "-H", "content-type: application/grpc", "-H", "te: trailers",
         "-H", f"authorization: Bearer {tok}",
         "-H", f"x-databricks-zerobus-table-name: {table}", "--data-binary", ""],
        capture_output=True, text=True).stdout
    reason = [l for l in out.splitlines() if "reason-phrase" in l.lower()]
    status = [l for l in out.splitlines() if l.lower().startswith("grpc-status")]
    print(out.strip() or "(no response — endpoint unreachable?)")
    if reason:
        print(f"\nREJECTED AT EDGE: {reason[0].split(':',1)[1].strip()}")
        return 1
    if status and status[0].split(":")[1].strip() == "13":
        print("\nAUTH OK (grpc-status 13 on empty body is the healthy signature)")
        return 0
    return 1


def sdk_probe(host: str, tok: str, table: str) -> int:
    import grpc
    from opentelemetry.proto.collector.logs.v1 import logs_service_pb2, logs_service_pb2_grpc
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord, ResourceLogs, ScopeLogs
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource

    stub = logs_service_pb2_grpc.LogsServiceStub(
        grpc.secure_channel(host, grpc.ssl_channel_credentials()))
    now = time.time_ns()
    req = logs_service_pb2.ExportLogsServiceRequest(resource_logs=[ResourceLogs(
        resource=Resource(attributes=[KeyValue(key="service.name",
                                               value=AnyValue(string_value="otlp-probe"))]),
        scope_logs=[ScopeLogs(log_records=[LogRecord(
            time_unix_nano=now, observed_time_unix_nano=now, severity_text="INFO",
            severity_number=9, body=AnyValue(string_value="zerobus otlp probe"))])])])
    md = (("authorization", f"Bearer {tok}"), ("x-databricks-zerobus-table-name", table))
    try:
        stub.Export(req, metadata=md, timeout=30)
    except grpc.RpcError as e:
        print(f"EXPORT FAILED: {e.code()} — {e.details()[:300]}")
        return 1
    print("Export RPC returned OK. REMEMBER: OK can mask edge auth rejection —")
    print("run with --raw to check for x-databricks-reason-phrase, then verify a row landed:")
    print(f"  SELECT * FROM {table} WHERE service_name='otlp-probe' ORDER BY time DESC LIMIT 1;")
    print("(time-to-table is P50 ≤5s; no row within ~30s means silent rejection)")
    return 0


if __name__ == "__main__":
    host = (f"{os.environ['WORKSPACE_ID']}.zerobus."
            f"{os.environ['ZEROBUS_REGION']}.cloud.databricks.com:443")
    table = os.environ["OTEL_LOGS_TABLE"]
    tok = token()
    sys.exit(raw_probe(host, tok, table) if "--raw" in sys.argv
             else sdk_probe(host, tok, table))
