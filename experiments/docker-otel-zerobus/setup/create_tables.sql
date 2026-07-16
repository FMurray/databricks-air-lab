-- Zerobus OTLP target tables (predefined v2 schemas — Zerobus will NOT create/alter these).
-- Replace <catalog>.<schema> and <sp-uuid>. Prefix: air_
-- Full spans/metrics DDL: docs.databricks.com/aws/en/ingestion/opentelemetry/configure

CREATE TABLE <catalog>.<schema>.air_otel_logs (
  record_id STRING, time TIMESTAMP, date DATE, service_name STRING, event_name STRING,
  trace_id STRING, span_id STRING, time_unix_nano LONG, observed_time_unix_nano LONG,
  severity_number STRING, severity_text STRING, body VARIANT, attributes VARIANT,
  dropped_attributes_count INT, flags INT,
  resource STRUCT<attributes: VARIANT, dropped_attributes_count: INT>,
  resource_schema_url STRING,
  instrumentation_scope STRUCT<name: STRING, version: STRING, attributes: VARIANT, dropped_attributes_count: INT>,
  log_schema_url STRING
) USING DELTA
CLUSTER BY (time, service_name)
TBLPROPERTIES (
  'otel.schemaVersion' = 'v2',
  'delta.checkpointPolicy' = 'classic',
  'delta.enableVariantShredding' = 'true',
  'delta.feature.variantShredding-preview' = 'supported',
  'delta.feature.variantType-preview' = 'supported'
);

-- Metrics table: paste the full metrics DDL from the configure page as
-- <catalog>.<schema>.air_otel_metrics (schema is long: gauge/sum/histogram/... structs).
-- Spans table likewise as air_otel_spans if/when we export traces.

-- Grants: ALL PRIVILEGES is NOT sufficient — MODIFY + SELECT must be explicit per table.
GRANT USE CATALOG ON CATALOG <catalog> TO `<sp-uuid>`;
GRANT USE SCHEMA ON SCHEMA <catalog>.<schema> TO `<sp-uuid>`;
GRANT MODIFY, SELECT ON TABLE <catalog>.<schema>.air_otel_logs TO `<sp-uuid>`;
GRANT MODIFY, SELECT ON TABLE <catalog>.<schema>.air_otel_metrics TO `<sp-uuid>`;
