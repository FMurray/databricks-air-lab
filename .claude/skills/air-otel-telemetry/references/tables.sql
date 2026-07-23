-- Zerobus OTLP target tables: predefined OTEL v2 schemas. Zerobus will NOT create or alter these.
-- Replace <catalog>.<schema> and <sp-application-id>. Source: docs.databricks.com/aws/en/ingestion/opentelemetry/configure
-- (spans table omitted — add from the same docs page if exporting traces).

CREATE TABLE IF NOT EXISTS <catalog>.<schema>.air_otel_logs (
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

CREATE TABLE IF NOT EXISTS <catalog>.<schema>.air_otel_metrics (
  record_id STRING, time TIMESTAMP, date DATE, service_name STRING,
  start_time_unix_nano LONG, time_unix_nano LONG, name STRING, description STRING,
  unit STRING, metric_type STRING,
  gauge STRUCT<value: DOUBLE, exemplars: ARRAY<STRUCT<time_unix_nano: LONG, value: DOUBLE, span_id: STRING, trace_id: STRING, filtered_attributes: VARIANT>>, attributes: VARIANT, flags: INT>,
  sum STRUCT<value: DOUBLE, exemplars: ARRAY<STRUCT<time_unix_nano: LONG, value: DOUBLE, span_id: STRING, trace_id: STRING, filtered_attributes: VARIANT>>, attributes: VARIANT, flags: INT, aggregation_temporality: STRING, is_monotonic: BOOLEAN>,
  histogram STRUCT<count: LONG, sum: DOUBLE, bucket_counts: ARRAY<LONG>, explicit_bounds: ARRAY<DOUBLE>, exemplars: ARRAY<STRUCT<time_unix_nano: LONG, value: DOUBLE, span_id: STRING, trace_id: STRING, filtered_attributes: VARIANT>>, attributes: VARIANT, flags: INT, min: DOUBLE, max: DOUBLE, aggregation_temporality: STRING>,
  exponential_histogram STRUCT<attributes: VARIANT, count: LONG, sum: DOUBLE, scale: INT, zero_count: LONG, positive_bucket: STRUCT<offset: INT, bucket_counts: ARRAY<LONG>>, negative_bucket: STRUCT<offset: INT, bucket_counts: ARRAY<LONG>>, flags: INT, exemplars: ARRAY<STRUCT<time_unix_nano: LONG, value: DOUBLE, span_id: STRING, trace_id: STRING, filtered_attributes: VARIANT>>, min: DOUBLE, max: DOUBLE, zero_threshold: DOUBLE, aggregation_temporality: STRING>,
  summary STRUCT<count: LONG, sum: DOUBLE, quantile_values: ARRAY<STRUCT<quantile: DOUBLE, value: DOUBLE>>, attributes: VARIANT, flags: INT>,
  metadata VARIANT,
  resource STRUCT<attributes: VARIANT, dropped_attributes_count: INT>,
  resource_schema_url STRING,
  instrumentation_scope STRUCT<name: STRING, version: STRING, attributes: VARIANT, dropped_attributes_count: INT>,
  metric_schema_url STRING
) USING DELTA
CLUSTER BY (time, service_name)
TBLPROPERTIES (
  'otel.schemaVersion' = 'v2',
  'delta.checkpointPolicy' = 'classic',
  'delta.enableVariantShredding' = 'true',
  'delta.feature.variantShredding-preview' = 'supported',
  'delta.feature.variantType-preview' = 'supported'
);

-- Grants: ALL PRIVILEGES is NOT sufficient — MODIFY + SELECT must be explicit per table.
GRANT USE CATALOG ON CATALOG <catalog> TO `<sp-application-id>`;
GRANT USE SCHEMA ON SCHEMA <catalog>.<schema> TO `<sp-application-id>`;
GRANT MODIFY, SELECT ON TABLE <catalog>.<schema>.air_otel_logs TO `<sp-application-id>`;
GRANT MODIFY, SELECT ON TABLE <catalog>.<schema>.air_otel_metrics TO `<sp-application-id>`;

-- Verification after a run:
-- SELECT time, service_name, severity_text, body::string,
--        resource.attributes:["air.requester"]::string AS requester
-- FROM <catalog>.<schema>.air_otel_logs ORDER BY time DESC LIMIT 20;
