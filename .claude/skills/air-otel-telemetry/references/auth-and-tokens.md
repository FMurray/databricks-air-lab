# Zerobus OTLP auth: token recipe and diagnostics

## Why ordinary tokens fail silently

The Databricks edge in front of `*.zerobus.<region>.cloud.databricks.com` maps auth failures to
`grpc-status: 0` (OK) with the real error only in response headers:

```
HTTP/2 200
x-databricks-reason-phrase: Invalid token audience     ← the actual problem
grpc-status: 0                                          ← what your OTLP client sees: "success"
```

Consequences: garbage tokens, wrong-audience tokens, nonexistent tables, and even bogus workspace
IDs all "export successfully". Rows never appear; no client-side error ever fires. (Once auth
passes, real errors DO surface properly — e.g. `PERMISSION_DENIED` for a table the token isn't
scoped to.)

## The working recipe (SP client_credentials only)

```bash
authorization_details='[
 {"type":"unity_catalog_privileges","privileges":["USE CATALOG"],"object_type":"CATALOG","object_full_path":"<catalog>"},
 {"type":"unity_catalog_privileges","privileges":["USE SCHEMA"],"object_type":"SCHEMA","object_full_path":"<catalog>.<schema>"},
 {"type":"unity_catalog_privileges","privileges":["SELECT","MODIFY"],"object_type":"TABLE","object_full_path":"<catalog>.<schema>.<table>"}]'

curl -u "$CLIENT_ID:$CLIENT_SECRET" \
  -d grant_type=client_credentials -d scope=all-apis \
  -d "resource=api://databricks/workspaces/${WORKSPACE_ID}/zerobusDirectWriteApi" \
  --data-urlencode "authorization_details=$authorization_details" \
  "https://<workspace-host>/oidc/v1/token"
```

- One token per signal (each is scoped to its one target table; requests carry exactly one
  `x-databricks-zerobus-table-name`).
- Verify the minted token's claims: `aud` must be `workspaces/<id>/zerobusDirectWriteApi` and
  `authorization_details` must be present in the JWT payload.
- Tokens expire in **1 hour**. Short jobs: mint at start (what `assets/airtel.py` does). Long
  jobs: run an OTEL Collector with `oauth2clientauthextension` refresh, or re-init exporters.
- Dead ends, so you don't retry them: user OAuth tokens (CLI `databricks auth token`) can't carry
  authorization_details; RFC 8693 token-exchange requires the subject token to already have them.
  PATs are not JWTs ("Malformed token").

## Endpoint

`https://<WORKSPACE_ID>.zerobus.<REGION>.cloud.databricks.com:443` — region must match the
workspace's region. gRPC only (OTLP/HTTP unsupported). Headers per request (lowercase for gRPC):
`authorization: Bearer <token>` and `x-databricks-zerobus-table-name: <catalog>.<schema>.<table>`.

## Diagnostics ladder (fastest first)

1. `scripts/probe.py` — synchronous single-record export, prints real gRPC code/details.
2. `scripts/probe.py --raw` (or curl `--http2 -D -` as below) — exposes
   `x-databricks-reason-phrase`, which the OTLP client never shows:
   ```bash
   curl -s --http2 -D - -o /dev/null -X POST \
     "https://$WS_ID.zerobus.$REGION.cloud.databricks.com:443/opentelemetry.proto.collector.logs.v1.LogsService/Export" \
     -H "content-type: application/grpc" -H "te: trailers" \
     -H "authorization: Bearer $TOKEN" \
     -H "x-databricks-zerobus-table-name: $TABLE" --data-binary "" \
     | grep -iE "reason-phrase|grpc-status"
   ```
   Healthy auth shows `grpc-status: 13` + "Missing request message" (curl sent no valid frame —
   that's expected and means you reached the real handler). Any `x-databricks-reason-phrase`
   means the edge rejected you.
3. Rows still missing with clean probe? Check the disqualifiers: table on default storage,
   catalog-commits enabled, non-managed table, region mismatch, table name with non-ASCII chars.
4. `system.lakeflow.zerobus_stream` (account-wide per region) shows stream open/close events and
   errors per table — if your table never appears there, requests aren't reaching Zerobus.

## Reason-phrase → fix map (observed)

| x-databricks-reason-phrase | Fix |
|---|---|
| `Malformed token` | Not a JWT (PAT/garbage). Mint SP OAuth token. |
| `Invalid token audience` | Missing/wrong `resource=` at mint time. Use the recipe above. |
| (none, grpc-status 13 "Missing request message") | Auth fine; your request body was empty/invalid — expected for the curl probe. |
| `PERMISSION_DENIED ... missing SELECT privilege for table X` (in-band, non-zero status) | Token's authorization_details don't cover THIS table — mint per-table. |
