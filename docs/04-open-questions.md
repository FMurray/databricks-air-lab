# Open Questions / Things to Verify Hands-On

Product & platform (verify against a real workspace, not docs):

1. `@distributed` — is `gpus=8` truly the only distributed shape (no 2/4-GPU)? What happens on A10?
2. Multi-node CLI: max `num_accelerators` in practice? Scheduling latency for 2+ nodes from a reserved pool vs on-demand?
   ✅ PARTIALLY ANSWERED 2026-07-22: 2 nodes (16xH100) on-demand on e2-demo-field-eng scheduled in
   ~40s submit→running (run 505819227973807). Multinode silently PuPr'd 2026-07-17 (#research-on-air);
   multiples of 8xH100 only (#ai-air-product-champions); field guide: max 16 nodes/128 GPUs, sweet
   spot 3–8 nodes, AWS-only. Max-in-practice + reserved-pool latency still untested.
3. What env vars does the CLI inject into `command` (rank, world size, master addr, `$CODE_SOURCE_PATH`, …)?
   ✅ ANSWERED for the Docker path (docker-images docs): `NUM_NODES`, `LOCAL_WORLD_SIZE`, `WORLD_SIZE`,
   `POD_RANK`/`NODE_RANK`, multi-node: `LOCAL_ADDR`, `MASTER_ADDR`, `MASTER_PORT`. Still verify the
   non-Docker snapshot path matches (the `env | sort` in our workloads covers this).
   ✅ SNAPSHOT PATH CONFIRMED 2026-07-22 (multinode probe, run 505819227973807): `NUM_NODES=2`,
   `NODE_RANK`/`POD_RANK`, `LOCAL_WORLD_SIZE=8`, `WORLD_SIZE=16`, `MASTER_ADDR`, `MASTER_PORT`
   all injected, plus NCCL tuning (`NCCL_DEBUG=INFO`, `NCCL_IB_TIMEOUT=22`, `NCCL_CUMEM_ENABLE=0`,
   `AWS_OFI_NCCL_VERSION=v1.15.0`). torchrun wires straight in — see
   experiments/foundation-models/multinode/probe_multinode.sh.
4. Reserved pool vs on-demand: how does a workload target the pool? Is it implicit per-workspace?
5. `usage_policy_name`: what actually lands in system.billing.usage — does it give any attribution inside a reserved pool?
6. Standard env contents: is there a JRE? gcc? What's on PATH? (matters for multi-language)
   ✅ PARTIALLY ANSWERED 2026-07-22, run 93215537511850, e2-demo-field-eng (env image "4"):
   JRE at /usr/bin/java and gcc at /usr/bin/gcc both present (versions not yet captured);
   Ubuntu 24.04.4, glibc 2.39; snapshot mount is rw + exec (git +x preserved); egress to
   Maven Central/Adoptium open. See experiments/multi-language/NOTES.md for receipts.
7. Docker path: how does `air register image` auth to Docker Hub? Private repos? Scan/allowlist step?
   ✅ ANSWERED (docs): per-user per-tag registration (2–6 min, pulls+caches); private via Docker Hub PAT
   (`docker login` / `--interactive-authenticate` / `--scope`+`--key` secret). No scan/allowlist documented.
7b. Can an AIR container reach `*.zerobus.<region>.cloud.databricks.com:443`?
   ✅ ANSWERED 2026-07-16: yes, egress allowed (verified from job 37776040541298 on e2-demo-field-eng).
   BUT Zerobus OTLP needs an SP-minted token with the zerobusDirectWriteApi resource audience, and the
   edge reports auth failures as grpc-status 0 (silent drop!) — see experiments/docker-otel-zerobus/NOTES.md.
8. UC volume mount semantics inside CLI workloads (FUSE path? read-only?) vs notebook sessions.
9. Snowflake access from an AIR workload (use case 2): egress rules, Lakehouse Federation vs direct connector?
10. Checkpoint/restart ergonomics at the 7-day cap: does `max_retries` resume or restart from scratch?
11. Cross-region fallback: is it observable (region in system tables?) and can it be disabled per-workload? (the customer's Private-Link-only posture makes this compliance-relevant)
12. XGBoost H100 hang: driver/env issue or docs-notebook issue? Repro and bisect.
13. Ray on AIR: what does cluster bootstrap look like on one 8xH100 node? Multi-node Ray via CLI possible?
14. SSH/IDE tunnel: session lifetime, idle timeout, can it attach to 8xH100?

Roadmap (chase PM/eng, don't guess):

15. GPU-only entitlement (customer P0 #2) — design/date from cost-control team (Shuyu, Harsh, Piyush).
16. Per-workload tagging in reserved pools (P0 #3) — Angel/Dima/Yu Peng thread; what's the eng design?
17. B200/B300 timeline (Tejas skeptical it's needed — validate with the customer FM's memory profile).
18. Private-Link-based cross-region backbone (Alex Esibov) — Cloudless Compute wiki (UN space, page 6117393607).
19. Dynamic pool reallocation self-service — anything beyond 48h-notice eng process?
