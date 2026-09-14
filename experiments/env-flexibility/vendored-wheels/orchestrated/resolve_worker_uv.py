# Databricks notebook source
# MAGIC %md
# MAGIC # Compatibility alias — use the canonical resolver
# MAGIC
# MAGIC The former uv implementation duplicated the resolver and retained name-only delta logic.
# MAGIC Resolution and AIR-tag wheel download now have one implementation so the pip and uv entry
# MAGIC points cannot disagree. Existing notebooks may continue `%run`-ing this path.

# COMMAND ----------
# MAGIC %run ./resolve_worker
