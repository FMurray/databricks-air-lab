# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — uv preference entry point
# MAGIC
# MAGIC This entry point selects the uv engine in the canonical worker. AIR's frozen versions become
# MAGIC preferences in `uv pip compile`: compatible versions stay fixed, while uv backtracks across
# MAGIC incompatible parents and transitive dependencies in one solve. The canonical worker then
# MAGIC computes the name+version delta and uses pip only to download and validate exact wheels.

# COMMAND ----------
resolver_engine = "uv"

# COMMAND ----------
# MAGIC %run ./resolve_worker
