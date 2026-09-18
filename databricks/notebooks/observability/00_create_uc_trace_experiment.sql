-- Databricks notebook source
-- MAGIC %python
-- MAGIC %pip install --upgrade --force-reinstall "mlflow[databricks]>=3.14.0"

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.library.restartPython()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import os
-- MAGIC import mlflow
-- MAGIC
-- MAGIC from mlflow.entities.trace_location import UnityCatalog
-- MAGIC
-- MAGIC
-- MAGIC mlflow.set_tracking_uri("databricks")
-- MAGIC
-- MAGIC os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = (
-- MAGIC     "cd48d9ff05b635fc"
-- MAGIC )
-- MAGIC
-- MAGIC experiment = mlflow.set_experiment(
-- MAGIC     experiment_name="/Shared/medquad-rag-traces-uc",
-- MAGIC     trace_location=UnityCatalog(
-- MAGIC         catalog_name="dbacademy",
-- MAGIC         schema_name="medquad_project",
-- MAGIC         table_prefix="medquad_rag_traces",
-- MAGIC     ),
-- MAGIC )
-- MAGIC
-- MAGIC print("Experiment ID:", experiment.experiment_id)
-- MAGIC print(
-- MAGIC     "Spans table:",
-- MAGIC     experiment.trace_location.full_otel_spans_table_name,
-- MAGIC )

-- COMMAND ----------
