# Databricks notebook source
# MAGIC %md
# MAGIC # MedQuAD Bronze Ingestion
# MAGIC
# MAGIC This notebook ingests the raw MedQuAD CSV dataset from the Unity Catalog volume into the Bronze layer.
# MAGIC
# MAGIC The Bronze layer preserves the source data without applying business or data-quality cleaning rules. Only ingestion metadata is added.

# COMMAND ----------

from pyspark import pipelines as dp
from pyspark.sql import functions as F

CSV_PATH = "/Volumes/dbacademy/medquad_project/raw/medquad/medquad.csv"

# COMMAND ----------

@dp.materialized_view(
    name="bronze_medquad",
    comment="Raw MedQuAD dataset ingested from the source CSV file."
)
def bronze_medquad():

    raw_df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("multiline", "true")
        .option("quote", '"')
        .option("escape", '"')
        .load(CSV_PATH)
    )

    return (
        raw_df
        .select(
            "*",
            F.col("_metadata.file_path").alias("source_file")
        )
        .withColumn(
            "ingestion_timestamp",
            F.current_timestamp()
        )
    )