-- Databricks notebook source
CREATE OR REPLACE TABLE dbacademy.medquad_project.gold_medquad_rag_index_source
USING DELTA
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
AS
SELECT *
FROM dbacademy.medquad_project.gold_medquad_rag_chunks;

-- COMMAND ----------

SELECT COUNT(*)
FROM dbacademy.medquad_project.gold_medquad_rag_index_source;

-- COMMAND ----------
