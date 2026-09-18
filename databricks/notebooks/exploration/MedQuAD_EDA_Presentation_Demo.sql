-- Databricks notebook source
-- MAGIC %md
-- MAGIC # MedQuAD EDA — presentation demo
-- MAGIC
-- MAGIC This notebook contains only the evidence needed during the live presentation.
-- MAGIC
-- MAGIC >
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Part A — after slide 6
-- MAGIC
-- MAGIC Show only the two results below. Explain the result and the pipeline decision, rather than every line of PySpark.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 1. Answer availability
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,DEMO 1/10 — Answer availability
-- MAGIC %python
-- MAGIC from pyspark.sql import functions as F
-- MAGIC
-- MAGIC eda_df = medquad_df.withColumn(
-- MAGIC     "answer_missing",
-- MAGIC     F.col("answer").isNull()
-- MAGIC     | (F.trim(F.col("answer")) == "")
-- MAGIC     | (F.trim(F.col("answer")) == "\\N")
-- MAGIC )
-- MAGIC
-- MAGIC eda_df.groupBy("answer_missing").count().show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 2. Answer-length distribution
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,DEMO 2/10 — Answer-length distribution
-- MAGIC %python
-- MAGIC
-- MAGIC answered_df = eda_df.filter(
-- MAGIC     ~F.col("answer_missing")
-- MAGIC )
-- MAGIC
-- MAGIC answer_length_df = answered_df.withColumn(
-- MAGIC     "answer_length",
-- MAGIC     F.length(F.col("answer"))
-- MAGIC )
-- MAGIC
-- MAGIC display(
-- MAGIC     answer_length_df.select(
-- MAGIC         F.min("answer_length").alias("min_length"),
-- MAGIC         F.avg("answer_length").alias("avg_length"),
-- MAGIC         F.max("answer_length").alias("max_length")
-- MAGIC     )
-- MAGIC )
-- MAGIC
-- MAGIC display(
-- MAGIC     answer_length_df.select(
-- MAGIC         F.expr("""
-- MAGIC             percentile_approx(
-- MAGIC                 answer_length,
-- MAGIC                 array(0.25, 0.5, 0.75, 0.90, 0.95, 0.99)
-- MAGIC             )
-- MAGIC         """).alias("answer_length_percentiles")
-- MAGIC     )
-- MAGIC )
-- MAGIC
-- MAGIC display(
-- MAGIC     answer_length_df
-- MAGIC     .select(
-- MAGIC         "document_source",
-- MAGIC         "question_focus",
-- MAGIC         "question_type",
-- MAGIC         "question",
-- MAGIC         "answer",
-- MAGIC         "answer_length"
-- MAGIC     )
-- MAGIC     .orderBy("answer_length")
-- MAGIC     .limit(30)
-- MAGIC )
-- MAGIC
-- MAGIC display(
-- MAGIC     answer_length_df
-- MAGIC     .select(
-- MAGIC         "document_source",
-- MAGIC         "question_focus",
-- MAGIC         "question_type",
-- MAGIC         "question",
-- MAGIC         "answer",
-- MAGIC         "answer_length"
-- MAGIC     )
-- MAGIC     .orderBy(F.desc("answer_length"))
-- MAGIC     .limit(30)
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC > **Return to PowerPoint slide 7.** Continue through the architecture, pipeline and Silver slides. Open this notebook again only after slide 11.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Part B — after slide 11
-- MAGIC
-- MAGIC This section gives one continuous explanation: classifier rules, evaluation sample, model call, comparison and final results.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 3. Classifier rules
-- MAGIC
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1, DEMO 3/10 — Classifier rules
-- MAGIC %python
-- MAGIC
-- MAGIC quality_prompt = """
-- MAGIC You are a data-quality classifier for a medical question-answer dataset.
-- MAGIC
-- MAGIC Your task is NOT to verify whether the medical information is factually correct.
-- MAGIC Your task is to determine whether the provided ANSWER meaningfully responds
-- MAGIC to the provided QUESTION.
-- MAGIC
-- MAGIC First classify the answer's quality_status as exactly one of:
-- MAGIC
-- MAGIC VALID
-- MAGIC - The answer provides meaningful information relevant to and responsive to the question.
-- MAGIC - Short answers can still be VALID.
-- MAGIC - An answer can begin with a heading or a restatement of the question and still
-- MAGIC   be VALID if useful answer content follows.
-- MAGIC
-- MAGIC INVALID
-- MAGIC - The answer does not meaningfully answer the question.
-- MAGIC
-- MAGIC If quality_status is INVALID, classify issue_type as exactly one of:
-- MAGIC
-- MAGIC QUESTION_AS_ANSWER
-- MAGIC - The answer only repeats or rephrases the question as another question.
-- MAGIC - There is no substantive answer after it.
-- MAGIC
-- MAGIC NON_INFORMATIVE
-- MAGIC - The answer is related to the topic but provides essentially no useful answer.
-- MAGIC - For example, it only tells the reader to consult another resource.
-- MAGIC
-- MAGIC IRRELEVANT
-- MAGIC - The answer contains information, but it does not meaningfully answer
-- MAGIC   the specific question being asked.
-- MAGIC
-- MAGIC FRAGMENT_OR_HEADING
-- MAGIC - The answer consists only of a webpage heading, navigation text,
-- MAGIC   label, or incomplete fragment with no substantive answer.
-- MAGIC
-- MAGIC If quality_status is VALID, issue_type must be null.
-- MAGIC
-- MAGIC Important:
-- MAGIC - Do not reject an answer merely because it is short.
-- MAGIC - Do not reject an answer merely because it contains a question.
-- MAGIC - If useful answer content follows a question-like heading, it can still be VALID.
-- MAGIC - An answer can still be VALID if it states that the requested information
-- MAGIC   is unknown, unavailable, uncertain, or has not yet been established,
-- MAGIC   as long as that statement directly responds to the question.
-- MAGIC - Information that is merely related to the same topic is not enough.
-- MAGIC   The answer must meaningfully address what the question specifically asks.
-- MAGIC - For questions asking what something is, information only about risk factors,
-- MAGIC   treatment, prevention, or management is not sufficient unless it also
-- MAGIC   meaningfully explains the thing being asked about.
-- MAGIC - Focus on whether the answer meaningfully responds to the question.
-- MAGIC - Do not judge whether the medical claims are factually correct.
-- MAGIC
-- MAGIC Return the result as a JSON object with exactly these three fields:
-- MAGIC
-- MAGIC {
-- MAGIC   "quality_status": "VALID or INVALID",
-- MAGIC   "issue_type": null,
-- MAGIC   "reason": "A short explanation for the classification"
-- MAGIC }
-- MAGIC
-- MAGIC For INVALID answers, issue_type must contain one of:
-- MAGIC QUESTION_AS_ANSWER, NON_INFORMATIVE, IRRELEVANT, FRAGMENT_OR_HEADING.
-- MAGIC
-- MAGIC For VALID answers, issue_type must be null.
-- MAGIC
-- MAGIC Return JSON only. Do not include markdown or any text outside the JSON object.
-- MAGIC
-- MAGIC QUESTION:
-- MAGIC %s
-- MAGIC
-- MAGIC ANSWER:
-- MAGIC %s
-- MAGIC """

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 4. Independent extended evaluation sample
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,DEMO 4/10 — Evaluation sample composition
-- MAGIC %python
-- MAGIC
-- MAGIC display(
-- MAGIC     extended_evaluation_df
-- MAGIC     .groupBy("sample_group")
-- MAGIC     .count()
-- MAGIC )
-- MAGIC
-- MAGIC print(
-- MAGIC     "Total extended evaluation rows:",
-- MAGIC     extended_evaluation_df.count()
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 5. Manual ground truth
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,DEMO 5/10 — Ground-truth distribution
-- MAGIC %python
-- MAGIC
-- MAGIC display(
-- MAGIC     extended_ground_truth_df
-- MAGIC     .groupBy("expected_status")
-- MAGIC     .count()
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 6. Classifier execution
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,DEMO 6/10 — LLM call with ai_query
-- MAGIC %python
-- MAGIC
-- MAGIC extended_predictions_raw_df = (
-- MAGIC     extended_classifier_input_df
-- MAGIC     .selectExpr(
-- MAGIC         "*",
-- MAGIC         """
-- MAGIC         ai_query(
-- MAGIC             'databricks-gpt-oss-20b',
-- MAGIC             classifier_prompt,
-- MAGIC             modelParameters => named_struct(
-- MAGIC                 'temperature', 0.0,
-- MAGIC                 'max_tokens', 500,
-- MAGIC                 'reasoning_effort', 'low'
-- MAGIC             ),
-- MAGIC             responseFormat => '{"type":"json_object"}'
-- MAGIC         ) AS classification
-- MAGIC         """
-- MAGIC     )
-- MAGIC )

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC from pyspark.sql import functions as F
-- MAGIC from pyspark.sql.types import StructType, StructField, StringType
-- MAGIC
-- MAGIC
-- MAGIC # Execute ai_query once and keep the returned results
-- MAGIC prediction_rows = (
-- MAGIC     extended_predictions_raw_df
-- MAGIC     .select(
-- MAGIC         "evaluation_id",
-- MAGIC         "sample_group",
-- MAGIC         "document_source",
-- MAGIC         "question",
-- MAGIC         "answer",
-- MAGIC         "expected_status",
-- MAGIC         "classification"
-- MAGIC     )
-- MAGIC     .collect()
-- MAGIC )
-- MAGIC
-- MAGIC
-- MAGIC # Convert the collected results back into a Spark DataFrame
-- MAGIC materialized_predictions_df = spark.createDataFrame(
-- MAGIC     prediction_rows
-- MAGIC )
-- MAGIC
-- MAGIC
-- MAGIC # Define the structure of the JSON returned by the LLM
-- MAGIC classification_schema = StructType([
-- MAGIC     StructField("quality_status", StringType(), True),
-- MAGIC     StructField("issue_type", StringType(), True),
-- MAGIC     StructField("reason", StringType(), True)
-- MAGIC ])
-- MAGIC
-- MAGIC
-- MAGIC # Parse the JSON and create readable prediction columns
-- MAGIC extended_predictions_df = (
-- MAGIC     materialized_predictions_df
-- MAGIC     .withColumn(
-- MAGIC         "parsed_classification",
-- MAGIC         F.from_json(
-- MAGIC             F.col("classification"),
-- MAGIC             classification_schema
-- MAGIC         )
-- MAGIC     )
-- MAGIC     .withColumn(
-- MAGIC         "predicted_status",
-- MAGIC         F.upper(
-- MAGIC             F.trim(
-- MAGIC                 F.col("parsed_classification.quality_status")
-- MAGIC             )
-- MAGIC         )
-- MAGIC     )
-- MAGIC     .withColumn(
-- MAGIC         "predicted_issue_type",
-- MAGIC         F.upper(
-- MAGIC             F.trim(
-- MAGIC                 F.col("parsed_classification.issue_type")
-- MAGIC             )
-- MAGIC         )
-- MAGIC     )
-- MAGIC     .withColumn(
-- MAGIC         "prediction_reason",
-- MAGIC         F.col("parsed_classification.reason")
-- MAGIC     )
-- MAGIC     .drop("parsed_classification")
-- MAGIC )
-- MAGIC
-- MAGIC
-- MAGIC # Show the manual label and the LLM prediction together
-- MAGIC display(
-- MAGIC     extended_predictions_df
-- MAGIC     .select(
-- MAGIC         "evaluation_id",
-- MAGIC         "question",
-- MAGIC         "answer",
-- MAGIC         "expected_status",
-- MAGIC         "predicted_status",
-- MAGIC         "predicted_issue_type",
-- MAGIC         "prediction_reason"
-- MAGIC     )
-- MAGIC     .orderBy("evaluation_id")
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 7. Evaluation comparison
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,DEMO 7/10 — Compare expected and predicted status
-- MAGIC %python
-- MAGIC
-- MAGIC extended_evaluation_results_df = (
-- MAGIC     extended_predictions_df
-- MAGIC     .withColumn(
-- MAGIC         "is_correct",
-- MAGIC         F.col("expected_status") == F.col("predicted_status")
-- MAGIC     )
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 8. Accuracy
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,DEMO 8/10 — Accuracy: 111 of 112
-- MAGIC %python
-- MAGIC
-- MAGIC display(
-- MAGIC     extended_evaluation_results_df
-- MAGIC     .agg(
-- MAGIC         F.count("*").alias("total_records"),
-- MAGIC         F.sum(
-- MAGIC             F.col("is_correct").cast("int")
-- MAGIC         ).alias("correct_predictions"),
-- MAGIC         F.round(
-- MAGIC             F.avg(
-- MAGIC                 F.col("is_correct").cast("double")
-- MAGIC             ) * 100,
-- MAGIC             2
-- MAGIC         ).alias("accuracy_percentage")
-- MAGIC     )
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 9. Confusion matrix
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,DEMO 9/10 — Confusion matrix
-- MAGIC %python
-- MAGIC
-- MAGIC display(
-- MAGIC     extended_evaluation_results_df
-- MAGIC     .groupBy(
-- MAGIC         "expected_status",
-- MAGIC         "predicted_status"
-- MAGIC     )
-- MAGIC     .count()
-- MAGIC     .orderBy(
-- MAGIC         "expected_status",
-- MAGIC         "predicted_status"
-- MAGIC     )
-- MAGIC )