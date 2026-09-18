# Databricks notebook source
# MAGIC %md
# MAGIC # MedQuAD Silver Cleaning
# MAGIC
# MAGIC This notebook transforms the raw Bronze MedQuAD data into a cleaned Silver dataset.
# MAGIC
# MAGIC The cleaning rules are based on the findings and decisions from the MedQuAD EDA and data-quality analysis.

# COMMAND ----------

from pyspark import pipelines as dp
from pyspark.sql import functions as F

# COMMAND ----------

@dp.materialized_view(
    name="silver_medquad_normalized",
    comment="MedQuAD Bronze data with missing-value representations normalized."
)
def silver_medquad_normalized():

    df = spark.read.table("bronze_medquad")

    optional_columns = [
        "category",
        "umls_cui",
        "umls_semantic_types",
        "umls_semantic_group",
        "synonyms",
        "question_focus",
        "question_type"
    ]

    for column_name in optional_columns:
        df = df.withColumn(
            column_name,
            F.when(
                F.col(column_name).isNull()
                | (F.trim(F.col(column_name)) == "")
                | (F.trim(F.col(column_name)) == "\\N"),
                F.lit(None)
            ).otherwise(F.trim(F.col(column_name)))
        )

    return df

# COMMAND ----------

@dp.materialized_view(
    name="silver_medquad_missing_answer_quarantine",
    comment="MedQuAD rows rejected because they do not contain a usable answer."
)
def silver_medquad_missing_answer_quarantine():

    df = spark.read.table("silver_medquad_normalized")

    return (
        df.filter(
            F.col("answer").isNull()
            | (F.trim(F.col("answer")) == "")
            | (F.trim(F.col("answer")) == "\\N")
        )
        .withColumn("rejected_reason",F.lit("MISSING ANSWER"))
    )



@dp.materialized_view(
    name="silver_medquad_answered",
    comment="MedQuAD rows containing an answer and eligible for further Silver validation."
)
def silver_medquad_answered():

    df = spark.read.table("silver_medquad_normalized")
    return (
        df.filter(
            F.col("answer").isNotNull()
            & (F.trim(F.col("answer")) != "")
            & (F.trim(F.col("answer")) != "\\N")
        )
    )


# COMMAND ----------

@dp.materialized_view(
    name="silver_medquad_question_cleaned",
    comment="Answered MedQuAD rows with harmless duplicate question punctuation normalized."
)
def silver_medquad_question_cleaned():

    df = spark.read.table("silver_medquad_answered")

    return(
        df.withColumn("question", F.regexp_replace(F.trim(F.col("question")), r"\?\s+\?\s*$", "?"))
    )

# COMMAND ----------

@dp.materialized_view(
    name="silver_medquad_malformed_question_quarantine",
    comment="MedQuAD rows rejected because the question is missing its semantic subject."
)
def silver_medquad_malformed_question_quarantine():

    df = spark.read.table("silver_medquad_question_cleaned")

    malformed_pattern = (
        r"(?i)^\s*(what is \(are\)|who is at risk for|how to prevent)\s*\?\s*$"
    )

    return (
        df
        .filter(F.col("question").rlike(malformed_pattern))
        .withColumn(
            "rejection_reason",
            F.lit("MALFORMED_QUESTION")
        )
    )


@dp.materialized_view(
    name="silver_medquad_question_valid",
    comment="Answered MedQuAD rows with usable questions."
)
def silver_medquad_question_valid():

    df = spark.read.table("silver_medquad_question_cleaned")

    malformed_pattern = (
        r"(?i)^\s*(what is \(are\)|who is at risk for|how to prevent)\s*\?\s*$"
    )

    return df.filter(
        ~F.col("question").rlike(malformed_pattern)
    )

# COMMAND ----------

from pyspark.sql.window import Window

@dp.materialized_view(
    name="silver_medquad_deduplicated",
    comment="Valid MedQuAD questions with duplicate question-answer pairs removed deterministically."
)
def silver_medquad_deduplicated():

    df = spark.read.table("silver_medquad_question_valid")

    duplicate_window = (
        Window.partitionBy("question", "answer")
        .orderBy(F.col("document_id").asc_nulls_last(),
                 F.col("question_id").asc_nulls_last())
    )

    return (
        df
        .withColumn("_duplicate_rank", F.row_number().over(duplicate_window))
        .filter(F.col("_duplicate_rank") == 1)
        .drop("_duplicate_rank")
    )

# COMMAND ----------

@dp.materialized_view(
    name="silver_medquad_structured_metadata",
    comment="Deduplicated MedQuAD data with mulit-valued metadata converted to arrays."
)
def silver_medquad_structured_metadata():
    df = spark.read.table("silver_medquad_deduplicated")

    return(
        df
        .withColumn(
            "umls_cui",
            F.when(F.col("umls_cui").isNull(), F.lit(None))
            .otherwise(
                F.split(F.col("umls_cui"), r"\|")
            )
        )
        .withColumn(
            "umls_semantic_types",
            F.when(
                F.col("umls_semantic_types").isNull(),
                F.lit(None)
            ).otherwise(
                F.split(F.col("umls_semantic_types"), r"\|")
            )
        )
        .withColumn(
            "synonyms",
            F.when(
                F.col("synonyms").isNull(),
                F.lit(None)
            ).otherwise(
                F.split(F.col("synonyms"), r"\|")
            )
        )
    )

# COMMAND ----------

quality_prompt = """
You are a data-quality classifier for a medical question-answer dataset.

Your task is NOT to verify whether the medical information is factually correct.
Your task is to determine whether the provided ANSWER meaningfully responds
to the provided QUESTION.

First classify the answer's quality_status as exactly one of:

VALID
- The answer provides meaningful information relevant to and responsive to the question.
- Short answers can still be VALID.
- An answer can begin with a heading or a restatement of the question and still
  be VALID if useful answer content follows.

INVALID
- The answer does not meaningfully answer the question.

If quality_status is INVALID, classify issue_type as exactly one of:

QUESTION_AS_ANSWER
- The answer only repeats or rephrases the question as another question.
- There is no substantive answer after it.

NON_INFORMATIVE
- The answer is related to the topic but provides essentially no useful answer.
- For example, it only tells the reader to consult another resource.

IRRELEVANT
- The answer contains information, but it does not meaningfully answer
  the specific question being asked.

FRAGMENT_OR_HEADING
- The answer consists only of a webpage heading, navigation text,
  label, or incomplete fragment with no substantive answer.

If quality_status is VALID, issue_type must be null.

Important:
- Do not reject an answer merely because it is short.
- Do not reject an answer merely because it contains a question.
- If useful answer content follows a question-like heading, it can still be VALID.
- An answer can still be VALID if it states that the requested information
  is unknown, unavailable, uncertain, or has not yet been established,
  as long as that statement directly responds to the question.
- Information that is merely related to the same topic is not enough.
  The answer must meaningfully address what the question specifically asks.
- For questions asking what something is, information only about risk factors,
  treatment, prevention, or management is not sufficient unless it also
  meaningfully explains the thing being asked about.
- Focus on whether the answer meaningfully responds to the question.
- Do not judge whether the medical claims are factually correct.

Return the result as a JSON object with exactly these three fields:

{
  "quality_status": "VALID or INVALID",
  "issue_type": null,
  "reason": "A short explanation for the classification"
}

For INVALID answers, issue_type must contain one of:
QUESTION_AS_ANSWER, NON_INFORMATIVE, IRRELEVANT, FRAGMENT_OR_HEADING.

For VALID answers, issue_type must be null.

Return JSON only. Do not include markdown or any text outside the JSON object.

QUESTION:
%s

ANSWER:
%s
"""

# COMMAND ----------

@dp.materialized_view(
    name="silver_medquad_classified",
    comment="Deduplicated MedQuAD records classified for semantic answer quality using an LLM."
)
def silver_medquad_classified():

    df = spark.read.table("silver_medquad_structured_metadata")

    df = df.withColumn(
        "_classifier_prompt",
        F.format_string(
            quality_prompt,
            F.col("question"),
            F.col("answer")
        )
    )

    classified_df = df.withColumn(
        "classifier_result",
        F.expr("""
            ai_query(
                'databricks-gpt-oss-20b',
                _classifier_prompt,
                modelParameters => named_struct(
                    'temperature', 0.0,
                    'max_tokens', 500,
                    'reasoning_effort', 'low'
                ),
                responseFormat => '{"type":"json_object"}'
            )
        """)
    )

    result_schema = """
        quality_status STRING,
        issue_type STRING,
        reason STRING
    """

    return (
        classified_df
        .withColumn(
            "_parsed_result",
            F.from_json(
                F.col("classifier_result"),
                result_schema
            )
        )
        .withColumn(
            "quality_status",
            F.col("_parsed_result.quality_status")
        )
        .withColumn(
            "quality_issue_type",
            F.col("_parsed_result.issue_type")
        )
        .withColumn(
            "quality_reason",
            F.col("_parsed_result.reason")
        )
        .drop("_parsed_result", "_classifier_prompt")
    )

# COMMAND ----------

@dp.materialized_view(
    name="silver_medquad",
    comment="Final cleaned MedQuAD dataset containing records accepted for downstream RAG processing."
)
def silver_medquad():

    df = spark.read.table("silver_medquad_classified")

    return (
        df
        .filter(F.col("quality_status") == "VALID")
        .drop("classifier_result")
    )

# COMMAND ----------

@dp.materialized_view(
    name="silver_medquad_quarantine",
    comment="MedQuAD records rejected during Silver data-quality validation."
)
def silver_medquad_quarantine():

    missing_answers = (
        spark.read.table("silver_medquad_missing_answer_quarantine")
        .select(
            "document_id",
            "question_id",
            "document_source",
            "question",
            "answer",
            "source_file",
            F.col("rejected_reason").alias("rejection_reason"),
            F.lit(None).cast("string").alias("classifier_issue_type"),
            F.lit(None).cast("string").alias("classifier_reason")
        )
    )

    malformed_questions = (
        spark.read.table("silver_medquad_malformed_question_quarantine")
        .select(
            "document_id",
            "question_id",
            "document_source",
            "question",
            "answer",
            "source_file",
            F.col("rejection_reason"),
            F.lit(None).cast("string").alias("classifier_issue_type"),
            F.lit(None).cast("string").alias("classifier_reason")
        )
    )

    classifier_invalid = (
        spark.read.table("silver_medquad_classified")
        .filter(F.col("quality_status") == "INVALID")
        .select(
            "document_id",
            "question_id",
            "document_source",
            "question",
            "answer",
            "source_file",
            F.lit("SEMANTIC_LOW_QUALITY_ANSWER").alias("rejection_reason"),
            F.col("quality_issue_type").alias("classifier_issue_type"),
            F.col("quality_reason").alias("classifier_reason")
        )
    )

    return (
        missing_answers
        .unionByName(malformed_questions)
        .unionByName(classifier_invalid)
        .withColumn(
            "rejected_at",
            F.current_timestamp()
        )
    )

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------
