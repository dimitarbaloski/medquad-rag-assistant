# Databricks notebook source
from pyspark import pipelines as dp
from pyspark.sql import functions as F

# COMMAND ----------


import re


TARGET_WORDS = 300
MAX_WORDS = 400


def chunk_answer(answer):

    if answer is None or not answer.strip():
        return []

    # Normalize whitespace
    text = re.sub(r"\s+", " ", answer).strip()

    # Short answers stay as one chunk
    if len(text.split()) <= MAX_WORDS:
        return [text]

    # Split the answer into sentences/bullet segments -? sentence-aware chunking
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])|\s+(?=-\s)",
        text
    )

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_word_count = len(sentence.split())

        # Edge case: one individual sentence is longer than MAX_WORDS
        if sentence_word_count > MAX_WORDS:

            # Save anything already accumulated first
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_word_count = 0

            words = sentence.split()

            for i in range(0, len(words), MAX_WORDS):
                chunks.append(
                    " ".join(words[i:i + MAX_WORDS])
                )

            continue

        # If we already reached the target, start a new chunk.
        # Keep the last complete sentence as overlap.
        if current_word_count >= TARGET_WORDS:

            chunks.append(" ".join(current_sentences))

            overlap_sentence = current_sentences[-1]
            current_sentences = [overlap_sentence]
            current_word_count = len(overlap_sentence.split())

        # If adding this sentence would exceed the hard maximum,
        # finish the current chunk first.
        if (
            current_sentences
            and current_word_count + sentence_word_count > MAX_WORDS
        ):

            chunks.append(" ".join(current_sentences))

            overlap_sentence = current_sentences[-1]
            overlap_word_count = len(overlap_sentence.split())

            # Keep overlap only if there is enough room for the new sentence.
            if overlap_word_count + sentence_word_count <= MAX_WORDS:
                current_sentences = [overlap_sentence]
                current_word_count = overlap_word_count
            else:
                current_sentences = []
                current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    # Save the final unfinished chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks

# COMMAND ----------

@dp.materialized_view(
    name="gold_medquad_selected",
    comment="Silver MedQuAD records projected to the columns required for RAG preparation."
)
def gold_medquad_selected():

    df = (
        spark.read.table("dbacademy.medquad_project.silver_medquad")
        .select(
            "question_id",
            "question",
            "answer",
            "question_focus",
            "question_type",
            "synonyms",
            "category",
            "document_source",
            "document_url"
        )
    )

    return (
        df
        .withColumn(
            "qa_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.coalesce(F.col("question"), F.lit("")),
                    F.coalesce(F.col("answer"), F.lit(""))
                ),
                256
            )
        )
    )

# COMMAND ----------

from pyspark.sql.types import ArrayType, StringType

chunk_answer_udf = F.udf(
    chunk_answer,
    ArrayType(StringType())
)

# COMMAND ----------

@dp.materialized_view(
    name="gold_medquad_chunked",
    comment="MedQuAD RAG data with long answers split into sentence-aware overlapping chunks."
)
def gold_medquad_chunked():

    df = spark.read.table("gold_medquad_selected")

    chunked_df = (
        df
        .withColumn(
            "_chunks",
            chunk_answer_udf(F.col("answer"))
        )
        .select(
            "*",
            F.posexplode(F.col("_chunks")).alias(
                "chunk_index",
                "answer_chunk"
            )
        )
    )

    return (
        chunked_df
        .withColumn(
            "chunk_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.coalesce(F.col("question_id").cast("string"), F.lit("")),
                    F.coalesce(F.col("question"), F.lit("")),
                    F.col("chunk_index").cast("string"),
                    F.coalesce(F.col("answer_chunk"), F.lit(""))
                ),
                256
            )
        )
        .drop(
            "answer",
            "_chunks"
        )
    )

# COMMAND ----------

@dp.materialized_view(
    name="gold_medquad_rag_chunks",
    comment="Final MedQuAD Gold table containing retrieval-ready text and metadata for RAG."
)
def gold_medquad_rag_chunks():

    df = spark.read.table("gold_medquad_chunked")

    return (
        df
        .withColumn(
            "retrieval_text",
            F.concat_ws(
                "\n\n",

                F.when(
                    F.col("question_focus").isNotNull(),
                    F.concat(
                        F.lit("Topic: "),
                        F.col("question_focus")
                    )
                ),

                F.concat(
                    F.lit("Question: "),
                    F.col("question")
                ),

                F.concat(
                    F.lit("Answer: "),
                    F.col("answer_chunk")
                )
            )
        )
        .select(
            "chunk_id",
            "qa_id",
            "chunk_index",
            "question_id",

            "retrieval_text",

            "question",
            "answer_chunk",
            "question_focus",
            "question_type",
            "synonyms",
            "category",

            "document_source",
            "document_url"
        )
    )