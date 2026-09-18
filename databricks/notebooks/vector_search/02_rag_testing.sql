-- Databricks notebook source
WITH retrieved_chunks AS (

    SELECT
        answer_chunk,
        document_source,
        document_url
    FROM vector_search(
        index => 'dbacademy.medquad_project.medquad_rag_index',
        query_text => 'What medicines can help control asthma over the long term?',
        query_type => 'HYBRID',
        num_results => 5
    )

),

rag_data AS (

    SELECT
        concat_ws(
            '\n\n---\n\n',
            collect_list(
                concat(
                    'Source: ',
                    coalesce(document_source, 'Unknown'),
                    '\n',
                    'Content: ',
                    answer_chunk
                )
            )
        ) AS context_text,

        collect_set(
            concat(
                coalesce(document_source, 'Unknown'),
                ' | ',
                coalesce(document_url, '')
            )
        ) AS sources

    FROM retrieved_chunks

)

SELECT
    ai_query(
        'databricks-gpt-oss-20b',

        concat(
    'You are a medical information assistant.

Answer the user question using ONLY facts explicitly stated in the provided context.

STRICT RULES:
- Do not use outside knowledge.
- Do not add examples, medicine names, facts, or explanations unless they explicitly appear in the context.
- If you list specific medicines, include only medicine names that appear in the context.
- Do not guess or fill in missing information.
- If the context is insufficient, say: "The provided sources do not contain enough information to answer this question."
- Do not diagnose the user.
- Do not provide personalized medical advice.
- Do not tell the user what they personally should take or do.
- Summarize the supplied information clearly and concisely.

CONTEXT:

',
    context_text,

    '

USER QUESTION:
What medicines can help control asthma over the long term?'
)
    ) AS generated_answer,

    sources

FROM rag_data;