# Gold Layer Column Selection

This document defines which columns from `silver_medquad` will be carried into the Gold RAG layer, which columns may be useful in a future version, and why each column is kept.

## Column Selection

| Column | Usage | Reason | Short Example |
|---|---|---|---|
| `question_id` | **USE NOW** | Provides a stable identifier for the original QA pair and lets Gold chunks be traced back to the source question. | If a retrieved chunk comes from `question_id = Q42`, we can inspect the original QA record. |
| `question` | **USE NOW** | Core retrieval context. User queries will often be semantically similar to the original MedQuAD question. | User asks “What are the symptoms of asthma?” and it matches a similar stored question. |
| `answer` | **USE NOW** | Contains the medical information that will be prepared and chunked for RAG retrieval. | A long asthma answer can be split into multiple smaller chunks. |
| `question_focus` | **USE NOW** | Identifies the main medical topic or entity discussed in the question. | `question_focus = "Asthma"` |
| `question_type` | **USE NOW** | Describes the intent of the question, such as treatment, symptoms, diagnosis, prevention, or risk. Useful for evaluation and possible later filtering. | `question_type = "Treatment"` |
| `synonyms` | **USE NOW** | Helps retrieval when users use an alternative name for the same medical concept. | A user searches with a synonym instead of the exact disease name used in the question. |
| `category` | **USE NOW** | Provides a high-level topic classification that can be used for analysis or optional filtering later. | A record can belong to a category such as cancer or genetic conditions. |
| `document_source` | **USE NOW** | Preserves the organization/source of the medical information and can be shown with chatbot citations. | `Source: CDC` or `Source: NIDDK` |
| `document_url` | **USE NOW** | Provides the original reference page and can be returned to the frontend as a clickable source. | The chatbot can show a link to the original CDC page. |
| `document_id` | **FUTURE / OPTIONAL** | Provides document-level lineage when several QA pairs originate from the same MedQuAD document. Not required for the first RAG version because `question_id` already provides sufficient QA-level traceability. | Several QA pairs may come from `document_id = 123`. |
| `umls_cui` | **FUTURE / OPTIONAL** | Standard UMLS concept identifiers could later support medical concept linking or concept-based filtering. | Different disease names can map to the same UMLS concept identifier. |
| `umls_semantic_types` | **FUTURE / OPTIONAL** | Could support more advanced filtering or evaluation based on the type of medical concept. | Restrict results to disease-related concepts instead of drug-related concepts. |
| `umls_semantic_group` | **FUTURE / OPTIONAL** | Provides a higher-level UMLS grouping that could be useful for broader semantic filtering. | Search only within disorder-related records. |

## Columns Not Planned for Gold

The following columns remain useful for Bronze/Silver processing and auditing, but they are not needed for the RAG Gold layer:

- `source_file` — ingestion lineage only.
- `ingestion_timestamp` — pipeline auditing information.
- `quality_status` — every row entering Gold has already passed Silver quality validation.
- `quality_issue_type` — relevant only to rejected/classified records.
- `quality_reason` — useful for classifier auditing, not retrieval.

## Final Columns Used in the First Gold Version

```text
question_id
question
answer
question_focus
question_type
synonyms
category
document_source
document_url
```

## Possible Future Metadata

```text
document_id
umls_cui
umls_semantic_types
umls_semantic_group
```

## New Columns Created in Gold

These columns do not currently exist in `silver_medquad`. They will be created during Gold RAG preparation.

| Column | Purpose |
|---|---|
| `chunk_id` | Unique identifier for each retrievable chunk. |
| `chunk_index` | Stores the position of a chunk within the original answer. |
| `answer_chunk` | Contains the smaller answer section used for retrieval. |
| `retrieval_text` | Contains the final text prepared for embedding/search, for example the question together with an answer chunk. |
