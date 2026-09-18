%md
# MedQuAD Data Quality Analysis and Cleaning Decisions

## Purpose of this notebook

This notebook documents the data-quality problems found while exploring the MedQuAD dataset, the decision taken for each problem, and the reasoning behind that decision.

The main goal is **not to clean the data as aggressively as possible**. The goal is to preserve useful medical information while preventing malformed, empty, misleading, or redundant content from entering the RAG-ready dataset.

The pipeline responsibilities are:

**Bronze**
- Preserve the raw source data.
- Do not permanently destroy information just because it is unusable for RAG.

**Silver**
- Normalize values.
- Remove or quarantine genuinely unusable question-answer pairs.
- Deduplicate exact question-answer duplicates.
- Keep useful metadata in a cleaner structure.

**Gold / RAG**
- Prepare retrieval documents.
- Chunk long answers.
- Add question/topic context to the text that will be embedded.
- Avoid unnecessary duplicate chunks.

---

## Dataset overview

During EDA we observed:

- Total rows: **47,441**
- Rows with missing answers: **31,034**
- Rows with real answers: **16,407**
- Columns:
  - `document_id`
  - `document_source`
  - `document_url`
  - `category`
  - `umls_cui`
  - `umls_semantic_types`
  - `umls_semantic_group`
  - `synonyms`
  - `question_id`
  - `question_focus`
  - `question_type`
  - `question`
  - `answer`

The dataset also uses the literal value `\N` in some metadata fields to represent a missing value.

%md
# 1. Missing answers

## Problem

A large part of MedQuAD contains a question but no usable answer.

I found:

- **31,034** rows with missing answers
- **16,407** rows with real answers

A row without an answer is not useful for a question-answering RAG system because there is no medical content to retrieve as an answer.

## Handling

**Bronze:** keep every raw row.

**Silver:** only answered rows are allowed into the clean QA dataset.

Rows with missing answers should either:
- remain only in Bronze, or
- be written to a rejected/quarantine table with a reason such as `MISSING_ANSWER`.

## Why

Bronze is the historical/raw layer, so deleting source rows there would make the ingestion layer incomplete.

Silver is the clean analytical/RAG layer. A missing answer contributes nothing to retrieval and could create empty chunks or meaningless embeddings.

**Decision:** missing answer = unusable for Silver QA/RAG, but preserved in Bronze.

%md
# 2. Missing optional metadata

## Problem

Some metadata fields contain:

- `NULL`
- empty strings
- the literal string `\N`

Examples include fields such as:
- `question_focus`
- UMLS metadata
- synonyms

## Handling

Normalize missing-like values to real Spark `NULL`.

For example:

```text
"\N"  -> NULL
""    -> NULL
```

Do **not** automatically reject a row just because optional metadata is missing.

Do **not** blindly replace missing values with `"Unknown"`.

## Why

`"Unknown"` is still a real string. If it is used everywhere, downstream code can incorrectly treat it as meaningful metadata.

`NULL` accurately represents the fact that the information was not provided.

A QA pair can still be useful if its question and answer are valid even when optional metadata is missing.

**Decision:** normalize missing metadata, but do not reject otherwise valid rows because of it.

%md
# 3. Exact duplicate question-answer pairs

## Problem

Full-row duplication was not a major problem, but we found repeated groups where the **same question and same answer** appeared more than once.

During EDA:

- Full-row duplicates: **0**
- Duplicate question-answer groups: **32**
- Roughly **80 rows** belonged to those duplicate QA groups

Most were associated with NIDDK data.

## Handling

When both normalized `question` and normalized `answer` are identical:

**keep one deterministic representative** in Silver.

The representative should be chosen deterministically, for example by ordering by a stable identifier such as `document_id` / `question_id`.

## Why

Repeating the exact same QA pair:

- adds no new information,
- increases storage,
- can create duplicate embeddings,
- can bias retrieval because the same semantic content appears multiple times.

The deduplication key should include **both question and answer**, not answer alone.

**Decision:** same question + same answer -> keep one.

%md
# 4. Same question, different answers

## Problem

Some rows use the same question wording but contain different answers.

This is not the same as a duplicate.

Two different sources, or two different records from the same source, may legitimately provide different useful information for the same question.

## Handling

Keep all **distinct answers** when the question is the same.

## Why

Example:

```text
Question: What are the treatments for Condition X?
Answer A: discusses medication.
Answer B: discusses surgery and supportive care.
```

Removing one of them would throw away potentially useful medical information.

**Decision:** same question + different answer -> keep all distinct answers.

%md
# 5. Low-quality or non-answer answers

## Problem

Some rows contain an `answer` value that is technically non-null but does not meaningfully answer the question.

Examples of possible problems:

- the answer only repeats the question,
- the answer contains only a heading,
- navigation/page text appears instead of an answer,
- the answer is related to the topic but does not answer the specific question,
- the answer only redirects the reader somewhere else.

A simple rule such as:

```text
answer_length < N
```

is not reliable.

Short answers can be valid, while long answers can still be irrelevant.

## Handling

We use a semantic LLM-based answer-quality classifier.

Main output:

```text
quality_status = VALID | INVALID
```

For invalid answers:

```text
issue_type =
    QUESTION_AS_ANSWER
    NON_INFORMATIVE
    IRRELEVANT
    FRAGMENT_OR_HEADING
```

`VALID` rows continue toward Silver.

`INVALID` rows are quarantined/rejected.

## Important classifier rules

The classifier is explicitly told:

- do not reject an answer merely because it is short,
- do not reject an answer merely because it contains a question,
- question-like headings are allowed if useful content follows,
- an answer can be valid when the requested information is unknown or unavailable,
- related information is not enough; the answer must respond to the actual question,
- for "what is" questions, risk factors/treatment/prevention alone are not enough unless the answer also explains the concept,
- the classifier is checking **answer relevance/quality**, not medical factual correctness.

## Why use an LLM instead of only hardcoded rules?

Hardcoded rules cannot reliably understand meaning.

For example:

```text
Question:
How many people are affected by Condition X?

Answer:
The incidence is unknown.
```

This answer is short and gives no number, but it still directly answers the question.

A semantic classifier can handle this better than a blocklist or minimum-length rule.

%md
## 5.1 Classifier evaluation

I did not simply trust the model immediately.

### Initial evaluation

A small manually reviewed prototype set contained **26 rows**.

Binary VALID/INVALID result:

- **26 / 26 correct**

This was useful, but too small to treat as strong evidence.

### Extended evaluation

I then created a separate sample of **112 new QA pairs** with no overlap with the original evaluation sample.

Human labels:

- **82 VALID**
- **30 INVALID**

Final classifier result:

- Total: **112**
- Correct: **109**
- Accuracy: **97.32%**

Confusion matrix:

```text
Human INVALID -> Model INVALID: 30
Human INVALID -> Model VALID:     0
Human VALID   -> Model VALID:    79
Human VALID   -> Model INVALID:   3
```

The important RAG-specific result is:

```text
known bad answers accepted = 0
useful answers rejected     = 3
```

## Why this error pattern is acceptable

For this project, allowing a bad answer into the retrieval index is more dangerous than losing a small number of useful rows.

A bad answer accepted into RAG can later be retrieved and supplied to the generation model.

A good answer rejected reduces recall slightly, but does not pollute the knowledge base.

I therefore froze the prompt instead of repeatedly tuning it until it memorized the validation sample.

**Decision:** classifier is good enough to use as a Silver quality gate, with rejected rows retained in quarantine for auditability.

%md
# 6. Malformed questions

## Problem

Some questions contain formatting problems. Others are genuinely missing the medical subject.

Initial suspicious-question exploration returned **324** rows.

After inspection:

- **322** had a duplicated question mark
- **2** were genuinely missing their subject

A later `question_focus` check found two more malformed forms, producing a final total of:

- **4 genuinely malformed questions**

Examples:

```text
Who is at risk for Lung Cancer? ?
```

versus:

```text
What is (are) ?
Who is at risk for ? ?
How to prevent ?
```

## Handling

### Formatting-only defect

```text
"Who is at risk for Lung Cancer? ?"
->
"Who is at risk for Lung Cancer?"
```

Keep the row and normalize the punctuation.

### Missing subject

Reject/quarantine the row.

## Why

The first case still preserves the full meaning of the question. Only punctuation is broken.

In the second case, the medical entity/topic is missing. We could guess it from the answer, but that would mean inventing source data.

For a clean data pipeline, we should not reconstruct missing source meaning unless there is a trusted deterministic source for it.

**Decision:** harmless punctuation -> clean; missing semantic subject -> reject.

%md
# 7. Missing `question_focus`

## Problem

I found **14 answered rows** where `question_focus` was missing.

Missing `question_focus` does not automatically mean that the actual question is unusable.

Examples of usable questions with missing focus included questions about:

- how vaccines prevent disease,
- vaccine-preventable diseases,
- vancomycin-resistant enterococci,
- VISA/VRSA.

Among the 14 rows:

- **10 had usable questions**

- **4 corresponded to genuinely malformed questions**

## Handling

If `question_focus` is missing but `question` itself is understandable:

- keep the row,
- leave `question_focus = NULL`.

If the question itself is malformed:

- reject because of the malformed question,
- not because `question_focus` is null.

## Why

`question_focus` is useful metadata, but it is not required for a QA pair to carry meaning.

We should not reject useful medical information simply because optional metadata is incomplete.

We also should not generate a replacement `question_focus` unless we explicitly decide to add a derived/enriched field later.

**Decision:** missing focus alone is not a rejection rule.

%md
# 8. Very long answers

## Problem

Answer-length EDA showed:

```text
minimum:       6 characters
average:   1,303.45
median:      890
p90:       2,447
p95:       3,398
p99:       8,336
maximum:  29,046
```

There were **166 answers longer than the 99th percentile**.

Inspection showed that many of these are large but legitimate medical pages containing detailed sections about treatments, diagnosis, disease types, risk factors, and follow-up.

## Handling

Do **not** reject an answer merely because it is long.

Silver keeps it if it passes the other quality rules.

Gold/RAG handles the length through **chunking**.

## Why

Length is primarily a retrieval-shaping problem, not a correctness problem.

A 20,000-character answer can contain excellent information.

But embedding the entire page as one vector is poor RAG design because:

- one vector must represent many unrelated sections,
- retrieval becomes less precise,
- the returned context can be much larger than necessary.

**Decision:** preserve long answers in Silver; split them into semantically manageable chunks in Gold.

%md
# 9. Repeated page-level and templated answers

## Problem

I grouped records by `document_source` and exact `answer` text and looked for answers reused by multiple distinct questions.

Results:

| Source | Repeated answer groups | Affected rows |
|---|---:|---:|
| GHR | 14 | 464 |
| GARD | 23 | 73 |
| CancerGov | 4 | 28 |
| NIDDK | 6 | 13 |
| CDC | 3 | 12 |
| NINDS | 2 | 4 |

Total affected rows: **594**

The largest GHR groups were strongly templated:

- autosomal recessive inheritance text reused for **348** different questions,
- autosomal dominant inheritance text reused for **70** different questions.

CDC showed a different pattern: several different questions could point to the same complete source-page answer.

## Handling

**Do not deduplicate Silver by `answer` alone.**

Preserve the question-answer relationship.

Later, in Gold:

- include question/topic context when constructing the text to embed,
- chunk long page-level answers,
- deduplicate identical chunks when doing so does not destroy question/topic context.

## Why

Consider:

```text
Question: How is Disease A inherited?
Answer: This condition is inherited in an autosomal recessive pattern...

Question: How is Disease B inherited?
Answer: This condition is inherited in an autosomal recessive pattern...
```

The answers are identical, but the medical topics are not.

If we deduplicate on `answer` alone, we lose one disease's QA relationship.

If we embed only the answer text, hundreds of GHR records become nearly indistinguishable.

A better Gold retrieval document includes contextual metadata:

```text
Topic: Disease A
Question: How is Disease A inherited?
Answer: This condition is inherited in an autosomal recessive pattern...
```

**Decision:** keep repeated answers when attached to different questions; solve retrieval duplication/context in Gold rather than deleting legitimate Silver rows.

%md
# 10. Multi-valued UMLS and synonym fields

## Problem

Some metadata columns contain multiple values inside a single string.

The delimiter is the pipe character:

```text
|
```

not a comma.

Examples:

```text
umls_cui:
C0017921|C0342751

umls_semantic_types:
T019|T047

synonyms:
acid maltase deficiency|acid maltase deficiency disease|Pompe's disease
```

Commas cannot safely be used as a delimiter because commas can appear inside one synonym:

```text
microphthalmia, isolated, with coloboma
```

Counts of multi-valued rows:

```text
umls_cui:             4,846
umls_semantic_types:  3,346
umls_semantic_group:      0
synonyms:             9,390
```

## Handling

Convert genuinely multi-valued fields into `ARRAY<STRING>`:

```text
umls_cui
"C0017921|C0342751"
->
["C0017921", "C0342751"]
```

```text
umls_semantic_types
"T019|T047"
->
["T019", "T047"]
```

```text
synonyms
"name one|name two|name three"
->
["name one", "name two", "name three"]
```

Keep `umls_semantic_group` as a normal string because no multi-valued rows were observed there.

Missing values should become `NULL`, not:

```text
["\N"]
```

## Why

Arrays represent the real structure of the data and make downstream operations easier:

- filtering by UMLS code,
- exploding synonyms,
- metadata filtering,
- constructing richer RAG documents.

**Decision:**

```text
umls_cui              -> ARRAY<STRING>
umls_semantic_types   -> ARRAY<STRING>
synonyms              -> ARRAY<STRING>
umls_semantic_group   -> STRING
```

%md
# 11. Final Silver decision matrix

| Data-quality issue | Silver handling | Reason |
|---|---|---|
| Missing answer | Reject / quarantine | No answer content for RAG |
| `\N`, blank optional metadata | Normalize to `NULL` | Represents missing data correctly |
| Same question + same answer | Keep one deterministic row | True duplicate |
| Same question + different answer | Keep all distinct answers | Potentially complementary information |
| Semantic low-quality answer | LLM classifier; INVALID -> quarantine | Meaning cannot be reliably detected with simple string rules |
| Short answer | Keep if semantically valid | Short does not mean bad |
| "Information is unknown" answer | Keep if it directly responds | Unknown can be the correct available answer |
| Duplicate `? ?` | Normalize to `?` | Formatting issue only |
| Missing question subject | Reject / quarantine | Semantic topic is absent |
| Missing `question_focus` | Keep if question is valid | Optional metadata |
| Very long answer | Keep | Length is a Gold chunking problem |
| Same answer across different questions | Keep | Question-answer relationship carries context |
| Pipe-delimited UMLS/synonyms | Convert to arrays | Represents multi-valued metadata correctly |

---

# 12. What belongs in Silver vs Gold?

A useful rule for this project:

## Silver asks:

**"Is this QA record clean, valid, and structurally usable?"**

Silver should:
- remove unusable rows,
- normalize data,
- deduplicate true duplicates,
- structure metadata.

## Gold asks:

**"How should this valid information be represented for retrieval?"**

Gold should:
- create RAG documents,
- combine question/topic/answer context,
- chunk long text,
- generate stable chunk IDs,
- potentially deduplicate identical chunks,
- prepare data for AI Search / Vector Search.

This separation prevents Silver from becoming over-aggressive.

We preserve valid source knowledge first, then optimize its retrieval representation later.

%md
# 13. Proposed Silver validation flow

Conceptually, a row should move through Silver like this:

```text
Raw Bronze row
      |
      v
Normalize missing values / whitespace
      |
      v
Is answer missing?
  yes -> quarantine
  no
      |
      v
Is question genuinely malformed?
  yes -> quarantine
  no
      |
      v
Normalize harmless question punctuation
      |
      v
Semantic answer-quality classifier
  INVALID -> quarantine
  VALID
      |
      v
Deduplicate exact normalized question + answer
      |
      v
Convert UMLS/synonym multi-values to arrays
      |
      v
Clean Silver QA table
```

This flow is intentionally conservative:

- reject only when there is a strong reason,
- preserve useful medical content,
- leave retrieval-specific transformations for Gold.

%md
# 14. Quarantine / rejected-row design

Rejected rows should not simply disappear.

A quarantine table makes the pipeline explainable and auditable.

Useful columns could include:

```text
document_id
question_id
question
answer
document_source
rejection_reason
classifier_issue_type
classifier_reason
rejected_at
```

Example rejection reasons:

```text
MISSING_ANSWER
MALFORMED_QUESTION
LOW_QUALITY_ANSWER
```

## Why keep quarantine?

It lets us:

- explain why rows were excluded,
- inspect classifier mistakes,
- change a cleaning rule later,
- reproduce the pipeline,
- show data-quality engineering clearly in the internship project.

%md
# 15. Final conclusion

The main principle used throughout this EDA is:

> **Do not confuse unusual data with bad data.**

A row should not be rejected merely because:

- the answer is short,
- the answer is very long,
- metadata is missing,
- another question has the same answer,
- the source uses awkward formatting.

We reject or quarantine data when the QA relationship itself is not usable:

- no answer,
- missing question subject,
- semantically non-responsive answer,
- true duplicate of the same normalized question and answer.

Everything else is either normalized in Silver or optimized later in Gold.

This gives the project a defensible pipeline:

```text
Bronze = preserve source truth
Silver = validate and normalize QA records
Gold   = optimize valid knowledge for retrieval
```
