import json
import os
import mlflow

from databricks.sdk import WorkspaceClient

from databricks.sdk.service.serving import (
    ChatMessage,
    ChatMessageRole,
)

from backend.schemas.chat import Message



mlflow.set_tracking_uri("databricks")

mlflow.set_experiment(
    os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "/Shared/medquad-rag-traces"
    )
)

class ChatService:

    def __init__(self):
        self.workspace = WorkspaceClient()

        self.model_endpoint = (
            "databricks-qwen3-next-80b-a3b-instruct"
        )

        self.index_name = os.environ.get(
            "MEDQUAD_INDEX_NAME",
            "dbacademy.medquad_project.medquad_rag_index",
        )

        self.top_k = int(
            os.environ.get(
                "RAG_TOP_K",
                "5",
            )
        )

    @mlflow.trace(
    name="prepare_question",
    span_type="CHAIN",
    )
    def prepare_question(
        self,
        messages: list[Message],
        user_question: str,
    ) -> tuple[str, str, list[str]]:

        # Use only previous user messages.
        # The assistant may have mentioned conditions that
        # the user did not actually ask about.
        recent_messages = messages[:-1][-4:]

        conversation_history = "\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in recent_messages
        )

        # print()
        # print("PREVIOUS USER MESSAGES:")
        # print(conversation_history)
        # print("LATEST QUESTION:")
        # print(user_question)

        prompt = f"""
Prepare the latest question for a medical knowledge-base search.

1. Resolve conversation references:
- First use a condition, symptom, or medical topic explicitly named in
  the latest question.
- Otherwise, use the main medical topic from the recent conversation.
- Ignore older or unrelated topics.
- If the reference cannot be resolved, classify the question as
  AMBIGUOUS.

2. Classify the resolved question:
- READY: a clear medical or healthcare question. Questions about
  symptoms or medical concerns are READY even without a disease name.
- AMBIGUOUS: the question is incomplete or contains an unresolved
  reference.
- OUT_OF_SCOPE: the question is not medical or healthcare-related.

3. For a READY question:
- Rewrite it as a standalone question.
- For multiple requested aspects, create one focused query per aspect.
- For a single aspect, keep one query.
- Preserve the exact condition, symptom, topic, and requested aspects.
- Do not introduce related conditions, specific subtypes, or additional
  aspects that the user did not request.

Include one short decision_reason explaining why the selected status applies.
Do not provide step-by-step reasoning.
Return no more than 4 queries.
For AMBIGUOUS or OUT_OF_SCOPE, return an empty queries list.
Do not answer the question.
Return only valid JSON without Markdown or explanation:

{{
  "status": "READY",
  "decision_reason": "The question clearly names a medical topic.",
  "question": "Standalone question",
  "queries": [
    "Search query"
  ]
}}

RECENT CONVERSATION:

{conversation_history or "None"}

LATEST QUESTION:

{user_question}
""".strip()

        try:
            with mlflow.start_span(
                name="question_classification",
                span_type="CHAT_MODEL",
            ) as span:

                span.set_inputs({
                    "classification_prompt": prompt
                })

                response = (
                    self.workspace
                    .serving_endpoints
                    .query(
                        name=self.model_endpoint,
                        messages=[
                            ChatMessage(
                                role=ChatMessageRole.USER,
                                content=prompt,
                            )
                        ],
                        max_tokens=250,
                        temperature=0,
                    )
                )

                content = (
                    response
                    .choices[0]
                    .message
                    .content
                    .strip()
                )

                span.set_outputs({
                    "raw_model_response": content
                })

            # print("RAW PREPARATION RESPONSE:")
            # print(content)

            # Handle JSON wrapped in a Markdown code block.
            if content.startswith("```"):
                content = content.strip("`").strip()

                if content.lower().startswith("json"):
                    content = content[4:].strip()

            parsed = json.loads(content)

            # print("PARSED PREPARATION RESPONSE:")
            # print(parsed)
            # print()

            if not isinstance(parsed, dict):
                return (
                    "READY",
                    user_question,
                    [user_question],
                )

            status = str(
                parsed.get("status", "READY")
            ).strip().upper()

            standalone_question = str(
                parsed.get(
                    "question",
                    user_question,
                )
            ).strip()

            queries = parsed.get(
                "queries",
                [],
            )

            if not isinstance(queries, list):
                queries = []

            queries = [
                query.strip()
                for query in queries
                if (
                    isinstance(query, str)
                    and query.strip()
                )
            ][:4]

            if status not in {
                "READY",
                "AMBIGUOUS",
                "OUT_OF_SCOPE",
            }:
                status = "READY"

            if not standalone_question:
                standalone_question = user_question

            if status == "READY" and not queries:
                queries = [standalone_question]

            if status != "READY":
                queries = []

            return (
                status,
                standalone_question,
                queries,
            )

        except Exception as error:
            print(
                f"Question preparation failed: {error}"
            )

            # Preserve the original behavior if preparation fails.
            return (
                "READY",
                user_question,
                [user_question],
            )

    @mlflow.trace(
    name="vector_search",
    span_type="RETRIEVER",
    )
    def search_medquad(
        self,
        query: str,
        num_results: int | None = None,
    ) -> list[dict]:

        result_limit = (
            num_results
            if num_results is not None
            else self.top_k
        )

        response = (
            self.workspace
            .vector_search_indexes
            .query_index(
                index_name=self.index_name,
                query_text=query,
                query_type="HYBRID",
                num_results=result_limit,
                columns=[
                    "chunk_id",
                    "retrieval_text",
                    "question",
                    "answer_chunk",
                    "question_focus",
                    "question_type",
                    "category",
                    "document_source",
                    "document_url",
                ],
            )
        )

        if (
            response.manifest is None
            or response.result is None
            or not response.result.data_array
        ):
            return []

        column_names = [
            column.name
            for column in response.manifest.columns
        ]


        results = []

        for row in response.result.data_array:

            result = dict(
                zip(
                    column_names,
                    row,
                )
            )

            results.append(result)

        return results

    @mlflow.trace(
        name="medquad_chat_request",
        span_type="CHAIN",
    )
    def generate_response(
        self,
        messages: list[Message],
    ) -> tuple[str, list[dict]]:

        # print()
        # print("RECEIVED MESSAGES:")

        # for message in messages:
        #     print(
        #         {
        #             "role": message.role,
        #             "content": message.content,
        #         }
        #     )

        user_question = next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "user"
            ),
            None,
        )

        if not user_question or not user_question.strip():
            return (
                "No user question was provided.",
                [],
            )

        user_question = user_question.strip()

        # 1. Resolve conversation references, classify the
        # question, and prepare search queries.

        # First analyze the latest question without conversation history.
        current_message = Message(
            role="user",
            content=user_question,
        )

        (
            status,
            standalone_question,
            search_queries,
        ) = self.prepare_question(
            messages=[current_message],
            user_question=user_question,
        )

        # Use conversation history only when the latest question
        # cannot be understood independently.
        if status == "AMBIGUOUS":

            (
                status,
                standalone_question,
                search_queries,
            ) = self.prepare_question(
                messages=messages,
                user_question=user_question,
            )

        if status == "AMBIGUOUS":
            return (
                "Please specify which medical condition you mean.",
                [],
            )

        if status == "OUT_OF_SCOPE":
            return (
                "This assistant can only answer medical and "
                "healthcare-related questions.",
                [],
            )

        # For a single-part question, search using the resolved standalone
        # question so the model cannot replace it with a subtype.
        if len(search_queries) == 1:
            search_queries = [
                standalone_question
            ]

        # 2. Retrieve relevant chunks for every search query.
        search_results = []
        seen_chunk_ids = set()

        results_per_query = (
            3
            if len(search_queries) > 1
            else self.top_k
        )

        for search_query in search_queries:

            query_results = self.search_medquad(
                search_query,
                num_results=results_per_query
            )

            for result in query_results:

                chunk_id = result.get("chunk_id")

                # The same chunk can be returned for more
                # than one search query.
                if (
                    chunk_id
                    and chunk_id in seen_chunk_ids
                ):
                    continue

                if chunk_id:
                    seen_chunk_ids.add(chunk_id)

                result["matched_query"] = (
                    search_query
                )

                search_results.append(result)

        if not search_results:
            return (
                "The provided sources do not contain enough "
                "information to answer this question.",
                [],
            )

        # 3. Build context.
        context_parts = []

        for result in search_results:

            context_parts.append(
                f"""
            Information requested: {
                result.get(
                    "matched_query",
                    standalone_question,
                )
            }
            Topic: {result.get("question_focus", "")}
            Question: {result.get("question", "")}
            Content: {result.get("answer_chunk", "")}
            """.strip()
                        )

        context = "\n\n---\n\n".join(
            context_parts
        )

        # 4. Collect unique sources.
        sources = []
        seen_sources = set()

        for result in search_results:

            source_name = result.get(
                "document_source"
            )

            source_url = result.get(
                "document_url"
            )

            if not source_name or not source_url:
                continue

            source_key = (
                source_name,
                source_url,
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            sources.append(
                {
                    "name": source_name,
                    "url": source_url,
                }
            )

        # 5. Grounding prompt.
        system_prompt = """
You are a medical information assistant.

Answer the user's question using ONLY facts explicitly stated
in the provided context.

STRICT RULES:
- Do not use outside knowledge.
- Do not add medical facts, medicine names, examples, or explanations
  unless they are supported by the provided context.
- Do not guess or fill in missing information.
- You may make minimal logical conclusions directly supported by
  the context.
- Preserve qualifiers such as "some", "may", "can", "often",
  "typically", and "variable".
- Do not turn a possibility into a certainty.
- Do not turn a partial statement into a universal statement.
- Do not diagnose the user.
- Do not provide personalized medical advice.
- Do not provide personalized medication dosages.
- If the question contains multiple parts, answer each supported part
  separately.
- If only some parts are supported, answer those parts and identify
  which parts are unsupported.
- Do not reject the entire question when at least one part is supported.
- If none of the requested information is supported, say:
  "The provided sources do not contain enough information to answer
  this question."
  - Answer only about the exact medical condition requested by the user.
- Do not combine different conditions merely because they have similar
  names or share the same term.
- Do not replace a general condition with a specific subtype.
- Ignore context passages about a different condition or specific subtype
  unless the user explicitly requested it.
- Answer clearly and concisely.
""".strip()

        user_prompt = f"""
CONTEXT:

{context}

USER QUESTION:

{standalone_question}
""".strip()

        # 6. Generate the grounded answer.
        with mlflow.start_span(
            name="qwen_generation",
            span_type="CHAT_MODEL",
        ) as span:

            span.set_inputs(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )

            response = (
                self.workspace
                .serving_endpoints
                .query(
                    name=self.model_endpoint,
                    messages=[
                        ChatMessage(
                            role=ChatMessageRole.SYSTEM,
                            content=system_prompt,
                        ),
                        ChatMessage(
                            role=ChatMessageRole.USER,
                            content=user_prompt,
                        ),
                    ],
                    max_tokens=800,
                    temperature=0,
                )
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

            span.set_outputs(
                {
                    "answer": answer,
                }
            )

        return answer, sources

    # def convert_role(
    #     self,
    #     role: str,
    # ) -> ChatMessageRole:

    #     if role == "assistant":
    #         return ChatMessageRole.ASSISTANT

    #     if role == "system":
    #         return ChatMessageRole.SYSTEM

    #     return ChatMessageRole.USER