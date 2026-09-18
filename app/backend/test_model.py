from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ChatMessage,
    ChatMessageRole,
)


workspace = WorkspaceClient()


response = workspace.serving_endpoints.query(
    name="databricks-meta-llama-3-1-8b-instruct",
    messages=[
        ChatMessage(
            role=ChatMessageRole.USER,
            content="Hello! Explain Databricks in one sentence.",
        )
    ],
    max_tokens=100,
)


print(
    response.choices[0].message.content
)