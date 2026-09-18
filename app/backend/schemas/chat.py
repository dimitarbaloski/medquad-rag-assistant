from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Source(BaseModel):
    name: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
