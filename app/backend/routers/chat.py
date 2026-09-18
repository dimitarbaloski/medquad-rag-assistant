from fastapi import APIRouter, HTTPException

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.chat_service import ChatService


router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
)

chat_service = ChatService()


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    try:
        answer, sources = chat_service.generate_response(request.messages)

        return ChatResponse(
            answer=answer,
            sources=sources,)

    except Exception as error:

        print(error)

        raise HTTPException(
            status_code=500,
            detail="Failed to generate response.",
        )