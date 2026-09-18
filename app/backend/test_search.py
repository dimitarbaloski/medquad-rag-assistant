from backend.services.chat_service import ChatService
from backend.schemas.chat import Message


service = ChatService()

question = "Does HSAN2 always cause breathing problems?"

print("USER QUESTION:")
print(question)
print("=" * 100)

messages = [
    Message(
        role="user",
        content=question,
    )
]

results = service.search_medquad(question)

print("\nSEARCH SCORES:")

for i, result in enumerate(results, start=1):
    print(
        i,
        result.get("question"),
        "SCORE:",
        result.get("score"),
    )

answer, sources = service.generate_response(messages)

print("\nGENERATED ANSWER:")
print(answer)

print("\nSOURCES:")

if not sources:
    print("No sources returned.")
else:
    for i, source in enumerate(sources, start=1):
        print(f"{i}. {source['name']}")
        print(f"   {source['url']}")

print("=" * 100)