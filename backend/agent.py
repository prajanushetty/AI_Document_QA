import json
from typing import List

from rag import CHAT_MODEL, get_llm_client
from tools import TOOLS, run_tool

MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the user's uploaded documents. "
    "Use search_documents to find information before answering questions about the documents, "
    "and cite the document names you used. If the documents don't contain the answer, say so. "
    "Use the calculator for arithmetic instead of computing it yourself."
)


def answer_question(question: str) -> dict:
    """Run the tool-calling loop until the model replies without requesting a tool."""
    client = get_llm_client()
    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps = []
    sources = []

    for _ in range(MAX_TOOL_ROUNDS):
        message = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOLS, temperature=0
        ).choices[0].message

        if not message.tool_calls:
            return {"answer": message.content, "sources": sources, "steps": steps}

        # Send the assistant turn back unchanged; Gemini needs its extra fields (thought signatures).
        assistant_turn = message.model_dump(exclude_none=True)
        assistant_turn.setdefault("content", "")
        messages.append(assistant_turn)

        for call in message.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            steps.append({"tool": call.function.name, "arguments": json.loads(call.function.arguments or "{}")})
            if call.function.name == "search_documents" and isinstance(result, list):
                sources.extend(r["source"] for r in result if r["source"] not in sources)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

    return {
        "answer": "I couldn't finish answering within the tool-call limit. Try a more specific question.",
        "sources": sources,
        "steps": steps,
    }
