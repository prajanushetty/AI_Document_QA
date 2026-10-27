"""Functions the LLM can call, plus the JSON schemas that describe them to the model."""

import ast
import json
import operator
from datetime import datetime

from rag import list_documents, search_documents


def calculator(expression: str) -> float:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))
        raise ValueError(f"Unsupported expression: {expression}")

    return evaluate(ast.parse(expression, mode="eval").body)


def get_current_datetime() -> str:
    return datetime.now().isoformat(timespec="seconds")


TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "list_documents": list_documents,
    "calculator": calculator,
    "get_current_datetime": get_current_datetime,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the uploaded documents and return the most relevant passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "top_k": {"type": "integer", "description": "Number of passages to return (default 4)."},
                    "source": {
                        "type": "string",
                        "description": "Only search this document (exact filename from list_documents).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the filenames of all uploaded documents.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression, e.g. '(12 * 4) / 3'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_tool(name: str, arguments: str):
    """Run a tool by name with JSON-encoded arguments. Errors are returned to the model, not raised."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOL_FUNCTIONS[name](**json.loads(arguments or "{}"))
    except Exception as e:
        return {"error": str(e)}
