from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from retriever import (
    hierarchical_rag_search,
    gather_context,
    node_store
)
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,   
    model="gpt-4o-mini"  
)

class AgentState(TypedDict):
    query: str
    current_query: str
    context: str
    nodes: List
    steps: List[str]
    final_answer: str
    retrieval_score: float
    chat_history: List[str]

def history_node(state: AgentState):
    history = state.get("chat_history", [])
    query = state["query"]

    if not history:
        return {
            "current_query": query,
            "steps": state["steps"] + ["history_skipped"]
        }

    # Use last 2-3 turns only
    recent_history = "\n".join(history[-3:])

    prompt = f"""
You are a query rewriter for a conversational system.

Given the chat history and current question,
rewrite the question to be self-contained.

Chat History:
{recent_history}

Current Question:
{query}

Return only the rewritten standalone query.
"""

    new_query = llm.invoke(prompt).content.strip()

    return {
        "current_query": new_query,
        "steps": state["steps"] + ["history"]
    }

def decide_node(state: AgentState):
    prompt = f"""
Decide next action:

Options:
- retrieve
- rewrite
- answer
- stop

Rules:
- If no context → retrieve
- If retrieval_score < 0.5 → rewrite
- If good context → answer

Query: {state['current_query']}
Context length: {len(state['context'])}
Score: {state.get('retrieval_score', 0)}
Steps: {state['steps']}

Return one word.
"""

    decision = llm.invoke(prompt).content.strip().lower()

    return {"decision": decision}

def rewrite_node(state):
    query = state["current_query"]

    prompt = f"""
Rewrite the query to make it more precise and suitable for document retrieval.

Query:
{query}

Return only the improved query.
"""

    new_query = llm.invoke(prompt).content.strip()

    return {
        "current_query": new_query,
        "steps": state["steps"] + ["rewrite"]
    }

def retrieve_node(state: AgentState):
    result = retrieval_tool(state["current_query"])

    return {
        "context": result["context"],
        "nodes": result["nodes"],
        "retrieval_score": result.get("confidence", 0),
        "steps": state["steps"] + ["retrieve"]
    }

def retrieval_tool(query):
    nodes = hierarchical_rag_search(query)
    context = gather_context(nodes)
    max_score = max([score for _, score in nodes]) if nodes else 0

    return {
        "nodes": nodes,        
        "context": context,    
        "confidence": max_score
    }

def answer_node(state: AgentState):
    query = state["current_query"]
    context = state["context"]

    prompt = f"""
Answer the question using ONLY the context.

Context:
{context}

Question:
{query}
"""

    answer = llm.invoke(prompt).content

    # 🔹 Update chat history
    updated_history = state.get("chat_history", [])
    updated_history.append(f"User: {state['query']}")
    updated_history.append(f"Assistant: {answer}")

    return {
        "final_answer": answer,
        "chat_history": updated_history,
        "steps": state["steps"] + ["answer"]
    }

def route_decision(state: AgentState):
    decision = state["decision"]

    if decision == "retrieve":
        return "retrieve"
    elif decision == "rewrite":
        return "rewrite"
    elif decision == "answer":
        return "answer"
    else:
        return END

builder = StateGraph(AgentState)
builder.add_node("decide", decide_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("answer", answer_node)
builder.add_node("history", history_node)

builder.set_entry_point("history")
builder.add_edge("history", "decide")
builder.add_conditional_edges(
    "decide",
    route_decision,
    {
        "retrieve": "retrieve",
        "rewrite": "rewrite",
        "answer": "answer",
        END: END
    }
)

builder.add_edge("retrieve", "decide")
builder.add_edge("rewrite", "decide")
builder.add_edge("answer", END)

graph = builder.compile()
chat_history = []

def chat(query):
    global chat_history

    result = graph.invoke({
        "query": query,
        "current_query": query,
        "context": "",
        "nodes": [],
        "steps": [],
        "final_answer": "",
        "retrieval_score": 0,
        "chat_history": chat_history
    })

    chat_history = result["chat_history"]

    return result

result = chat("Explain probability")

print("\nAnswer:\n", result["final_answer"])
print("\nSteps:\n", result["steps"])