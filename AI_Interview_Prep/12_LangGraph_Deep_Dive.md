# LangGraph: The Definitive Interview Guide (2025-2026)
## Agent Orchestration Framework for Reliable AI Agents

---

# TABLE OF CONTENTS

1. [What is LangGraph?](#section-1)
2. [Explaining to a Layman](#section-2)
3. [Core Concepts](#section-3)
4. [Graph Architecture & Design Patterns](#section-4)
5. [State Management Deep Dive](#section-5)
6. [Human-in-the-Loop](#section-6)
7. [Multi-Agent Patterns](#section-7)
8. [Streaming](#section-8)
9. [Memory & Persistence](#section-9)
10. [Error Handling & Reliability](#section-10)
11. [LangGraph vs CrewAI vs AutoGen](#section-11)
12. [Interview Questions (25+)](#section-12)
13. [Follow-up Questions](#section-13)
14. [Complete Code Examples](#section-14)
15. [Production Best Practices](#section-15)

---

# SECTION 1: WHAT IS LANGGRAPH?

## 1.1 Definition

**LangGraph** is a Python framework for building **stateful, multi-actor applications** with LLMs, using graph-based orchestration. Built by LangChain Inc., it models agent workflows as **directed graphs** where:
- **Nodes** = functions/agents that process state
- **Edges** = transitions between nodes (can be conditional)
- **State** = shared data that flows through the graph

**Why LangGraph exists:**
- LangChain chains are linear (A → B → C) - not enough for complex agent workflows
- Real agent workflows need **cycles** (retry loops), **branching** (conditional logic), **persistence** (save/resume), and **human-in-the-loop** (pause for approval)

**Key Features (2025-2026):**

| Feature | Description |
|---------|-------------|
| **Cycles & Branching** | Create loops, conditional paths, parallel execution |
| **Persistence** | Save state to SQLite, PostgreSQL, or custom backends |
| **Human-in-the-Loop** | Pause execution, wait for human input, resume |
| **Streaming** | Token streaming, event streaming, state updates |
| **Subgraphs** | Compose complex workflows from reusable sub-graphs |
| **Time Travel** | Replay from any checkpoint, fork state |
| **Tool Calling** | Native integration with LLM tool/function calling |
| **Multi-Agent** | Supervisor, hierarchical, and collaborative patterns |

> 🔵 **YOUR EXPERIENCE**: At RavianAI, you use LangGraph alongside Autogen for your agentic AI platform. You designed end-to-end agentic architecture from intent recognition to task execution using graph-based orchestration.

---

# SECTION 2: EXPLAINING TO A LAYMAN

## The Factory Assembly Line Analogy

> Imagine a smart factory where each worker station does one job. A car frame arrives, goes to painting, then engine, then quality check. But unlike a simple line, the factory manager can:
> - Send the car BACK to painting if quality check fails (cycles/loops)
> - Send SUVs to one path and sedans to another (conditional routing)
> - Pause and ask a supervisor "Should we use red paint?" (human-in-the-loop)
> - Save the car state so if power goes out, work resumes (persistence)
>
> LangGraph is that smart factory for AI workflows.

---

# SECTION 3: CORE CONCEPTS

## 3.1 StateGraph

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage
import operator

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # append-only
    query: str
    documents: List[str]
    final_answer: str

# 2. Create graph
graph = StateGraph(AgentState)

# 3. Add nodes
def retrieve(state: AgentState) -> dict:
    docs = search(state["query"])
    return {"documents": docs}

def generate(state: AgentState) -> dict:
    answer = llm.invoke(state["documents"])
    return {"final_answer": answer}

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

# 4. Add edges
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# 5. Compile and run
app = graph.compile()
result = app.invoke({"query": "What is RAG?", "messages": [], "documents": [], "final_answer": ""})
```

## 3.2 Nodes

Nodes are Python functions that receive state, do work, return state updates:

```python
def my_node(state: AgentState) -> dict:
    query = state["query"]       # Read from state
    result = process(query)       # Do work
    return {"final_answer": result}  # Return only changed fields
```

## 3.3 Edges

| Edge Type | Description | Example |
|-----------|-------------|---------|
| **Normal** | Always go A to B | `graph.add_edge("A", "B")` |
| **Conditional** | Route based on state | `graph.add_conditional_edges("A", router_fn, {...})` |
| **Entry** | Where graph starts | `graph.add_edge(START, "first_node")` |

```python
def should_continue(state: AgentState) -> str:
    if state["relevance_score"] > 0.8:
        return "generate"
    elif state["retry_count"] < 3:
        return "retry"
    else:
        return "fallback"

graph.add_conditional_edges("evaluate", should_continue, {
    "generate": "generate",
    "retry": "retrieve",
    "fallback": "web_search"
})
```

## 3.4 State Annotations & Reducers

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]    # Append-only
    query: str                                  # Replace on update
    documents: Annotated[list, lambda o, n: list(set(o + n))]  # Deduplicate
    step_count: Annotated[int, operator.add]   # Increment
```

---

# SECTION 4: GRAPH ARCHITECTURE PATTERNS

## 4.1 Linear (Simple Pipeline)
```
START → Retrieve → Generate → END
```

## 4.2 Router (Branching)
```
                  ┌── Vector Search ──┐
START → Router ───┼── SQL Query ──────┼── Generate → END
                  └── Web Search ─────┘
```

## 4.3 ReAct Agent (Cyclic)
```
START → Agent ── Tool Call ── Agent ── Tool Call ── ... ── END
         │                     │
         └── (no tool needed) ─┘── Generate → END
```

```python
from langgraph.prebuilt import create_react_agent

tools = [search_tool, calculator_tool, code_executor]
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=tools,
    state_modifier="You are a helpful assistant."
)
```

## 4.4 Corrective Loop (Self-Correcting)
```
START → Generate → Evaluate ── (good) ── END
                      │
                      └── (bad) → Critique → Regenerate → ...
```

> 🔵 **YOUR EXPERIENCE**: At RavianAI, your asynchronous coordination protocols and intelligent routing map to Router + Cyclic patterns. Your error-handling fallback strategy maps to Corrective Loop.

---

# SECTION 5: STATE MANAGEMENT

## 5.1 Checkpointing (Persistence)

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# Development
memory = SqliteSaver.from_conn_string(":memory:")
# Production
# memory = PostgresSaver.from_conn_string("postgresql://user:pass@host/db")

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"query": "Hello"}, config)
# Continue same conversation
result = app.invoke({"query": "Follow up"}, config)
```

## 5.2 Time Travel

```python
checkpoints = list(memory.list(config))
state_at_step_3 = app.get_state(config, checkpoint_id=checkpoints[2].id)
app.update_state(config, {"query": "Modified"}, checkpoint_id=checkpoints[1].id)
```

> 🔵 **YOUR EXPERIENCE**: At RavianAI, you developed context persistence and memory management combining short-term conversational memory with structured task state tracking - maps directly to LangGraph checkpointing.

---

# SECTION 6: HUMAN-IN-THE-LOOP

```python
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["execute"]  # Pause before execution
)

result = app.invoke({"task": "Send email"}, config)
# Paused - human reviews
app.update_state(config, {"approved": True})
result = app.invoke(None, config)  # Resume
```

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Approval Gate** | Before dangerous actions | `interrupt_before=["action"]` |
| **Edit & Continue** | Human corrects plan | `update_state()` then resume |
| **Input Required** | Agent needs info | Interrupt, collect, resume |
| **Review Loop** | Quality check | Interrupt after generation |

---

# SECTION 7: MULTI-AGENT PATTERNS

## 7.1 Supervisor Pattern

```python
def supervisor(state):
    response = supervisor_llm.invoke(
        f"Which agent next? researcher, coder, or FINISH? Context: {state['messages']}"
    )
    return {"next_agent": response.content}

graph.add_conditional_edges("supervisor", lambda s: s["next_agent"], {
    "researcher": "researcher", "coder": "coder", "FINISH": END
})
```

## 7.2 Hierarchical Pattern
```
                    ┌──────────────┐
                    │  CEO Agent    │
                    └──────┬───────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        Engineering     Sales      Marketing
         Manager        Lead        Manager
         ┌─┼─┐                     ┌─┼─┐
         ▼ ▼ ▼                     ▼ ▼ ▼
       Dev QA DevOps           Content SEO Design
```

## 7.3 Swarm/Handoff Pattern

```python
def agent_handoff(state):
    response = agent_llm.invoke_with_tools(
        state["messages"],
        tools=[transfer_to_billing, transfer_to_support]
    )
    if response.tool_calls:
        return {"next_agent": response.tool_calls[0]["name"]}
    return {"messages": [response], "next_agent": "end"}
```

> 🔵 **YOUR EXPERIENCE**: At RavianAI, you built specialized agents for email, code ops, scheduling, and orchestration - this is the Supervisor + specialized workers pattern.

---

# SECTION 8: STREAMING

| Mode | What Streams | Use Case |
|------|-------------|----------|
| **values** | Full state per node | Debug |
| **updates** | Delta per node | Monitoring |
| **messages** | LLM tokens | Chat UI |
| **events** | All events | Advanced debug |

```python
async for event in app.astream_events(
    {"query": "Explain RAG"}, config, version="v2"
):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

> 🔵 **YOUR EXPERIENCE**: At RavianAI, you built WebSocket support for real-time agent communication - LangGraph streaming integrates perfectly with WebSocket frontends.

---

# SECTION 9: MEMORY & PERSISTENCE

| Memory Type | Scope | Feature | Example |
|-------------|-------|---------|---------|
| **Short-term** | Single conversation | `messages` in state | Chat history |
| **Thread** | Across turns | Checkpointer | Multi-turn chat |
| **Cross-thread** | All threads | Store | User preferences |
| **Long-term** | Persistent | External store | Vector DB |

---

# SECTION 10: ERROR HANDLING

```python
# 1. Recursion limits
result = app.invoke(input, config={"recursion_limit": 25})

# 2. Retry in nodes
def reliable_node(state):
    for attempt in range(3):
        try:
            return {"result": llm.invoke(state["messages"])}
        except Exception:
            if attempt == 2: return {"error": True}
            time.sleep(2 ** attempt)

# 3. Fallback edges
graph.add_conditional_edges("risky", lambda s: "fallback" if s.get("error") else "next")
```

> 🔵 **YOUR EXPERIENCE**: Your robust error-handling and fallback strategy at RavianAI maps directly to LangGraph fallback patterns.

---

# SECTION 11: LANGGRAPH vs CREWAI vs AUTOGEN

| Feature | LangGraph | CrewAI | AutoGen (AG2) |
|---------|-----------|--------|---------------|
| **Paradigm** | Graph-based | Role-based crew | Conversational |
| **State** | First-class TypedDict | Task-based | Message history |
| **Persistence** | Built-in (SQLite/Postgres) | Limited | Custom |
| **HITL** | Native interrupt | Basic | UserProxyAgent |
| **Streaming** | Token + state + events | Basic | Basic |
| **Flexibility** | Very High | Medium | High |
| **Learning Curve** | Medium-High | Low | Medium |
| **Best For** | Complex custom workflows | Quick prototypes | Research/experimentation |

> 🔵 **YOUR EXPERIENCE**: You use both LangGraph AND AutoGen at RavianAI, and contributed to AG2AI open source - uniquely qualified to compare.

---

# SECTION 12: INTERVIEW QUESTIONS (25+)

**Q1: What is LangGraph and how does it differ from LangChain?**
LangGraph is a graph-based orchestration framework for stateful, multi-actor LLM applications. LangChain provides building blocks (chains, prompts, tools); LangGraph provides control flow with cycles, branching, persistence, and HITL.

**Q2: Core components of a LangGraph app?**
(1) State - TypedDict with shared data, (2) Nodes - functions processing state, (3) Edges - connections (normal, conditional, entry). Compiles into runnable app.

**Q3: How do conditional edges work?**
A function examines state and returns a string key. The key maps to the next node. Used for routing, retry logic, agent decisions.

**Q4: Explain state reducers.**
Control how state updates merge. `operator.add` appends lists. Default replaces values. Custom lambdas for dedup, max, etc.

**Q5: What is checkpointing?**
Persisting state after every node. Enables: resume after failure, time travel, HITL, multi-turn conversations.

**Q6: Describe the Supervisor pattern.**
Central supervisor decides which worker agent to invoke based on conversation state. Workers return control to supervisor. Supervisor routes or ends.

**Q7: How do subgraphs work?**
Compiled graphs used as nodes in parent graphs. For encapsulation, reuse, state isolation.

**Q8: How does streaming work?**
Four modes: values (full state), updates (delta), messages (LLM tokens), events (all). Use `astream_events` for real-time chat.

**Q9: What is time travel?**
List checkpoints, inspect state at any point, fork from previous state. For debugging and undo.

**Q10: How to handle errors?**
Try/except in nodes, conditional edges for error routing, recursion limits, checkpoint recovery.

**Q11: How to implement Corrective RAG?**
Retrieve → Grade relevance → if good: generate; if bad: rewrite query → re-retrieve; if exhausted: web search fallback.

**Q12: LangGraph vs simple while loop?**
LangGraph adds persistence, observability (LangSmith), HITL, streaming, type safety. While loop has none.

**Q13: How to deploy in production?**
LangGraph Platform (managed), self-hosted FastAPI, or serverless. Use PostgresSaver, LangSmith for monitoring.

**Q14: How to handle concurrent users?**
Unique thread_id per user. Checkpointer handles concurrent writes. Double-texting policies for rapid inputs.

**Q15: How to test LangGraph apps?**
Unit test nodes, integration test compiled graph with mock LLMs, LangSmith datasets for eval, edge case testing.

**Q16: What is the Send API?**
Dynamic fan-out for map-reduce. Send state to multiple node instances in parallel.

**Q17: How to manage token costs?**
Smaller models for routing, summarize long histories, cache LLM calls, monitor per-node with LangSmith, recursion limits.

**Q18: How to integrate with FastAPI WebSocket?**
Compile with checkpointer, invoke in WebSocket handler with thread_id, stream events back via WebSocket, handle interrupts.

**Q19: What is LangGraph Platform?**
Managed deployment: API server, scaling, monitoring, Studio debugger, cron jobs, Assistants API.

**Q20: How to handle long-running tasks?**
Checkpoints survive restarts, interrupt_before for human checkpoints, timeout logic, background task patterns.

**Q21: Explain create_react_agent.**
Prebuilt ReAct agent: LLM with tools → tool call → execute → feed back → repeat until done. Configurable model, tools, system message.

**Q22: How does state schema evolution work?**
New fields need defaults. Existing checkpoints use defaults. Breaking changes need migration.

**Q23: How to implement multi-source RAG agent?**
Agent with tools for vector search, SQL, web search. Routes based on query analysis. Can query multiple in parallel.

**Q24: What is the difference between interrupt_before and interrupt_after?**
interrupt_before: pause BEFORE node (approval gate). interrupt_after: pause AFTER (review).

**Q25: How to debug LangGraph workflows?**
LangSmith tracing for every node, LangGraph Studio visual debugger, time travel to inspect any state, event streaming for real-time monitoring.

---

# SECTION 13: FOLLOW-UP QUESTIONS

| After You Say... | They Ask... |
|-------------------|-------------|
| "We use LangGraph" | "Why not CrewAI?" |
| "We persist to Postgres" | "How do you handle schema migrations?" |
| "We have HITL" | "What's the UX? How long can it wait?" |
| "We use supervisor pattern" | "What if supervisor hallucinates?" |
| "We stream tokens" | "Through WebSockets? Disconnection handling?" |

---

# SECTION 14: COMPLETE CODE EXAMPLE

```python
"""Multi-Agent Research Assistant with LangGraph - Supervisor Pattern"""
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class ResearchState(TypedDict):
    messages: Annotated[list, operator.add]
    topic: str
    notes: str
    draft: str
    feedback: str
    revision_count: int
    next_agent: str

supervisor_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def supervisor(state):
    if not state.get("notes"): return {"next_agent": "researcher"}
    elif not state.get("draft"): return {"next_agent": "writer"}
    elif state.get("feedback") and state["revision_count"] < 2: return {"next_agent": "writer"}
    elif not state.get("feedback"): return {"next_agent": "reviewer"}
    else: return {"next_agent": "done"}

def researcher(state):
    r = ChatOpenAI(model="gpt-4o").invoke([
        SystemMessage(content="You are a thorough researcher."),
        HumanMessage(content=f"Research: {state['topic']}")
    ])
    return {"notes": r.content, "messages": [AIMessage(content=f"[Researcher] Done", name="researcher")]}

def writer(state):
    prompt = f"Write about: {state['topic']}\nResearch: {state['notes']}"
    if state.get("feedback"): prompt += f"\nFeedback: {state['feedback']}"
    r = ChatOpenAI(model="gpt-4o", temperature=0.7).invoke([
        SystemMessage(content="You are an excellent writer."),
        HumanMessage(content=prompt)
    ])
    return {"draft": r.content, "revision_count": 1, "feedback": ""}

def reviewer(state):
    r = ChatOpenAI(model="gpt-4o").invoke([
        SystemMessage(content="Review. Say APPROVED if good, else give feedback."),
        HumanMessage(content=state["draft"])
    ])
    return {"feedback": r.content}

graph = StateGraph(ResearchState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", lambda s: s["next_agent"],
    {"researcher": "researcher", "writer": "writer", "reviewer": "reviewer", "done": END})
for node in ["researcher", "writer", "reviewer"]:
    graph.add_edge(node, "supervisor")

memory = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory)

result = app.invoke({
    "topic": "Impact of Agentic AI on Enterprise Software",
    "messages": [], "notes": "", "draft": "", "feedback": "",
    "revision_count": 0, "next_agent": ""
}, config={"configurable": {"thread_id": "research-1"}})
```

---

# SECTION 15: PRODUCTION BEST PRACTICES

1. **Always use checkpointer** - Even simple graphs benefit from persistence
2. **Set recursion limits** - Prevent runaway costs
3. **Smaller models for routing** - gpt-4o-mini for classification
4. **Monitor with LangSmith** - Trace every node
5. **Test edge cases** - Max retries, empty states, errors
6. **Design for failure** - Every node handles errors
7. **Stream everything** - Users expect real-time feedback
8. **Version graphs** - LangGraph Platform Assistants API

---

## Sources
- [LangGraph Docs](https://python.langchain.com/docs/concepts/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph Multi-Agent Blog](https://blog.langchain.com/langgraph-multi-agent-workflows/)
- [Latenode: Architecture Guide 2025](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/)
