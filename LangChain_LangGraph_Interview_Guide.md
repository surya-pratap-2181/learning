# LangChain & LangGraph Comprehensive Interview Guide for AI Engineers (2025-2026)

---

## TABLE OF CONTENTS

1. [LangChain Core Concepts](#1-langchain-core-concepts)
2. [LangGraph Deep Dive](#2-langgraph-deep-dive)
3. [LangSmith](#3-langsmith)
4. [LangServe](#4-langserve)
5. [Common Agentic Patterns](#5-common-agentic-patterns)
6. [Advanced Topics](#6-advanced-topics)
7. [LCEL (LangChain Expression Language)](#7-lcel-langchain-expression-language)
8. [Real-World Interview Questions with Code Examples](#8-real-world-interview-questions-with-code-examples)
9. [Best Practices & Anti-Patterns](#9-best-practices--anti-patterns)

---

## 1. LANGCHAIN CORE CONCEPTS

### Q1: What is LangChain and what problem does it solve?

**Answer:** LangChain is an open-source framework for building applications powered by Large Language Models (LLMs). It solves the problem of connecting LLMs to external data sources, tools, and APIs while providing abstractions for common patterns like chaining operations, managing memory, and building agents.

**Key value propositions:**
- **Composability**: Build complex LLM pipelines by chaining simple components
- **Data-awareness**: Connect LLMs to external data (documents, APIs, databases)
- **Agentic behavior**: Enable LLMs to take actions and make decisions
- **Ecosystem**: Standardized interfaces for 100+ LLM providers, vector stores, and tools

**Architecture (as of v0.3):**
```
langchain-core    -> Base abstractions and LCEL
langchain         -> Chains, agents, retrieval strategies (higher-level)
langchain-community -> Third-party integrations
Partner packages  -> langchain-openai, langchain-anthropic, etc.
langgraph         -> Stateful, multi-actor orchestration
langserve         -> Deploy chains as REST APIs
langsmith         -> Observability, tracing, evaluation
```

---

### Q2: What are Chains in LangChain? How have they evolved?

**Answer:** Chains are sequences of operations that process input and produce output. They evolved significantly:

**Legacy Chains (v0.1 - deprecated):**
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# OLD WAY - deprecated
llm = ChatOpenAI()
prompt = PromptTemplate.from_template("Tell me about {topic}")
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("quantum computing")
```

**Modern LCEL Chains (v0.2+):**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# NEW WAY - LCEL
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "Tell me about quantum computing"})
```

**Key difference:** LCEL chains are composable using the pipe `|` operator, support streaming natively, have built-in async support, and provide better observability through LangSmith.

**Common Chain Types:**
- **Sequential chains**: Operations run one after another
- **Parallel chains (RunnableParallel)**: Multiple operations run concurrently
- **Branching chains (RunnableBranch)**: Conditional routing based on input
- **Retrieval chains**: Combine retrieval with generation (RAG)

---

### Q3: Explain LangChain Prompt Templates in depth.

**Answer:** Prompt templates are reusable, parameterized prompts that separate prompt logic from application logic.

**Types of Prompt Templates:**

```python
# 1. Simple PromptTemplate
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Summarize the following text in {language}:\n{text}"
)
result = prompt.format(language="Spanish", text="LangChain is great...")

# 2. ChatPromptTemplate (most common in production)
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Respond in {style} style."),
    ("human", "{input}"),
])

# 3. MessagesPlaceholder - for injecting dynamic message history
from langchain_core.prompts import MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# 4. FewShotPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]
example_prompt = PromptTemplate.from_template(
    "Input: {input}\nOutput: {output}"
)
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input:",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

# 5. Partial prompt templates
prompt = PromptTemplate.from_template("Tell me about {topic} in {language}")
partial_prompt = prompt.partial(language="English")
result = partial_prompt.format(topic="AI")
```

**Interview Follow-up: When would you use MessagesPlaceholder?**
Use it when you need to inject a dynamic list of messages (e.g., chat history) into a prompt. The number of messages varies per invocation, so you cannot use fixed message slots.

---

### Q4: What are Output Parsers and how do they work?

**Answer:** Output parsers transform raw LLM string output into structured formats.

```python
# 1. StrOutputParser - simplest, extracts string content
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()

# 2. JsonOutputParser - parses JSON from LLM output
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline")

parser = JsonOutputParser(pydantic_object=Joke)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Return a joke in JSON format.\n{format_instructions}"),
    ("human", "{query}"),
])

chain = prompt.partial(
    format_instructions=parser.get_format_instructions()
) | llm | parser

result = chain.invoke({"query": "Tell me a joke about programming"})
# result = {"setup": "Why do programmers...", "punchline": "Because..."}

# 3. PydanticOutputParser
from langchain_core.output_parsers import PydanticOutputParser

class MovieReview(BaseModel):
    title: str
    rating: float = Field(ge=0, le=10)
    summary: str

parser = PydanticOutputParser(pydantic_object=MovieReview)

# 4. Structured Output (preferred in v0.3) - uses tool calling under the hood
structured_llm = llm.with_structured_output(MovieReview)
result = structured_llm.invoke("Review the movie Inception")
# result is a MovieReview Pydantic object

# 5. CommaSeparatedListOutputParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()

# 6. XMLOutputParser
from langchain_core.output_parsers import XMLOutputParser
```

**Interview Key Point:** In modern LangChain (v0.3), prefer `.with_structured_output()` over manual output parsers. It uses the model's native tool-calling/function-calling capability, which is more reliable than asking the model to output a specific format.

---

### Q5: Explain Document Loaders and Text Splitters.

**Answer:**

**Document Loaders** ingest data from various sources into LangChain `Document` objects:

```python
from langchain_core.documents import Document

# A Document has two fields:
# - page_content: str (the text)
# - metadata: dict (source info, page number, etc.)

# Common loaders:
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    UnstructuredFileLoader,
    DirectoryLoader,
    WikipediaLoader,
    ArxivLoader,
    GitLoader,
    NotionDirectoryLoader,
)

# Example: Load a PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()  # Returns List[Document]

# Lazy loading for large files
for doc in loader.lazy_load():
    process(doc)

# Load from web
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# Load entire directory
loader = DirectoryLoader("./data/", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()
```

**Text Splitters** break large documents into smaller chunks for embedding/retrieval:

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    RecursiveJsonSplitter,
    SemanticChunker,
)

# Most commonly used: RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # max characters per chunk
    chunk_overlap=200,      # overlap between consecutive chunks
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],  # tries these in order
)

chunks = splitter.split_documents(documents)

# Token-based splitting (more accurate for LLM context windows)
from langchain_text_splitters import TokenTextSplitter
splitter = TokenTextSplitter(
    chunk_size=500,     # in tokens
    chunk_overlap=50,
    encoding_name="cl100k_base"  # tiktoken encoding
)

# Markdown-aware splitting
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)

# Semantic chunking (groups semantically similar text)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"
)
```

**Interview Question: How do you choose chunk_size and chunk_overlap?**
- **chunk_size**: Depends on model context window, embedding model limits, and retrieval precision. Typical: 500-1500 characters. Smaller chunks = more precise retrieval but less context. Larger = more context but noisier retrieval.
- **chunk_overlap**: Usually 10-20% of chunk_size. Ensures sentences split at boundaries are not lost. 100-200 characters is common.
- **Best practice**: Experiment and evaluate using LangSmith. Use semantic chunking for heterogeneous documents.

---

### Q6: Explain Retrievers in LangChain.

**Answer:** Retrievers take a query string and return relevant `Document` objects. They are the "R" in RAG.

```python
# Base interface
from langchain_core.retrievers import BaseRetriever

# 1. Vector Store Retriever (most common)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma, Pinecone

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# Convert to retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",       # or "mmr" or "similarity_score_threshold"
    search_kwargs={"k": 5}          # return top 5 results
)

# MMR (Maximal Marginal Relevance) - balances relevance and diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)

# 2. Contextual Compression Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 3. Multi-Query Retriever (generates multiple query variations)
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# 4. Parent Document Retriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 5. Ensemble Retriever (combines multiple retrievers)
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

# 6. Self-Query Retriever (LLM generates metadata filters)
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="year", description="Year published", type="integer"),
    AttributeInfo(name="genre", description="Genre of the movie", type="string"),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Movie reviews",
    metadata_field_info=metadata_field_info,
)

# Using retriever in LCEL chain
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

### Q7: Explain all Memory types in LangChain.

**Answer:** Memory allows chains/agents to remember previous interactions. Note: In LangGraph, memory is handled through state and checkpointing, which is the recommended modern approach.

```python
# 1. ConversationBufferMemory - stores ALL messages
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True  # returns Message objects vs string
)
memory.save_context(
    {"input": "Hi, my name is Alice"},
    {"output": "Hello Alice! How can I help you?"}
)
# Stores: [HumanMessage("Hi..."), AIMessage("Hello Alice...")]
# Pros: Simple, complete history
# Cons: Grows unbounded, can exceed context window

# 2. ConversationBufferWindowMemory - stores last K interactions
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=10, return_messages=True)
# Pros: Fixed size, recent context preserved
# Cons: Loses older context

# 3. ConversationSummaryMemory - LLM summarizes conversation
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
# After several messages, stores: "The human introduced themselves as Alice.
# They asked about Python decorators and the AI explained..."
# Pros: Compact, captures key info from long conversations
# Cons: Loses detail, requires extra LLM calls (cost + latency)

# 4. ConversationSummaryBufferMemory - hybrid summary + buffer
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000  # summarize when exceeds this
)
# Keeps recent messages as-is, summarizes older ones
# Pros: Best of both worlds
# Cons: More complex

# 5. ConversationTokenBufferMemory - limits by token count
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=1000
)

# 6. ConversationEntityMemory - extracts and stores entities
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(llm=llm)
# Tracks entities: {"Alice": "A software engineer interested in AI"}

# 7. VectorStoreRetrieverMemory - stores in vector DB
from langchain.memory import VectorStoreRetrieverMemory

memory = VectorStoreRetrieverMemory(retriever=retriever)
# Retrieves most relevant past interactions (not just recent)
```

**Interview Key Point:** In LangGraph (the recommended approach for stateful apps), memory is handled through:
- **Short-term**: Graph state (within a conversation)
- **Long-term**: Checkpointers (SQLite, Postgres) for persistence across sessions
- **Cross-thread**: Store API for shared memories across conversations

---

### Q8: What are Callbacks in LangChain?

**Answer:** Callbacks provide hooks into the LLM application lifecycle for logging, monitoring, streaming, and custom logic.

```python
from langchain_core.callbacks import (
    BaseCallbackHandler,
    StdOutCallbackHandler,
    AsyncCallbackHandler,
)

# Custom callback handler
class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM finished. Tokens used: {response.llm_output}")

    def on_llm_error(self, error, **kwargs):
        print(f"LLM error: {error}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain started with: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"Chain ended with: {outputs}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool called: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"Tool result: {output}")

    def on_retriever_start(self, serialized, query, **kwargs):
        print(f"Retriever query: {query}")

    def on_retriever_end(self, documents, **kwargs):
        print(f"Retrieved {len(documents)} documents")

# Using callbacks
handler = MyCallbackHandler()

# Method 1: Pass at invocation (recommended)
result = chain.invoke({"input": "hello"}, config={"callbacks": [handler]})

# Method 2: Pass at construction
llm = ChatOpenAI(callbacks=[handler])

# Async callbacks
class AsyncHandler(AsyncCallbackHandler):
    async def on_llm_start(self, serialized, prompts, **kwargs):
        await log_to_db(prompts)

# Streaming callback (token by token)
class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
```

**Important:** LangSmith tracing is the production-grade alternative to custom callbacks for observability.

---

## 2. LANGGRAPH DEEP DIVE

### Q9: What is LangGraph and how does it differ from LangChain?

**Answer:** LangGraph is a library for building stateful, multi-actor applications with LLMs, using a graph-based approach. It is built on top of LangChain's primitives but solves different problems.

| Feature | LangChain (LCEL) | LangGraph |
|---------|------------------|-----------|
| Execution model | DAG (directed acyclic) | Cyclic graphs allowed |
| State management | Input/output passing | Explicit state with reducers |
| Persistence | Manual | Built-in checkpointing |
| Human-in-the-loop | Difficult | First-class support |
| Use case | Simple chains, RAG | Complex agents, multi-step workflows |
| Streaming | Token-level | Token + state update streaming |
| Error recovery | Restart from beginning | Resume from checkpoint |

**When to use LangGraph over LCEL:**
- Need cycles/loops (e.g., agent retrying after error)
- Need persistent state across interactions
- Need human approval steps
- Complex multi-agent workflows
- Need fine-grained control over agent behavior

---

### Q10: Explain StateGraph, Nodes, and Edges in detail.

**Answer:**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

# 1. STATE DEFINITION
# TypedDict defines the shape of state flowing through the graph
class AgentState(TypedDict):
    messages: Annotated[list, add]        # 'add' is a reducer - appends new messages
    current_step: str
    iteration_count: int
    final_answer: str

# 2. NODE FUNCTIONS
# Nodes are regular Python functions that take state and return partial state updates
def call_model(state: AgentState) -> dict:
    """Node that calls the LLM."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}  # reducer 'add' appends this

def process_tool(state: AgentState) -> dict:
    """Node that processes tool calls."""
    last_message = state["messages"][-1]
    # ... execute tool
    tool_result = execute_tool(last_message.tool_calls[0])
    return {
        "messages": [tool_result],
        "iteration_count": state["iteration_count"] + 1
    }

def generate_answer(state: AgentState) -> dict:
    """Node that generates final answer."""
    return {"final_answer": "Based on my research..."}

# 3. CONDITIONAL EDGES
def should_continue(state: AgentState) -> str:
    """Routing function for conditional edges."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"          # route to tools node
    elif state["iteration_count"] > 5:
        return "generate"       # route to answer generation
    else:
        return END              # finish

# 4. BUILD THE GRAPH
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", call_model)
graph.add_node("tools", process_tool)
graph.add_node("generate", generate_answer)

# Add edges
graph.add_edge(START, "agent")              # start -> agent
graph.add_conditional_edges(                 # agent -> conditional routing
    "agent",
    should_continue,
    {
        "tools": "tools",
        "generate": "generate",
        END: END,
    }
)
graph.add_edge("tools", "agent")            # tools -> agent (creates a cycle!)
graph.add_edge("generate", END)             # generate -> end

# 5. COMPILE AND RUN
app = graph.compile()

# Invoke
result = app.invoke({
    "messages": [HumanMessage(content="What is the weather?")],
    "current_step": "start",
    "iteration_count": 0,
    "final_answer": "",
})
```

**Key Concepts Explained:**
- **Nodes**: Python functions that receive state and return partial state updates. They contain the actual logic.
- **Edges**: Define the flow between nodes. Can be static (always go A -> B) or conditional (go to B or C based on state).
- **START/END**: Special sentinel nodes representing graph entry and exit points.
- **Conditional Edges**: A routing function examines state and returns the name of the next node.

---

### Q11: Explain Reducer Functions and State Management.

**Answer:** Reducers define HOW state updates are applied when a node returns a value for a state key.

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import MessagesState

# WITHOUT REDUCER: value is OVERWRITTEN
class SimpleState(TypedDict):
    count: int          # new value replaces old value
    name: str           # new value replaces old value

# WITH REDUCER: values are COMBINED
class ReducerState(TypedDict):
    messages: Annotated[list, add]      # lists are concatenated
    scores: Annotated[list, add]        # lists are concatenated
    count: int                          # still overwritten (no reducer)

# Custom reducer function
def keep_last_n(existing: list, new: list, n: int = 10) -> list:
    combined = existing + new
    return combined[-n:]

# Using custom reducer
class BoundedState(TypedDict):
    messages: Annotated[list, keep_last_n]

# BUILT-IN: MessagesState (most common for chat agents)
# This is a pre-built state with smart message handling
from langgraph.graph import MessagesState

# MessagesState is equivalent to:
# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
#
# The add_messages reducer:
# - Appends new messages
# - Handles message ID deduplication
# - Supports message updates (same ID = replace)

class MyState(MessagesState):
    # Extend with custom fields
    user_name: str
    tool_results: Annotated[list, add]
    iteration: int
```

**Interview Question: What happens if two nodes run in parallel and both update the same state key?**
- If the key has a reducer: Both updates are combined using the reducer (e.g., both lists are concatenated).
- If the key has NO reducer: This is an error -- LangGraph will raise an `InvalidUpdateError` because it doesn't know which value to keep.

---

### Q12: Explain Checkpointing and Persistence.

**Answer:** Checkpointing saves the complete state of a graph execution at each step, enabling pause/resume, time-travel, and fault tolerance.

```python
# 1. In-Memory Checkpointer (for development)
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Must provide thread_id in config
config = {"configurable": {"thread_id": "user-123"}}

# First conversation turn
result1 = app.invoke(
    {"messages": [HumanMessage(content="My name is Alice")]},
    config=config
)

# Second turn - remembers previous state!
result2 = app.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config
)
# AI responds: "Your name is Alice"

# 2. SQLite Checkpointer (for lightweight persistence)
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as memory:
    app = graph.compile(checkpointer=memory)
    result = app.invoke(input, config={"configurable": {"thread_id": "t1"}})

# 3. PostgreSQL Checkpointer (for production)
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as memory:
    app = graph.compile(checkpointer=memory)

# 4. Time Travel - replay from specific checkpoint
# Get all checkpoints for a thread
checkpoints = list(memory.list(config))

# Get state at specific checkpoint
state_at_step_2 = app.get_state(config, checkpoint_id="checkpoint-xyz")

# Resume from specific checkpoint
app.update_state(config, {"messages": [HumanMessage("new input")]})

# 5. Get current state
current_state = app.get_state(config)
print(current_state.values)     # current state values
print(current_state.next)       # next node(s) to execute
```

---

### Q13: How does Human-in-the-Loop work in LangGraph?

**Answer:** LangGraph provides first-class support for pausing graph execution to get human input/approval.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# METHOD 1: Using interrupt_before / interrupt_after
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],   # pause BEFORE executing "tools" node
    # interrupt_after=["agent"],  # OR pause AFTER executing "agent" node
)

config = {"configurable": {"thread_id": "hitl-1"}}

# Step 1: Run until interrupt
result = app.invoke(
    {"messages": [HumanMessage("Delete all files in /tmp")]},
    config=config
)

# Graph is now PAUSED before the "tools" node
state = app.get_state(config)
print(state.next)  # ("tools",) - shows where it will resume

# Step 2: Human reviews and either approves or modifies
# Option A: Approve - just continue
result = app.invoke(None, config=config)  # resumes from checkpoint

# Option B: Modify state before continuing
app.update_state(
    config,
    {"messages": [HumanMessage("Actually, only delete .tmp files")]},
    as_node="agent"  # pretend this update came from the "agent" node
)
result = app.invoke(None, config=config)

# Option C: Reject - replace the tool call entirely
app.update_state(
    config,
    {"messages": [AIMessage(content="I won't do that. Too dangerous.")]},
    as_node="agent"
)

# METHOD 2: Using interrupt() function (LangGraph >= 0.2.57)
from langgraph.types import interrupt, Command

def human_review_node(state):
    """This node pauses for human input."""
    tool_call = state["messages"][-1].tool_calls[0]

    # This PAUSES execution and sends data to the client
    human_response = interrupt({
        "question": f"Approve tool call: {tool_call}?",
        "tool_call": tool_call,
    })

    if human_response["approved"]:
        return Command(goto="tools")
    else:
        return Command(
            goto="agent",
            update={"messages": [AIMessage(content="Tool call rejected.")]}
        )
```

---

### Q14: Explain Subgraphs in LangGraph.

**Answer:** Subgraphs allow you to compose graphs hierarchically, enabling modular and reusable agent components.

```python
# Define a subgraph for research
class ResearchState(TypedDict):
    query: str
    sources: Annotated[list, add]
    summary: str

def search_web(state: ResearchState):
    results = web_search(state["query"])
    return {"sources": results}

def summarize(state: ResearchState):
    summary = llm.invoke(f"Summarize: {state['sources']}")
    return {"summary": summary.content}

# Build subgraph
research_graph = StateGraph(ResearchState)
research_graph.add_node("search", search_web)
research_graph.add_node("summarize", summarize)
research_graph.add_edge(START, "search")
research_graph.add_edge("search", "summarize")
research_graph.add_edge("summarize", END)
research_subgraph = research_graph.compile()

# Define parent graph
class ParentState(TypedDict):
    messages: Annotated[list, add]
    research_query: str
    research_result: str

def prepare_research(state: ParentState) -> dict:
    return {"research_query": state["messages"][-1].content}

def call_research_subgraph(state: ParentState) -> dict:
    # Invoke the subgraph
    result = research_subgraph.invoke({
        "query": state["research_query"],
        "sources": [],
        "summary": ""
    })
    return {"research_result": result["summary"]}

# Build parent graph
parent = StateGraph(ParentState)
parent.add_node("prepare", prepare_research)
parent.add_node("research", call_research_subgraph)
# OR directly add the compiled subgraph as a node:
# parent.add_node("research", research_subgraph)
parent.add_edge(START, "prepare")
parent.add_edge("prepare", "research")
parent.add_edge("research", END)

app = parent.compile()
```

**Key benefit:** Each subgraph has its own state schema, making them independently testable and reusable.

---

### Q15: How does Streaming work in LangGraph?

**Answer:**

```python
# 1. Stream state updates (default)
config = {"configurable": {"thread_id": "stream-1"}}

for event in app.stream(
    {"messages": [HumanMessage("Tell me about AI")]},
    config=config,
    stream_mode="updates"    # only state changes per node
):
    print(event)
# Output: {"agent": {"messages": [AIMessage(...)]}}
# Output: {"tools": {"messages": [ToolMessage(...)]}}

# 2. Stream full state values
for event in app.stream(input, config, stream_mode="values"):
    print(event)
# Output: full state dict after each node

# 3. Stream LLM tokens
for event in app.stream(input, config, stream_mode="messages"):
    # event is (message_chunk, metadata)
    chunk, metadata = event
    if isinstance(chunk, AIMessageChunk):
        print(chunk.content, end="", flush=True)

# 4. Combined streaming modes
async for event in app.astream(
    input, config,
    stream_mode=["updates", "messages"]
):
    mode, data = event
    if mode == "messages":
        chunk, metadata = data
        # handle token streaming
    elif mode == "updates":
        # handle state updates

# 5. Stream events (most granular)
async for event in app.astream_events(input, config, version="v2"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        print(token, end="")
    elif kind == "on_tool_start":
        print(f"Tool called: {event['name']}")
```

---

### Q16: What are Breakpoints and how do they differ from interrupts?

**Answer:**

**Breakpoints** are compile-time configuration that pause execution at specific nodes:
```python
# Breakpoints - defined at compile time
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["dangerous_action"],   # breakpoint BEFORE node
    interrupt_after=["llm_call"],            # breakpoint AFTER node
)
# Always pauses at these nodes, regardless of state
```

**Interrupts** are runtime, dynamic pauses inside node functions:
```python
from langgraph.types import interrupt

def my_node(state):
    if state["requires_approval"]:
        # Only pauses conditionally
        response = interrupt({"question": "Approve?"})
    return state
```

| Feature | Breakpoints | Interrupts |
|---------|------------|------------|
| Defined at | Compile time | Runtime (inside node) |
| Conditional | No (always pauses) | Yes (can be conditional) |
| Data to human | State only | Custom payload |
| Response from human | State update | Custom response value |

---

## 3. LANGSMITH

### Q17: What is LangSmith and why is it important?

**Answer:** LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications.

```python
# Setup - just set environment variables
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All LangChain/LangGraph operations are automatically traced!
# No code changes needed.

# Manual tracing with @traceable decorator
from langsmith import traceable

@traceable(name="my_custom_function", run_type="chain")
def my_function(query: str) -> str:
    # This function's inputs/outputs are traced
    result = chain.invoke({"input": query})
    return result

# Wrap non-LangChain code
from langsmith import Client
from langsmith.run_helpers import traceable

@traceable
def call_custom_api(prompt: str):
    response = requests.post("https://my-llm-api.com/v1/chat", json={"prompt": prompt})
    return response.json()

# Evaluation with LangSmith
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Create a dataset
dataset = client.create_dataset("QA-eval-set")
client.create_examples(
    inputs=[
        {"question": "What is LangChain?"},
        {"question": "What is LangGraph?"},
    ],
    outputs=[
        {"answer": "A framework for LLM applications"},
        {"answer": "A library for stateful LLM workflows"},
    ],
    dataset_id=dataset.id,
)

# Define evaluators
def correctness(run, example):
    """Custom evaluator."""
    prediction = run.outputs["output"]
    reference = example.outputs["answer"]
    # Use LLM to judge
    score = llm.invoke(
        f"Score 0-1 if prediction matches reference.\n"
        f"Prediction: {prediction}\nReference: {reference}"
    )
    return {"score": float(score.content), "key": "correctness"}

# Run evaluation
results = evaluate(
    lambda inputs: chain.invoke(inputs),
    data="QA-eval-set",
    evaluators=[correctness],
    experiment_prefix="v1-gpt4o",
)
```

**LangSmith Key Features:**
1. **Tracing**: Visualize every step of chain/agent execution, see inputs/outputs/latency/tokens at each step
2. **Debugging**: Identify where chains fail, see exact prompts sent to LLMs
3. **Evaluation**: Run experiments, compare model versions, use custom evaluators
4. **Monitoring**: Track latency, cost, error rates in production
5. **Datasets**: Create golden datasets for regression testing
6. **Feedback**: Collect human feedback on outputs
7. **Playground**: Test prompt variations interactively
8. **Online evaluation**: Run evaluators on production traces automatically

---

## 4. LANGSERVE

### Q18: What is LangServe and how do you deploy LangChain apps?

**Answer:** LangServe deploys LangChain runnables (LCEL chains) as REST APIs with auto-generated docs, playground UI, and streaming support.

```python
# server.py
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create chain
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
chain = prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()

# Create FastAPI app
app = FastAPI(title="My LangChain API", version="1.0")

# Add chain as API route
add_routes(
    app,
    chain,
    path="/chat",
    # Optional: restrict input/output schema
    enabled_endpoints=["invoke", "stream", "batch"],
)

# Add another chain
add_routes(app, rag_chain, path="/rag")

# Run: uvicorn server:app --host 0.0.0.0 --port 8000

# Auto-generated endpoints:
# POST /chat/invoke     - single invocation
# POST /chat/stream     - streaming response
# POST /chat/batch      - batch invocations
# GET  /chat/playground  - interactive UI
# GET  /chat/input_schema
# GET  /chat/output_schema
```

```python
# client.py - consuming the API
from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chat")

# Invoke
result = chain.invoke({"topic": "quantum computing"})

# Stream
for chunk in chain.stream({"topic": "quantum computing"}):
    print(chunk, end="", flush=True)

# Batch
results = chain.batch([{"topic": "AI"}, {"topic": "ML"}])

# Async
result = await chain.ainvoke({"topic": "AI"})
```

**Interview Note:** For LangGraph applications, the recommended deployment path is **LangGraph Platform** (formerly LangGraph Cloud), which provides:
- Managed infrastructure for stateful graph execution
- Built-in persistence (checkpointing)
- Cron jobs, webhooks
- Horizontal scaling
- LangGraph Studio for visual debugging

---

## 5. COMMON AGENTIC PATTERNS

### Q19: Explain the ReAct Agent pattern.

**Answer:** ReAct (Reasoning + Acting) is a pattern where the LLM alternates between reasoning about what to do and taking actions (calling tools).

```python
# Using LangGraph's prebuilt ReAct agent
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}..."

@tool
def calculator(expression: str) -> float:
    """Calculate a mathematical expression."""
    return eval(expression)

llm = ChatOpenAI(model="gpt-4o")
tools = [search, calculator]

# Create ReAct agent
agent = create_react_agent(llm, tools)

# Run
result = agent.invoke({
    "messages": [HumanMessage("What is the population of France times 2?")]
})

# Flow:
# 1. LLM reasons: "I need to search for France's population"
# 2. LLM calls: search("population of France")
# 3. Tool returns: "67 million"
# 4. LLM reasons: "Now I need to multiply by 2"
# 5. LLM calls: calculator("67000000 * 2")
# 6. Tool returns: 134000000
# 7. LLM answers: "The population of France times 2 is 134 million"

# CUSTOM ReAct implementation in LangGraph
from langgraph.graph import StateGraph, MessagesState, START, END

def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: MessagesState):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = tool_map[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    return {"messages": results}

def should_continue(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")  # cycle back for more reasoning

agent = graph.compile()
```

---

### Q20: Explain the Plan-and-Execute pattern.

**Answer:** The agent first creates a plan (list of steps), then executes each step, and optionally revises the plan based on results.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class PlanExecuteState(TypedDict):
    messages: Annotated[list, add]
    plan: list[str]                # list of steps
    current_step: int
    step_results: Annotated[list, add]
    final_response: str

def planner(state: PlanExecuteState) -> dict:
    """Create a plan to answer the query."""
    response = llm.invoke(
        f"Create a step-by-step plan to answer: {state['messages'][-1].content}\n"
        f"Return as a numbered list."
    )
    steps = parse_plan(response.content)
    return {"plan": steps, "current_step": 0}

def executor(state: PlanExecuteState) -> dict:
    """Execute the current step."""
    step = state["plan"][state["current_step"]]
    result = agent_executor.invoke({"input": step})
    return {
        "step_results": [result["output"]],
        "current_step": state["current_step"] + 1
    }

def replanner(state: PlanExecuteState) -> dict:
    """Optionally revise the plan based on results so far."""
    response = llm.invoke(
        f"Original plan: {state['plan']}\n"
        f"Completed steps and results: {state['step_results']}\n"
        f"Should the remaining plan be revised? If so, provide new plan."
    )
    if "REVISE" in response.content:
        new_plan = parse_plan(response.content)
        return {"plan": new_plan}
    return {}

def should_continue(state: PlanExecuteState) -> str:
    if state["current_step"] >= len(state["plan"]):
        return "final"
    return "execute"

graph = StateGraph(PlanExecuteState)
graph.add_node("plan", planner)
graph.add_node("execute", executor)
graph.add_node("replan", replanner)
graph.add_node("final", generate_final_answer)

graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", "replan")
graph.add_conditional_edges("replan", should_continue, {
    "execute": "execute",
    "final": "final"
})
graph.add_edge("final", END)
```

---

### Q21: Explain the Reflection and Self-Correction patterns.

**Answer:**

**Reflection Pattern:** An LLM generates output, then a second LLM (or the same one) critiques it, and the output is refined.

```python
class ReflectionState(TypedDict):
    messages: Annotated[list, add]
    draft: str
    critique: str
    iteration: int

def generate(state: ReflectionState) -> dict:
    if state.get("critique"):
        prompt = f"Revise your essay based on this feedback:\n{state['critique']}\n\nOriginal:\n{state['draft']}"
    else:
        prompt = f"Write an essay about: {state['messages'][-1].content}"
    response = llm.invoke(prompt)
    return {"draft": response.content, "iteration": state.get("iteration", 0) + 1}

def reflect(state: ReflectionState) -> dict:
    response = llm.invoke(
        f"Critique this essay. Be constructive and specific:\n\n{state['draft']}"
    )
    return {"critique": response.content}

def should_continue(state: ReflectionState) -> str:
    if state["iteration"] >= 3:  # max 3 iterations
        return END
    return "reflect"

graph = StateGraph(ReflectionState)
graph.add_node("generate", generate)
graph.add_node("reflect", reflect)
graph.add_edge(START, "generate")
graph.add_conditional_edges("generate", should_continue, {"reflect": "reflect", END: END})
graph.add_edge("reflect", "generate")  # cycle: reflect -> regenerate
```

**Self-Correction Pattern:** The agent validates its own output and corrects errors.

```python
def generate_code(state):
    code = llm.invoke(f"Write Python code for: {state['task']}")
    return {"code": code.content}

def validate_code(state):
    try:
        # Try to execute/validate
        exec(state["code"])
        return {"is_valid": True, "error": ""}
    except Exception as e:
        return {"is_valid": False, "error": str(e)}

def fix_code(state):
    fixed = llm.invoke(
        f"Fix this code:\n{state['code']}\n\nError: {state['error']}"
    )
    return {"code": fixed.content}

def route(state):
    if state["is_valid"]:
        return END
    return "fix"

graph = StateGraph(CodeState)
graph.add_node("generate", generate_code)
graph.add_node("validate", validate_code)
graph.add_node("fix", fix_code)
graph.add_edge(START, "generate")
graph.add_edge("generate", "validate")
graph.add_conditional_edges("validate", route, {"fix": "fix", END: END})
graph.add_edge("fix", "validate")  # cycle: fix -> validate again
```

---

### Q22: Explain Tool-Calling Agents and Structured Output.

**Answer:**

```python
# TOOL CALLING - Modern approach (v0.3+)
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

# Method 1: @tool decorator
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to get weather for.
        unit: Temperature unit - 'celsius' or 'fahrenheit'.
    """
    return f"Weather in {city}: 22{unit[0].upper()}, sunny"

# Method 2: StructuredTool with Pydantic schema
class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Max results to return")

search_tool = StructuredTool.from_function(
    func=search_function,
    name="web_search",
    description="Search the web for information",
    args_schema=SearchInput,
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools([get_weather, search_tool])

# The LLM now returns tool_calls in its response
response = llm_with_tools.invoke("What's the weather in Paris?")
print(response.tool_calls)
# [{"name": "get_weather", "args": {"city": "Paris"}, "id": "call_abc123"}]

# STRUCTURED OUTPUT - Force LLM to return specific schema
class MovieRecommendation(BaseModel):
    """A movie recommendation."""
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre")
    reason: str = Field(description="Why this movie is recommended")

# Method 1: with_structured_output (recommended)
structured_llm = llm.with_structured_output(MovieRecommendation)
result = structured_llm.invoke("Recommend a sci-fi movie")
# result is a MovieRecommendation pydantic object
print(result.title)  # "Blade Runner 2049"

# Method 2: with_structured_output using JSON schema
structured_llm = llm.with_structured_output({
    "title": "MovieRecommendation",
    "description": "A movie recommendation",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
    },
    "required": ["title", "year"],
})

# Method 3: Multiple structured outputs with include_raw
structured_llm = llm.with_structured_output(
    MovieRecommendation,
    include_raw=True  # returns {"raw": AIMessage, "parsed": MovieRecommendation, "parsing_error": None}
)
```

---

## 6. ADVANCED TOPICS

### Q23: How do you create Custom Tools?

**Answer:**

```python
from langchain_core.tools import BaseTool, tool, StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, Type

# Method 1: @tool decorator (simplest)
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

# Method 2: @tool with parsing/error handling
@tool(parse_docstring=True, response_format="content_and_artifact")
def search_docs(query: str, k: int = 5) -> tuple[str, list]:
    """Search the document database.

    Args:
        query: The search query
        k: Number of results to return
    """
    docs = vectorstore.similarity_search(query, k=k)
    content = "\n".join(d.page_content for d in docs)
    return content, docs  # content for LLM, docs as artifact

# Method 3: Subclass BaseTool (most control)
class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Execute a SQL query against the database"
    args_schema: Type[BaseModel] = DatabaseQueryInput
    return_direct: bool = False  # if True, tool output goes directly to user

    db_connection: Any = None  # custom fields

    def _run(self, query: str, limit: int = 100) -> str:
        """Execute synchronously."""
        try:
            results = self.db_connection.execute(query, limit=limit)
            return str(results)
        except Exception as e:
            return f"Error: {e}"

    async def _arun(self, query: str, limit: int = 100) -> str:
        """Execute asynchronously."""
        results = await self.db_connection.aexecute(query, limit=limit)
        return str(results)

# Method 4: StructuredTool from function
tool = StructuredTool.from_function(
    func=my_function,
    name="my_tool",
    description="Does something useful",
    args_schema=MyInputSchema,
    coroutine=my_async_function,  # optional async version
    handle_tool_error=True,  # return error message instead of raising
)

# Method 5: Tool with error handling
@tool
def risky_tool(input: str) -> str:
    """A tool that might fail."""
    try:
        result = external_api.call(input)
        return result
    except Exception as e:
        # Return error message so the agent can adapt
        return f"Tool error: {str(e)}. Please try a different approach."
```

---

### Q24: How do you create Custom Retrievers?

**Answer:**

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List

class CustomRetriever(BaseRetriever):
    """A custom retriever that searches a SQL database."""

    db_url: str
    top_k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Synchronous retrieval."""
        # Your custom retrieval logic
        conn = connect(self.db_url)
        results = conn.execute(
            "SELECT content, source FROM docs WHERE content LIKE %s LIMIT %s",
            (f"%{query}%", self.top_k)
        )
        return [
            Document(
                page_content=row["content"],
                metadata={"source": row["source"]}
            )
            for row in results
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager
    ) -> List[Document]:
        """Async retrieval."""
        # async implementation
        pass

# Use in a chain
retriever = CustomRetriever(db_url="postgresql://...", top_k=3)
docs = retriever.invoke("How does RAG work?")

# Use in RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

### Q25: Explain Streaming in LangChain/LangGraph.

**Answer:**

```python
# 1. LCEL Chain Streaming
chain = prompt | llm | StrOutputParser()

# Sync streaming
for chunk in chain.stream({"input": "Tell me a story"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"input": "Tell me a story"}):
    print(chunk, end="", flush=True)

# 2. Stream Events (granular visibility into all chain steps)
async for event in chain.astream_events(
    {"input": "Tell me a story"},
    version="v2"
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
    elif kind == "on_parser_start":
        print("Parser started...")
    elif kind == "on_chain_end":
        print(f"\nChain finished in {event['data']}")

# 3. Streaming with FastAPI/LangServe
from fastapi.responses import StreamingResponse
import json

async def stream_endpoint(request: Request):
    async def generate():
        async for chunk in chain.astream(request.json()):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# 4. LangGraph Streaming (multiple modes)
# Stream state updates
for chunk in graph.stream(input, config, stream_mode="updates"):
    node_name = list(chunk.keys())[0]
    print(f"Node '{node_name}' produced: {chunk[node_name]}")

# Stream values (full state after each step)
for chunk in graph.stream(input, config, stream_mode="values"):
    print(f"Current state: {chunk}")

# Stream messages (LLM tokens within graph nodes)
for mode, data in graph.stream(input, config, stream_mode=["messages", "updates"]):
    if mode == "messages":
        msg_chunk, metadata = data
        print(msg_chunk.content, end="")

# 5. Streaming with tool calls
async for event in agent.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content:
            print(chunk.content, end="")
        if chunk.tool_call_chunks:
            for tc in chunk.tool_call_chunks:
                print(f"Tool: {tc}")
```

---

### Q26: Explain Async Operations and Caching.

**Answer:**

```python
# ASYNC OPERATIONS
# Every LangChain component has async variants

# Async chain invocation
result = await chain.ainvoke({"input": "hello"})

# Async streaming
async for chunk in chain.astream({"input": "hello"}):
    print(chunk)

# Async batch with concurrency control
results = await chain.abatch(
    [{"input": "q1"}, {"input": "q2"}, {"input": "q3"}],
    config={"max_concurrency": 5}
)

# Async retriever
docs = await retriever.ainvoke("search query")

# Async in LangGraph
result = await graph.ainvoke(input, config)
async for event in graph.astream(input, config):
    process(event)

# CACHING STRATEGIES

# 1. LLM Response Caching - InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())
# Identical prompts return cached responses instantly

# 2. SQLite Cache (persistent)
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path="llm_cache.db"))

# 3. Redis Cache
from langchain_community.cache import RedisCache
import redis

set_llm_cache(RedisCache(redis_=redis.Redis()))

# 4. Semantic Cache (cache based on meaning, not exact match)
from langchain_community.cache import RedisSemanticCache

set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95  # how similar queries need to be
))

# 5. Per-chain caching
llm_cached = ChatOpenAI(model="gpt-4o", cache=True)
llm_uncached = ChatOpenAI(model="gpt-4o", cache=False)

# 6. Embedding caching (avoid re-embedding same documents)
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./embedding_cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store,
    namespace="openai-embed-3"
)
```

---

## 7. LCEL (LANGCHAIN EXPRESSION LANGUAGE)

### Q27: Explain LCEL and the pipe operator in detail.

**Answer:** LCEL is a declarative way to compose LangChain components. Every component implements the `Runnable` interface.

```python
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableBranch,
    RunnableConfig,
)

# The Runnable interface provides:
# .invoke(input)        - single input
# .batch(inputs)        - multiple inputs
# .stream(input)        - streaming output
# .ainvoke(input)       - async invoke
# .abatch(inputs)       - async batch
# .astream(input)       - async stream
# .astream_events()     - granular async event stream

# PIPE OPERATOR |
# chain = component1 | component2 | component3
# Output of each component becomes input of the next

chain = prompt | llm | parser
# Equivalent to: parser.invoke(llm.invoke(prompt.invoke(input)))

# COMPLEX LCEL EXAMPLE
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Chain 1: Translate
translate_prompt = ChatPromptTemplate.from_template(
    "Translate to English: {text}"
)
translate_chain = translate_prompt | ChatOpenAI() | StrOutputParser()

# Chain 2: Summarize
summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize in 2 sentences: {text}"
)
summarize_chain = summarize_prompt | ChatOpenAI() | StrOutputParser()

# Compose: translate THEN summarize
full_chain = (
    translate_chain
    | (lambda x: {"text": x})  # transform output for next chain
    | summarize_chain
)
```

---

### Q28: Explain RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableBranch.

**Answer:**

```python
# ==============================
# 1. RunnablePassthrough
# ==============================
# Passes input through unchanged. Used to forward data alongside transformations.

from langchain_core.runnables import RunnablePassthrough

# Common RAG pattern:
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Input "What is AI?" flows as:
# retriever gets "What is AI?" -> returns docs
# RunnablePassthrough() passes "What is AI?" unchanged
# Both go into prompt as context and question

# RunnablePassthrough.assign() - adds new keys while keeping existing ones
chain = RunnablePassthrough.assign(
    word_count=lambda x: len(x["text"].split()),
    upper=lambda x: x["text"].upper()
)
result = chain.invoke({"text": "hello world"})
# {"text": "hello world", "word_count": 2, "upper": "HELLO WORLD"}

# ==============================
# 2. RunnableLambda
# ==============================
# Wraps any Python function as a Runnable

from langchain_core.runnables import RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

format_runnable = RunnableLambda(format_docs)

# With async support
async def async_lookup(user_id: str) -> dict:
    return await db.get_user(user_id)

lookup_runnable = RunnableLambda(async_lookup)  # works in both sync and async

# In a chain
chain = retriever | RunnableLambda(format_docs) | prompt | llm

# Shorthand: lambda works directly in pipe
chain = retriever | (lambda docs: "\n".join(d.page_content for d in docs)) | prompt | llm

# ==============================
# 3. RunnableParallel
# ==============================
# Runs multiple runnables in parallel, returns dict of results

from langchain_core.runnables import RunnableParallel

# Explicit construction
parallel = RunnableParallel(
    summary=summarize_chain,
    translation=translate_chain,
    sentiment=sentiment_chain,
)
result = parallel.invoke({"text": "LangChain is amazing"})
# result = {
#     "summary": "LangChain is a framework...",
#     "translation": "LangChain es increible...",
#     "sentiment": "positive"
# }

# Dict shorthand (most common)
parallel = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
}
# This is implicitly a RunnableParallel!

rag_chain = parallel | prompt | llm | StrOutputParser()

# Parallel with different processing per branch
analysis_chain = RunnableParallel(
    topic=prompt1 | llm | StrOutputParser(),
    entities=prompt2 | llm | JsonOutputParser(),
    summary=prompt3 | llm | StrOutputParser(),
) | combine_results_prompt | llm

# ==============================
# 4. RunnableBranch
# ==============================
# Conditional routing based on input

from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    # (condition, runnable) pairs - checked in order
    (lambda x: "math" in x["topic"].lower(), math_chain),
    (lambda x: "science" in x["topic"].lower(), science_chain),
    (lambda x: "history" in x["topic"].lower(), history_chain),
    general_chain,  # default (last positional arg, no condition)
)

result = branch.invoke({"topic": "math problem", "question": "What is 2+2?"})
# Routes to math_chain

# Alternative: Use a routing function (often cleaner)
def route(input):
    if "math" in input["topic"]:
        return math_chain
    elif "science" in input["topic"]:
        return science_chain
    return general_chain

chain = RunnableLambda(route)

# ==============================
# 5. Combining everything
# ==============================
full_chain = (
    RunnablePassthrough.assign(
        category=classify_chain    # add category to input
    )
    | RunnableBranch(
        (lambda x: x["category"] == "technical", technical_chain),
        (lambda x: x["category"] == "creative", creative_chain),
        default_chain,
    )
)
```

---

### Q29: How does `.configurable_fields()` and `.configurable_alternatives()` work?

**Answer:**

```python
from langchain_core.runnables import ConfigurableField

# configurable_fields - make parameters adjustable at runtime
llm = ChatOpenAI(model="gpt-4o", temperature=0.7).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="Controls randomness",
    ),
    model_name=ConfigurableField(
        id="llm_model",
        name="Model",
    ),
)

chain = prompt | llm | StrOutputParser()

# Use default settings
result = chain.invoke({"input": "hello"})

# Override at runtime
result = chain.invoke(
    {"input": "hello"},
    config={"configurable": {"llm_temperature": 0.0, "llm_model": "gpt-4o-mini"}}
)

# configurable_alternatives - swap entire components at runtime
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import ConfigurableField

llm = ChatOpenAI(model="gpt-4o").configurable_alternatives(
    ConfigurableField(id="llm_provider"),
    default_key="openai",
    anthropic=ChatAnthropic(model="claude-sonnet-4-20250514"),
    fast=ChatOpenAI(model="gpt-4o-mini"),
)

chain = prompt | llm | StrOutputParser()

# Use default (OpenAI)
result = chain.invoke({"input": "hello"})

# Use Anthropic
result = chain.invoke(
    {"input": "hello"},
    config={"configurable": {"llm_provider": "anthropic"}}
)

# Use fast model
result = chain.invoke(
    {"input": "hello"},
    config={"configurable": {"llm_provider": "fast"}}
)
```

---

## 8. REAL-WORLD INTERVIEW QUESTIONS WITH CODE EXAMPLES

### Q30: Build a complete RAG pipeline with LangChain.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Load documents
loader = PyPDFLoader("company_docs.pdf")
documents = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)

# Step 3: Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Step 4: Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance for diversity
    search_kwargs={"k": 5, "fetch_k": 20}
)

# Step 5: Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question based ONLY
    on the following context. If the context doesn't contain the answer,
    say "I don't have enough information to answer that."

    Context: {context}"""),
    ("human", "{question}"),
])

# Step 6: Build RAG chain
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page: {doc.metadata.get('page', 'N/A')}\n"
        f"{doc.page_content}"
        for doc in docs
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Step 7: Use it
answer = rag_chain.invoke("What is our company's refund policy?")

# Step 8: Stream the answer
for chunk in rag_chain.stream("What is our company's refund policy?"):
    print(chunk, end="", flush=True)

# BONUS: RAG with sources
from langchain_core.runnables import RunnableParallel

rag_with_sources = RunnableParallel(
    answer=rag_chain,
    sources=retriever | (lambda docs: [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in docs
    ])
)
result = rag_with_sources.invoke("What is the refund policy?")
print(result["answer"])
print(result["sources"])
```

---

### Q31: Build a multi-tool agent with LangGraph.

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
import json

# Define tools
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for company information."""
    docs = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)

@tool
def get_customer_info(customer_id: str) -> str:
    """Look up customer information by their ID."""
    # Simulated database lookup
    customers = {
        "C001": {"name": "Alice", "plan": "Enterprise", "since": "2023"},
        "C002": {"name": "Bob", "plan": "Starter", "since": "2024"},
    }
    info = customers.get(customer_id, "Customer not found")
    return json.dumps(info)

@tool
def create_ticket(
    customer_id: str,
    subject: str,
    priority: str = "medium"
) -> str:
    """Create a support ticket for a customer.

    Args:
        customer_id: The customer's ID
        subject: Brief description of the issue
        priority: low, medium, or high
    """
    ticket_id = f"TKT-{hash(subject) % 10000:04d}"
    return f"Ticket {ticket_id} created for customer {customer_id} with priority {priority}"

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        reason: Why the conversation needs human attention
    """
    return f"Escalated to human agent. Reason: {reason}"

# Setup
tools = [search_knowledge_base, get_customer_info, create_ticket, escalate_to_human]
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a customer support agent. You can:
1. Search the knowledge base for product information
2. Look up customer information
3. Create support tickets
4. Escalate to human agents when needed

Always be helpful and professional. If you can't resolve an issue, create a ticket or escalate."""

# Define nodes
def agent(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Define routing
def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

# Build graph
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

# Compile with persistence
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "support-session-1"}}

# Turn 1
result = app.invoke(
    {"messages": [HumanMessage("Hi, I'm customer C001 and I can't access my dashboard")]},
    config=config
)
print(result["messages"][-1].content)

# Turn 2 - remembers context from turn 1
result = app.invoke(
    {"messages": [HumanMessage("Can you create a ticket for this?")]},
    config=config
)
print(result["messages"][-1].content)
```

---

### Q32: How would you implement a multi-agent system?

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, Annotated, Literal
from operator import add

class MultiAgentState(TypedDict):
    messages: Annotated[list, add]
    current_agent: str
    research_notes: str
    draft: str
    review_feedback: str
    final_output: str

# Agent 1: Researcher
def researcher(state: MultiAgentState) -> dict:
    response = research_llm.invoke([
        SystemMessage("You are a research agent. Gather relevant information."),
        HumanMessage(f"Research this topic: {state['messages'][-1].content}")
    ])
    return {"research_notes": response.content, "current_agent": "writer"}

# Agent 2: Writer
def writer(state: MultiAgentState) -> dict:
    response = writer_llm.invoke([
        SystemMessage("You are a writing agent. Create well-structured content."),
        HumanMessage(f"Write about this based on the research:\n{state['research_notes']}")
    ])
    return {"draft": response.content, "current_agent": "reviewer"}

# Agent 3: Reviewer
def reviewer(state: MultiAgentState) -> dict:
    response = reviewer_llm.invoke([
        SystemMessage("You are a critical reviewer. Provide constructive feedback."),
        HumanMessage(f"Review this draft:\n{state['draft']}")
    ])
    return {"review_feedback": response.content, "current_agent": "router"}

# Router
def router(state: MultiAgentState) -> Literal["writer", "final"]:
    if "APPROVED" in state["review_feedback"]:
        return "final"
    return "writer"  # send back for revision

def finalize(state: MultiAgentState) -> dict:
    return {"final_output": state["draft"]}

# Build multi-agent graph
graph = StateGraph(MultiAgentState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)
graph.add_node("final", finalize)

graph.add_edge(START, "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_conditional_edges("reviewer", router, {
    "writer": "writer",    # revision cycle
    "final": "final"
})
graph.add_edge("final", END)

app = graph.compile()

result = app.invoke({
    "messages": [HumanMessage("Write an article about AI agents")],
    "current_agent": "",
    "research_notes": "",
    "draft": "",
    "review_feedback": "",
    "final_output": "",
})
```

---

### Q33: How do you handle error recovery and retries in LangGraph?

```python
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
import time

class RetryState(MessagesState):
    error_count: int
    last_error: str

def call_api(state: RetryState) -> dict:
    try:
        result = unreliable_api.call(state["messages"][-1].content)
        return {"messages": [AIMessage(content=result)], "error_count": 0}
    except Exception as e:
        return {
            "last_error": str(e),
            "error_count": state.get("error_count", 0) + 1
        }

def should_retry(state: RetryState) -> str:
    if state.get("error_count", 0) == 0:
        return END  # success
    if state["error_count"] >= 3:
        return "fallback"  # too many retries
    return "wait_and_retry"

def wait_and_retry(state: RetryState) -> dict:
    time.sleep(2 ** state["error_count"])  # exponential backoff
    return {}  # just waits, then flows back to api_call

def fallback(state: RetryState) -> dict:
    return {
        "messages": [AIMessage(
            content=f"Sorry, I encountered an error: {state['last_error']}. "
                    f"Please try again later."
        )]
    }

graph = StateGraph(RetryState)
graph.add_node("api_call", call_api)
graph.add_node("wait_and_retry", wait_and_retry)
graph.add_node("fallback", fallback)

graph.add_edge(START, "api_call")
graph.add_conditional_edges("api_call", should_retry, {
    END: END,
    "wait_and_retry": "wait_and_retry",
    "fallback": "fallback",
})
graph.add_edge("wait_and_retry", "api_call")  # retry cycle
graph.add_edge("fallback", END)

# With LCEL, you can also use .with_retry()
from langchain_core.runnables import RunnableWithRetry

chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
    retry_if_exception_type=(TimeoutError, ConnectionError),
)

# And .with_fallbacks()
chain_with_fallback = primary_chain.with_fallbacks(
    [fallback_chain_1, fallback_chain_2]
)
```

---

### Q34: Implement a Corrective RAG (CRAG) system.

```python
"""
Corrective RAG: Retrieves documents, evaluates their relevance,
and falls back to web search if retrieved docs are not relevant.
"""
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from operator import add
from langchain_core.documents import Document

class CRAGState(TypedDict):
    question: str
    documents: List[Document]
    web_results: List[Document]
    relevance_scores: List[str]  # "relevant" or "irrelevant"
    generation: str

def retrieve(state: CRAGState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"documents": docs}

def grade_documents(state: CRAGState) -> dict:
    """Grade each retrieved document for relevance."""
    scores = []
    filtered_docs = []
    for doc in state["documents"]:
        grade = grading_llm.with_structured_output(GradeSchema).invoke(
            f"Is this document relevant to '{state['question']}'?\n\n{doc.page_content}"
        )
        if grade.relevant:
            scores.append("relevant")
            filtered_docs.append(doc)
        else:
            scores.append("irrelevant")
    return {"documents": filtered_docs, "relevance_scores": scores}

def decide_to_search(state: CRAGState) -> str:
    """If most docs are irrelevant, search the web."""
    relevant_count = state["relevance_scores"].count("relevant")
    total = len(state["relevance_scores"])
    if relevant_count / max(total, 1) < 0.5:
        return "web_search"
    return "generate"

def web_search(state: CRAGState) -> dict:
    """Fall back to web search."""
    results = tavily_search.invoke(state["question"])
    web_docs = [Document(page_content=r["content"]) for r in results]
    return {"documents": state["documents"] + web_docs}

def generate(state: CRAGState) -> dict:
    context = "\n\n".join(d.page_content for d in state["documents"])
    response = llm.invoke(
        f"Answer based on context:\n{context}\n\nQuestion: {state['question']}"
    )
    return {"generation": response.content}

# Build CRAG graph
graph = StateGraph(CRAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade_documents)
graph.add_node("web_search", web_search)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", decide_to_search, {
    "web_search": "web_search",
    "generate": "generate"
})
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)

crag = graph.compile()
```

---

### Q35: How do you evaluate RAG quality?

```python
# Using LangSmith for RAG evaluation
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Create evaluation dataset
dataset = client.create_dataset("rag-eval")
client.create_examples(
    inputs=[
        {"question": "What is our return policy?"},
        {"question": "How do I reset my password?"},
    ],
    outputs=[
        {"answer": "30-day return policy for unused items"},
        {"answer": "Click 'Forgot Password' on the login page"},
    ],
    dataset_id=dataset.id,
)

# Define evaluators

# 1. Faithfulness - is the answer grounded in the retrieved context?
def faithfulness_evaluator(run, example):
    prediction = run.outputs["answer"]
    context = run.outputs.get("context", "")
    grade = llm.with_structured_output(FaithfulnessGrade).invoke(
        f"Is this answer fully supported by the context?\n"
        f"Context: {context}\nAnswer: {prediction}"
    )
    return {"key": "faithfulness", "score": float(grade.is_faithful)}

# 2. Relevance - does the answer address the question?
def relevance_evaluator(run, example):
    question = run.inputs["question"]
    prediction = run.outputs["answer"]
    grade = llm.with_structured_output(RelevanceGrade).invoke(
        f"Does this answer address the question?\n"
        f"Question: {question}\nAnswer: {prediction}"
    )
    return {"key": "relevance", "score": float(grade.is_relevant)}

# 3. Correctness - does it match the expected answer?
def correctness_evaluator(run, example):
    prediction = run.outputs["answer"]
    reference = example.outputs["answer"]
    grade = llm.with_structured_output(CorrectnessGrade).invoke(
        f"Does the prediction match the reference?\n"
        f"Prediction: {prediction}\nReference: {reference}"
    )
    return {"key": "correctness", "score": grade.score}

# Run evaluation
results = evaluate(
    lambda inputs: rag_chain.invoke(inputs["question"]),
    data="rag-eval",
    evaluators=[faithfulness_evaluator, relevance_evaluator, correctness_evaluator],
    experiment_prefix="rag-v2-gpt4o",
)

# RAGAS-style metrics can also be computed:
# - Context Precision: Are the retrieved docs relevant?
# - Context Recall: Are all needed facts retrieved?
# - Answer Faithfulness: Is the answer grounded in context?
# - Answer Relevancy: Does the answer address the question?
```

---

## 9. BEST PRACTICES & ANTI-PATTERNS

### Q36: When should you use LangChain vs direct API calls?

**Use LangChain when:**
- Building RAG applications (document loading, splitting, retrieval, generation pipeline)
- Need to swap between LLM providers easily
- Building complex multi-step chains with error handling and retries
- Need observability/tracing (LangSmith integration)
- Building agents that use multiple tools
- Need structured output parsing
- Prototyping and rapid iteration

**Use direct API calls when:**
- Simple, single LLM call (no chaining needed)
- Need maximum control over request/response handling
- Performance-critical path where abstraction overhead matters
- Using provider-specific features not yet in LangChain
- Team is not familiar with LangChain and timeline is tight
- Very simple use case that doesn't benefit from the framework

**Use LangGraph when:**
- Need cycles in your workflow (retry loops, iterative refinement)
- Building stateful multi-turn agents
- Need human-in-the-loop approval flows
- Complex multi-agent orchestration
- Need persistent state across sessions
- Need fault tolerance (resume from checkpoint after crash)

```python
# DIRECT API CALL - simple use case
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
# Fine for a simple chatbot endpoint

# LANGCHAIN LCEL - when you need composition
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm.with_structured_output(Answer)
)
# Better for RAG, structured output, composable pipelines

# LANGGRAPH - when you need state and cycles
graph = StateGraph(AgentState)
# ... complex agent with tools, retries, human approval
# Better for complex agent workflows
```

---

### Q37: What are common anti-patterns to avoid?

**Anti-Pattern 1: Using legacy chains instead of LCEL**
```python
# BAD - deprecated, less composable
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# GOOD - modern LCEL
chain = prompt | llm | StrOutputParser()
```

**Anti-Pattern 2: Not using structured output properly**
```python
# BAD - fragile string parsing
response = llm.invoke("Return JSON: {name, age}")
data = json.loads(response.content)  # might fail!

# GOOD - structured output with validation
class Person(BaseModel):
    name: str
    age: int

result = llm.with_structured_output(Person).invoke("Extract person info from: John is 30")
```

**Anti-Pattern 3: Ignoring token limits in memory**
```python
# BAD - unbounded memory
memory = ConversationBufferMemory()  # grows forever, will exceed context

# GOOD - bounded memory
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000)

# BEST - use LangGraph with proper state management
```

**Anti-Pattern 4: Not handling retrieval failures**
```python
# BAD - assumes retrieval always works
chain = retriever | prompt | llm

# GOOD - handle empty retrieval
def safe_retrieve(query):
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return format_docs(docs)
```

**Anti-Pattern 5: Overly complex single chains**
```python
# BAD - monolithic chain that's hard to debug
mega_chain = step1 | step2 | step3 | step4 | step5 | step6 | step7

# GOOD - break into logical sub-chains with clear names
retrieval_chain = retriever | format_docs
reasoning_chain = prompt | llm | parser
full_chain = {"context": retrieval_chain, "q": RunnablePassthrough()} | reasoning_chain
```

**Anti-Pattern 6: Not using streaming for user-facing applications**
```python
# BAD - user waits for entire response
result = chain.invoke(input)  # 10 second wait, then full response

# GOOD - stream tokens to user
for chunk in chain.stream(input):
    yield chunk  # user sees tokens as they arrive
```

**Anti-Pattern 7: Embedding all documents at once without batching**
```python
# BAD - OOM for large document sets
vectorstore = FAISS.from_documents(million_docs, embeddings)

# GOOD - batch processing
batch_size = 500
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    if i == 0:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)
```

---

### Q38: Performance Optimization Tips

```python
# 1. Use async for I/O-bound operations
results = await chain.abatch(
    inputs,
    config={"max_concurrency": 10}  # parallel execution
)

# 2. Cache embeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    OpenAIEmbeddings(),
    LocalFileStore("./cache/"),
    namespace="embeddings"
)

# 3. Cache LLM responses
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache

set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95
))

# 4. Use smaller models for classification/routing
router_llm = ChatOpenAI(model="gpt-4o-mini")  # fast, cheap
main_llm = ChatOpenAI(model="gpt-4o")          # powerful

# 5. Limit retrieval results
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # not 20

# 6. Use connection pooling for vector stores
# (Pinecone, Weaviate, etc. handle this internally)

# 7. Parallelize independent operations
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summary_chain,
    entities=entity_chain,
    sentiment=sentiment_chain,
)  # all three run concurrently

# 8. Use streaming to reduce perceived latency
# First token arrives much sooner than full response

# 9. Precompute and store embeddings
# Don't re-embed documents on every startup

# 10. Use appropriate chunk sizes
# Smaller chunks = faster retrieval, less context
# Larger chunks = slower retrieval, more context
# Profile and test with your specific use case
```

---

### Q39: How do you handle rate limits and costs?

```python
# 1. Rate limiting with max_concurrency
results = await chain.abatch(
    large_input_list,
    config={"max_concurrency": 5}  # max 5 concurrent API calls
)

# 2. Token tracking
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke(input)
    print(f"Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost:.4f}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")

# 3. Model fallbacks for cost optimization
cheap_chain = prompt | ChatOpenAI(model="gpt-4o-mini") | parser
expensive_chain = prompt | ChatOpenAI(model="gpt-4o") | parser

cost_optimized = cheap_chain.with_fallbacks([expensive_chain])

# 4. Caching to reduce API calls (see Q26)

# 5. Batching similar requests
results = llm.batch(
    [prompt1, prompt2, prompt3],
    config={"max_concurrency": 3}
)
```

---

### Q40: Explain the LangChain ecosystem versioning and migration.

**Answer:**

```
LangChain Ecosystem (2025-2026):
================================

langchain-core (0.3.x)
  - Base abstractions: Runnables, Messages, Prompts, Output Parsers
  - LCEL runtime
  - Minimal dependencies
  - Stable API

langchain (0.3.x)
  - Higher-level chains, agents, retrieval strategies
  - Depends on langchain-core

langchain-community (0.3.x)
  - Third-party integrations (vector stores, loaders, etc.)
  - Being gradually replaced by partner packages

Partner packages (independent versioning):
  - langchain-openai
  - langchain-anthropic
  - langchain-google-genai
  - langchain-google-vertexai
  - langchain-mistralai
  - langchain-aws
  - langchain-pinecone
  - langchain-chroma
  - langchain-weaviate
  - etc.

langgraph (0.2.x+)
  - Stateful orchestration
  - Independent release cycle

langsmith (SDK: 0.2.x)
  - Observability platform SDK
```

**Key Migration Notes (v0.1 -> v0.2 -> v0.3):**
1. Legacy chains (`LLMChain`, `ConversationalRetrievalChain`, etc.) deprecated. Use LCEL.
2. Imports moved: `from langchain.chat_models` -> `from langchain_openai`
3. `langchain.callbacks` -> `langchain_core.callbacks`
4. Memory classes still available but LangGraph state management preferred
5. `.run()` deprecated -> use `.invoke()`
6. Agent classes deprecated -> use LangGraph or `create_react_agent`

---

## QUICK REFERENCE: TOP 20 RAPID-FIRE INTERVIEW QUESTIONS

| # | Question | Short Answer |
|---|----------|-------------|
| 1 | What is LCEL? | Declarative composition of LangChain components using pipe `\|` operator |
| 2 | Difference between `invoke`, `batch`, `stream`? | Single call, multiple calls in parallel, streaming output |
| 3 | What is a Runnable? | Base interface with invoke/batch/stream/async methods |
| 4 | What does `RunnablePassthrough` do? | Passes input through unchanged, used to forward data alongside other transformations |
| 5 | How does LangGraph handle cycles? | Uses conditional edges that can route back to previous nodes |
| 6 | What is a reducer in LangGraph? | Function defining how state updates are merged (e.g., `add` appends lists) |
| 7 | How is memory managed in LangGraph? | Through state + checkpointers (SQLite, Postgres) for persistence |
| 8 | What is `MessagesState`? | Pre-built state with `messages` field and `add_messages` reducer |
| 9 | How does human-in-the-loop work? | `interrupt_before`/`interrupt_after` at compile, or `interrupt()` in nodes |
| 10 | What is tool calling? | LLM generates structured function call requests instead of text |
| 11 | `.with_structured_output()` vs output parsers? | Structured output uses native tool calling; more reliable than prompt-based parsing |
| 12 | What is MMR retrieval? | Maximal Marginal Relevance - balances relevance and diversity in results |
| 13 | What is a checkpointer? | Saves graph state at each step for persistence and recovery |
| 14 | What does `thread_id` do? | Identifies a conversation thread for state persistence |
| 15 | Difference between LangChain and LangGraph? | LangChain = DAG chains; LangGraph = stateful cyclic graphs for agents |
| 16 | What is LangSmith used for? | Tracing, evaluation, monitoring, and debugging LLM applications |
| 17 | How do you deploy LangChain apps? | LangServe (LCEL) or LangGraph Platform (graphs) |
| 18 | What is ReAct? | Reasoning + Acting pattern: LLM alternates between thinking and tool use |
| 19 | What is Corrective RAG? | RAG that validates retrieval quality and falls back to web search if poor |
| 20 | When NOT to use LangChain? | Simple single API calls where the framework adds unnecessary complexity |

---

## BONUS: COMMON CODING CHALLENGES IN INTERVIEWS

### Challenge 1: Implement a chatbot with memory using LangGraph

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: MessagesState):
    messages = [SystemMessage(content="You are a helpful assistant.")] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=MemorySaver())

# Conversation
config = {"configurable": {"thread_id": "user-1"}}

response = app.invoke(
    {"messages": [HumanMessage("I'm Alice, I work at Acme Corp")]},
    config=config
)
print(response["messages"][-1].content)

response = app.invoke(
    {"messages": [HumanMessage("Where do I work?")]},
    config=config
)
print(response["messages"][-1].content)  # "You work at Acme Corp"
```

### Challenge 2: Build a chain that classifies and routes queries

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

class RouteQuery(BaseModel):
    category: Literal["technical", "billing", "general"] = Field(
        description="The category of the user's query"
    )

llm = ChatOpenAI(model="gpt-4o-mini")
classifier = llm.with_structured_output(RouteQuery)

technical_prompt = ChatPromptTemplate.from_template(
    "You are a technical support agent. Help with: {query}"
)
billing_prompt = ChatPromptTemplate.from_template(
    "You are a billing specialist. Help with: {query}"
)
general_prompt = ChatPromptTemplate.from_template(
    "You are a general assistant. Help with: {query}"
)

main_llm = ChatOpenAI(model="gpt-4o")

def classify_and_route(query: str):
    classification = classifier.invoke(f"Classify this query: {query}")
    return {"query": query, "category": classification.category}

chain = (
    RunnableLambda(classify_and_route)
    | RunnableBranch(
        (lambda x: x["category"] == "technical", technical_prompt | main_llm),
        (lambda x: x["category"] == "billing", billing_prompt | main_llm),
        general_prompt | main_llm,  # default
    )
)

result = chain.invoke("My API key isn't working")
```

### Challenge 3: Implement a self-correcting SQL agent

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class SQLAgentState(TypedDict):
    question: str
    sql_query: str
    sql_result: str
    error: str
    attempts: int
    final_answer: str

def generate_sql(state: SQLAgentState) -> dict:
    context = ""
    if state.get("error"):
        context = f"\nPrevious query failed: {state['sql_query']}\nError: {state['error']}\nPlease fix it."

    response = llm.invoke(
        f"Generate a SQL query for: {state['question']}\n"
        f"Database schema: {SCHEMA}\n{context}"
    )
    return {"sql_query": response.content, "attempts": state.get("attempts", 0) + 1}

def execute_sql(state: SQLAgentState) -> dict:
    try:
        result = db.execute(state["sql_query"])
        return {"sql_result": str(result), "error": ""}
    except Exception as e:
        return {"error": str(e), "sql_result": ""}

def check_result(state: SQLAgentState) -> str:
    if state["error"]:
        if state["attempts"] >= 3:
            return "give_up"
        return "retry"
    return "answer"

def generate_answer(state: SQLAgentState) -> dict:
    response = llm.invoke(
        f"Question: {state['question']}\n"
        f"SQL Result: {state['sql_result']}\n"
        f"Generate a natural language answer."
    )
    return {"final_answer": response.content}

def give_up(state: SQLAgentState) -> dict:
    return {"final_answer": f"Sorry, I couldn't query the database after {state['attempts']} attempts."}

graph = StateGraph(SQLAgentState)
graph.add_node("generate_sql", generate_sql)
graph.add_node("execute_sql", execute_sql)
graph.add_node("answer", generate_answer)
graph.add_node("give_up", give_up)

graph.add_edge(START, "generate_sql")
graph.add_edge("generate_sql", "execute_sql")
graph.add_conditional_edges("execute_sql", check_result, {
    "retry": "generate_sql",
    "answer": "answer",
    "give_up": "give_up",
})
graph.add_edge("answer", END)
graph.add_edge("give_up", END)

sql_agent = graph.compile()
```

---

*This guide covers the complete LangChain and LangGraph ecosystem as of early 2026. The framework evolves rapidly -- always check the official documentation at python.langchain.com and langchain-ai.github.io/langgraph for the latest APIs.*
