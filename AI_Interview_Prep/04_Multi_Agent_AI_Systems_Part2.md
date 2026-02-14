# Multi-Agent AI Systems - Complete Interview Guide (Part 2)
# Memory Systems, Tool Use, Error Handling & Reliability
# For AI Engineers 2025-2026

---

## TABLE OF CONTENTS - PART 2

4. Memory in Multi-Agent Systems
5. Tool Use and Function Calling
6. Error Handling and Reliability

---

# ============================================================
# SECTION 4: MEMORY IN MULTI-AGENT SYSTEMS
# ============================================================

## 4.1 Why Memory Matters in Multi-Agent Systems

Without memory, agents are stateless -- they forget everything between conversations
and cannot learn from past interactions. Memory enables:
- **Continuity**: Agents remember past conversations and decisions
- **Learning**: Agents improve over time based on experience
- **Context**: Agents access relevant information without re-processing
- **Coordination**: Agents share knowledge to avoid redundant work

---

## 4.2 Short-Term Memory (Working Memory)

**Definition:** The current conversation context. It's the messages in the ongoing
conversation that an agent can reference. Analogous to human working memory.

**In AutoGen:**
```python
# Short-term memory is the conversation history
# Stored in agent._oai_messages[other_agent]

agent = ConversableAgent(name="Agent", llm_config=llm_config)
# After a conversation, messages are stored:
# agent._oai_messages[partner] = [
#     {"role": "user", "content": "Hello"},
#     {"role": "assistant", "content": "Hi there!"},
#     {"role": "user", "content": "What's 2+2?"},
#     {"role": "assistant", "content": "4"},
# ]

# Short-term memory is limited by the LLM context window
# For GPT-4: ~128K tokens
# For Claude: ~200K tokens

# Managing context window overflow:
# 1. Summarization: Summarize older messages to save space
# 2. Sliding window: Keep only the last N messages
# 3. Selective memory: Keep only relevant messages based on current task
```

**In LangGraph:**
```python
from langgraph.graph import StateGraph, MessagesState

# Messages state automatically manages short-term memory
class AgentState(MessagesState):
    # messages: list  -- inherited, this IS the short-term memory
    current_task: str
    iteration: int

def agent_node(state: AgentState):
    # Agent has access to full message history (short-term memory)
    all_messages = state["messages"]
    # Last message is the most recent context
    last_message = all_messages[-1]
    # Generate response using full context
    response = model.invoke(all_messages)
    return {"messages": [response]}
```

**In CrewAI:**
```python
from crewai import Agent, Task, Crew

# CrewAI manages short-term memory automatically within a crew execution
# Each task execution maintains conversation context
agent = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="You are an experienced researcher",
    memory=True,  # Enable memory (includes short-term)
    verbose=True,
)
```

---

## 4.3 Long-Term Memory

**Definition:** Persistent memory that survives across conversations and sessions.
Typically backed by a vector database (ChromaDB, Pinecone, Weaviate) or
traditional database.

**Implementation with AutoGen Teachability:**
```python
from autogen.agentchat.contrib.capabilities.teachability import Teachability

teachable_agent = ConversableAgent(
    name="LongMemoryAgent",
    system_message="You are an assistant with long-term memory.",
    llm_config=llm_config,
)

teachability = Teachability(
    verbosity=0,
    reset_db=False,            # Persist across sessions
    path_to_db_dir="./long_term_memory_db",
    recall_threshold=1.5,      # Cosine distance threshold for retrieval
    max_num_retrievals=10,     # Max memories to retrieve
)
teachability.add_to_agent(teachable_agent)

# Session 1: Store information
# Agent automatically detects "remember this" or factual statements
# and stores them in ChromaDB

# Session 2: Retrieve information
# Before each response, agent queries ChromaDB with the current topic
# Relevant memories are injected into the system message
```

**Custom Long-Term Memory with Vector DB:**
```python
import chromadb
from sentence_transformers import SentenceTransformer

class LongTermMemory:
    """Custom long-term memory using ChromaDB."""

    def __init__(self, collection_name="agent_memory", persist_dir="./memory_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def store(self, text: str, metadata: dict = None):
        """Store a memory."""
        embedding = self.encoder.encode(text).tolist()
        doc_id = f"mem_{hash(text)}_{int(time.time())}"
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve relevant memories."""
        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        return results["documents"][0] if results["documents"] else []

    def forget(self, doc_id: str):
        """Delete a specific memory."""
        self.collection.delete(ids=[doc_id])

# Usage in an agent
memory = LongTermMemory()

# Store important information
memory.store(
    "The client prefers Python over Java for backend services",
    metadata={"category": "preferences", "client": "acme_corp"}
)
memory.store(
    "Previous project deadline was missed due to underestimated testing time",
    metadata={"category": "lessons_learned", "project": "project_x"}
)

# Retrieve relevant memories before generating a response
relevant_memories = memory.retrieve("What language should we use for the new backend?")
# Returns: ["The client prefers Python over Java for backend services"]
```

---

## 4.4 Episodic Memory

**Definition:** Memory of specific events, interactions, and their outcomes.
Like human episodic memory -- remembering specific experiences, not just facts.

```python
class EpisodicMemory:
    """Stores and retrieves specific interaction episodes."""

    def __init__(self):
        self.episodes = []

    def record_episode(self, episode: dict):
        """Record a complete interaction episode."""
        self.episodes.append({
            "timestamp": time.time(),
            "task": episode["task"],
            "agents_involved": episode["agents"],
            "steps": episode["steps"],          # List of actions taken
            "outcome": episode["outcome"],       # success/failure
            "lessons": episode.get("lessons", ""),
            "duration": episode.get("duration", 0),
        })

    def recall_similar(self, task_description: str, top_k: int = 3):
        """Recall similar past episodes for experience-based reasoning."""
        # In production, use vector similarity
        relevant = []
        for ep in self.episodes:
            if self._is_similar(task_description, ep["task"]):
                relevant.append(ep)
        return sorted(relevant, key=lambda x: x["timestamp"], reverse=True)[:top_k]

    def get_lessons_learned(self, task_type: str) -> list[str]:
        """Extract lessons from past episodes of similar tasks."""
        lessons = []
        for ep in self.episodes:
            if task_type.lower() in ep["task"].lower() and ep["lessons"]:
                lessons.append(ep["lessons"])
        return lessons

# Usage
episodic = EpisodicMemory()

# After completing a task, record the episode
episodic.record_episode({
    "task": "Deploy machine learning model to production",
    "agents": ["DevOps", "MLEngineer", "QA"],
    "steps": [
        "MLEngineer packaged model as Docker container",
        "DevOps set up Kubernetes deployment",
        "QA ran integration tests -- found memory leak",
        "MLEngineer fixed memory leak by adding batch processing",
        "DevOps redeployed successfully",
    ],
    "outcome": "success",
    "lessons": "Always run memory profiling before deploying ML models",
    "duration": 3600,
})

# Before a new similar task, recall past experience
past_deployments = episodic.recall_similar("Deploy new NLP model to production")
# Agent can use past experience to avoid known pitfalls
```

---

## 4.5 Semantic Memory

**Definition:** Structured knowledge about concepts, relationships, and facts.
Like a knowledge graph. Stores "what" things are, not "when" they happened
(that's episodic).

```python
class SemanticMemory:
    """Knowledge graph-based semantic memory."""

    def __init__(self):
        self.knowledge_graph = {}  # {entity: {relation: [targets]}}
        self.concepts = {}          # {concept: description}

    def add_fact(self, subject: str, relation: str, obj: str):
        """Add a fact: subject -[relation]-> object."""
        if subject not in self.knowledge_graph:
            self.knowledge_graph[subject] = {}
        if relation not in self.knowledge_graph[subject]:
            self.knowledge_graph[subject][relation] = []
        self.knowledge_graph[subject][relation].append(obj)

    def add_concept(self, concept: str, description: str):
        """Define a concept."""
        self.concepts[concept] = description

    def query(self, subject: str, relation: str = None) -> dict:
        """Query facts about a subject."""
        if subject not in self.knowledge_graph:
            return {}
        if relation:
            return self.knowledge_graph[subject].get(relation, [])
        return self.knowledge_graph[subject]

    def get_related(self, entity: str) -> list[str]:
        """Get all entities related to the given entity."""
        related = set()
        # Forward relations
        for relation, targets in self.knowledge_graph.get(entity, {}).items():
            related.update(targets)
        # Reverse relations
        for subj, relations in self.knowledge_graph.items():
            for relation, targets in relations.items():
                if entity in targets:
                    related.add(subj)
        return list(related)

# Usage
semantic = SemanticMemory()

# Build knowledge about the system
semantic.add_fact("UserService", "depends_on", "AuthService")
semantic.add_fact("UserService", "depends_on", "DatabaseService")
semantic.add_fact("AuthService", "uses", "JWT")
semantic.add_fact("PaymentService", "depends_on", "UserService")
semantic.add_concept("microservice", "An independently deployable service")

# Agent can query semantic memory
deps = semantic.query("UserService", "depends_on")
# Returns: ["AuthService", "DatabaseService"]

related = semantic.get_related("UserService")
# Returns: ["AuthService", "DatabaseService", "PaymentService"]
```

---

## 4.6 Shared Memory Across Agents

**The Central Challenge:** How do multiple agents share memory without conflicts?

```python
# Pattern 1: Centralized Shared Memory Store
class SharedMemoryStore:
    """Thread-safe shared memory for multi-agent systems."""

    def __init__(self):
        self._store = {}
        self._lock = threading.RLock()
        self._history = []  # Audit trail

    def write(self, key: str, value: any, agent_name: str):
        with self._lock:
            old_value = self._store.get(key)
            self._store[key] = value
            self._history.append({
                "action": "write",
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "agent": agent_name,
                "timestamp": time.time(),
            })

    def read(self, key: str) -> any:
        with self._lock:
            return self._store.get(key)

    def read_all(self) -> dict:
        with self._lock:
            return self._store.copy()

    def get_history(self, key: str = None) -> list:
        if key:
            return [h for h in self._history if h["key"] == key]
        return self._history.copy()

# Pattern 2: Namespace-based Shared Memory
class NamespacedMemory:
    """Each agent has a private namespace plus access to shared namespace."""

    def __init__(self):
        self._namespaces = {"shared": {}}  # "shared" is accessible to all

    def create_namespace(self, agent_name: str):
        self._namespaces[agent_name] = {}

    def write_private(self, agent_name: str, key: str, value: any):
        """Write to agent's private namespace."""
        self._namespaces[agent_name][key] = value

    def write_shared(self, key: str, value: any):
        """Write to the shared namespace (visible to all agents)."""
        self._namespaces["shared"][key] = value

    def read_private(self, agent_name: str, key: str):
        return self._namespaces[agent_name].get(key)

    def read_shared(self, key: str):
        return self._namespaces["shared"].get(key)

    def read_all_accessible(self, agent_name: str) -> dict:
        """Get everything an agent can see (private + shared)."""
        result = {}
        result.update(self._namespaces.get("shared", {}))
        result.update(self._namespaces.get(agent_name, {}))
        return result

# Pattern 3: LangGraph Shared State as Memory
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from operator import add

class MultiAgentState(TypedDict):
    messages: Annotated[list, add]
    # Shared memory fields accessible by all agent nodes
    shared_findings: Annotated[list, add]    # Append-only shared list
    shared_decisions: dict                    # Shared decision register
    agent_scratchpads: dict                   # Per-agent working memory

def agent_a_node(state):
    # Read shared memory
    findings = state.get("shared_findings", [])
    # Write to shared memory
    return {
        "shared_findings": [{"agent": "A", "finding": "New data pattern found"}],
        "agent_scratchpads": {
            **state.get("agent_scratchpads", {}),
            "agent_a": "Working on hypothesis X"
        }
    }
```

**CrewAI Shared Memory:**
```python
from crewai import Agent, Task, Crew, Process

# CrewAI has built-in shared memory across crew members
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    memory=True,              # Enable crew-level memory
    # Memory types enabled:
    # - Short-term: Current crew execution context
    # - Long-term: Persists across executions (uses SQLite/vector DB)
    # - Entity memory: Tracks entities mentioned across tasks
    verbose=True,
)

# All agents in the crew can access shared context from previous tasks
result = crew.kickoff()
```

---

## 4.7 Memory Interview Questions

**Q: How would you implement memory for a multi-agent customer support system?**

A: "I'd implement a layered memory architecture:

1. Short-term memory: The current conversation with the customer, maintained as
   the message history within the active agent session.

2. Long-term memory: Customer profile, past interactions, preferences, stored in
   a vector database. Before each response, the agent retrieves relevant past
   interactions using semantic search on the customer's current query.

3. Episodic memory: Records of how past support tickets were resolved. When a new
   ticket comes in, the system recalls similar past tickets and their resolution
   strategies.

4. Semantic memory: Product knowledge graph -- features, known issues, pricing,
   compatibility. Agents query this for factual answers.

5. Shared memory: When a ticket is transferred between agents (L1 -> L2 -> L3),
   the conversation context and internal notes are shared via a centralized
   store so the next agent has full context.

For the vector database, I'd use ChromaDB or Pinecone with sentence-transformer
embeddings. Memory retrieval would be done before each LLM call, injecting
relevant memories into the system prompt."

**Q: What are the tradeoffs between shared state and message-based memory?**

A: "Shared state (like LangGraph) is simpler to implement -- all agents read/write
a common state object. But it creates coupling (all agents must agree on schema),
can have write conflicts, and state can grow unbounded.

Message-based memory (like AutoGen) provides better agent isolation -- each agent
maintains private conversation histories. But sharing context requires explicit
message passing, and global state is harder to reason about.

For deterministic workflows with well-defined data flow, shared state is better.
For dynamic, conversational systems where agents need autonomy, message-based
memory is better. In practice, hybrid approaches work best -- shared state for
structured data, messages for conversational context."

---

# ============================================================
# SECTION 5: TOOL USE AND FUNCTION CALLING
# ============================================================

## 5.1 How Agents Call Tools

**The Tool-Calling Pipeline:**
1. Agent receives a task/query
2. Agent's LLM determines a tool is needed
3. LLM generates a structured function call (JSON)
4. Runtime/executor parses the function call
5. The actual function is executed
6. Result is returned to the agent
7. Agent incorporates the result and continues

**AutoGen Tool Calling:**
```python
from autogen import ConversableAgent, register_function
from typing import Annotated

# Define tools as Python functions with type annotations
def search_database(
    query: Annotated[str, "The search query"],
    limit: Annotated[int, "Maximum number of results"] = 10,
) -> str:
    """Search the company database for relevant records."""
    # Actual implementation
    results = db.search(query, limit=limit)
    return json.dumps(results)

def send_email(
    to: Annotated[str, "Recipient email address"],
    subject: Annotated[str, "Email subject line"],
    body: Annotated[str, "Email body content"],
) -> str:
    """Send an email to the specified recipient."""
    # Actual implementation
    email_service.send(to=to, subject=subject, body=body)
    return f"Email sent successfully to {to}"

def create_jira_ticket(
    title: Annotated[str, "Ticket title"],
    description: Annotated[str, "Ticket description"],
    priority: Annotated[str, "Priority: low, medium, high, critical"] = "medium",
) -> str:
    """Create a JIRA ticket for tracking."""
    ticket = jira.create_issue(
        project="PROJ",
        summary=title,
        description=description,
        priority=priority,
    )
    return f"Created ticket: {ticket.key}"

# Create agents
assistant = ConversableAgent(
    name="SupportAssistant",
    system_message="""You are a customer support agent with access to tools.
    Use search_database to look up customer information.
    Use send_email for follow-ups.
    Use create_jira_ticket for escalations.""",
    llm_config=llm_config,
)

executor = ConversableAgent(
    name="ToolExecutor",
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
)

# Register all tools
for tool_func in [search_database, send_email, create_jira_ticket]:
    register_function(
        tool_func,
        caller=assistant,
        executor=executor,
        name=tool_func.__name__,
        description=tool_func.__doc__,
    )

# The assistant will automatically call tools when appropriate
executor.initiate_chat(
    assistant,
    message="Customer john@example.com is reporting a billing issue. Look up their account and create a ticket."
)
```

**LangGraph Tool Calling:**
```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the company database for relevant records."""
    results = db.search(query, limit=limit)
    return json.dumps(results)

@tool
def calculate_metrics(data: str, metric_type: str) -> str:
    """Calculate business metrics from data."""
    # Implementation
    return f"Calculated {metric_type}: ..."

# Create a ReAct agent with tools
model = ChatOpenAI(model="gpt-4o")
tools = [search_database, calculate_metrics]

agent = create_react_agent(model, tools)
result = agent.invoke({"messages": [("user", "Calculate our Q4 revenue")]})
```

**CrewAI Tool Calling:**
```python
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# CrewAI custom tool
class SearchDatabaseInput(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Max results")

class SearchDatabaseTool(BaseTool):
    name: str = "search_database"
    description: str = "Search the company database"
    args_schema: type[BaseModel] = SearchDatabaseInput

    def _run(self, query: str, limit: int = 10) -> str:
        results = db.search(query, limit=limit)
        return json.dumps(results)

# Assign tools to agents
researcher = Agent(
    role="Data Researcher",
    goal="Find relevant data",
    backstory="Expert data researcher",
    tools=[SearchDatabaseTool()],
    verbose=True,
)
```

---

## 5.2 Dynamic Tool Discovery

Agents discover available tools at runtime rather than having them hardcoded.

```python
class ToolRegistry:
    """Central registry for dynamic tool discovery."""

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, name: str, func: callable, description: str,
                 category: str = "general", parameters: dict = None,
                 requires_approval: bool = False):
        """Register a tool in the registry."""
        self._tools[name] = {
            "function": func,
            "description": description,
            "category": category,
            "parameters": parameters or {},
            "requires_approval": requires_approval,
            "usage_count": 0,
        }
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

    def discover(self, query: str = None, category: str = None) -> list[dict]:
        """Discover tools matching criteria."""
        results = []
        for name, tool in self._tools.items():
            if category and tool["category"] != category:
                continue
            if query and query.lower() not in tool["description"].lower():
                continue
            results.append({
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"],
                "requires_approval": tool["requires_approval"],
            })
        return results

    def execute(self, name: str, **kwargs):
        """Execute a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        tool = self._tools[name]
        tool["usage_count"] += 1
        return tool["function"](**kwargs)

    def get_openai_functions_schema(self, category: str = None) -> list[dict]:
        """Generate OpenAI function calling schema for discovered tools."""
        schemas = []
        tools = self.discover(category=category)
        for tool in tools:
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                }
            })
        return schemas

# Usage
registry = ToolRegistry()

# Register tools from different modules/services
registry.register(
    name="search_web",
    func=web_search,
    description="Search the web for information",
    category="research",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
)

registry.register(
    name="execute_sql",
    func=run_sql_query,
    description="Execute a SQL query against the database",
    category="data",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    requires_approval=True,  # Dangerous tool needs approval
)

# Agent dynamically discovers relevant tools
data_tools = registry.discover(category="data")
research_tools = registry.discover(category="research")

# Generate LLM function schemas dynamically
llm_config = {
    "config_list": config_list,
    "tools": registry.get_openai_functions_schema(category="data"),
}
```

---

## 5.3 Tool Chaining

Tools are called in sequence, with the output of one becoming the input to the next.

```python
# Tool Chaining Pattern
class ToolChain:
    """Execute a chain of tools where outputs flow to inputs."""

    def __init__(self):
        self.steps = []

    def add_step(self, tool_name: str, input_mapping: dict = None):
        """Add a step to the chain.
        input_mapping: {param_name: "previous_result" | "step_N_result" | literal}
        """
        self.steps.append({
            "tool_name": tool_name,
            "input_mapping": input_mapping or {},
        })
        return self  # For fluent API

    def execute(self, registry: ToolRegistry, initial_input: dict) -> list[dict]:
        """Execute the chain."""
        results = []
        current_input = initial_input

        for i, step in enumerate(self.steps):
            # Resolve input mappings
            kwargs = {}
            for param, source in step["input_mapping"].items():
                if source == "previous_result":
                    kwargs[param] = results[-1]["output"] if results else None
                elif source.startswith("step_"):
                    step_idx = int(source.split("_")[1])
                    kwargs[param] = results[step_idx]["output"]
                else:
                    kwargs[param] = current_input.get(source, source)

            # Execute the tool
            output = registry.execute(step["tool_name"], **kwargs)
            results.append({
                "step": i,
                "tool": step["tool_name"],
                "input": kwargs,
                "output": output,
            })

        return results

# Usage: Research -> Summarize -> Translate chain
chain = ToolChain()
chain.add_step("search_web", {"query": "topic"})
chain.add_step("summarize_text", {"text": "previous_result"})
chain.add_step("translate", {"text": "previous_result", "target_lang": "language"})

results = chain.execute(
    registry,
    initial_input={"topic": "quantum computing advances", "language": "Spanish"}
)
```

**LLM-Driven Tool Chaining (Agent decides the chain):**
```python
# The agent itself decides which tools to chain
# This is the ReAct (Reason + Act) pattern

# AutoGen handles this naturally:
assistant = AssistantAgent(
    name="Assistant",
    system_message="""You have access to these tools:
    1. search_web(query) - Search the web
    2. extract_data(url) - Extract structured data from a URL
    3. analyze_data(data) - Analyze structured data
    4. generate_report(analysis) - Generate a report

    For complex tasks, chain tools together:
    search -> extract -> analyze -> report""",
    llm_config=llm_config,
)
# The LLM will naturally chain tools based on the task
```

---

## 5.4 Multi-Agent Tool Coordination

When multiple agents share tools, coordination is needed.

```python
# Tool Access Control
class ManagedToolRegistry(ToolRegistry):
    """Tool registry with access control and coordination."""

    def __init__(self):
        super().__init__()
        self._permissions: dict[str, set[str]] = {}  # {agent: {tool_names}}
        self._locks: dict[str, threading.Lock] = {}    # {tool_name: lock}
        self._rate_limits: dict[str, dict] = {}        # {tool_name: {limit, window}}

    def grant_access(self, agent_name: str, tool_names: list[str]):
        """Grant an agent access to specific tools."""
        self._permissions[agent_name] = set(tool_names)

    def can_access(self, agent_name: str, tool_name: str) -> bool:
        """Check if an agent has access to a tool."""
        return tool_name in self._permissions.get(agent_name, set())

    def execute_as(self, agent_name: str, tool_name: str, **kwargs):
        """Execute a tool on behalf of an agent (with access check)."""
        if not self.can_access(agent_name, tool_name):
            raise PermissionError(
                f"Agent '{agent_name}' does not have access to tool '{tool_name}'"
            )

        # Acquire lock for exclusive-access tools
        if tool_name in self._locks:
            with self._locks[tool_name]:
                return self.execute(tool_name, **kwargs)
        return self.execute(tool_name, **kwargs)

    def set_exclusive(self, tool_name: str):
        """Mark a tool as exclusive (only one agent can use at a time)."""
        self._locks[tool_name] = threading.Lock()

# Usage
managed_registry = ManagedToolRegistry()
managed_registry.register("read_db", read_db_func, "Read from database", "data")
managed_registry.register("write_db", write_db_func, "Write to database", "data")
managed_registry.register("send_email", send_email_func, "Send email", "comm")

# Grant different access levels
managed_registry.grant_access("researcher", ["read_db"])
managed_registry.grant_access("admin", ["read_db", "write_db", "send_email"])
managed_registry.grant_access("notifier", ["send_email"])

# Make write_db exclusive (only one agent at a time)
managed_registry.set_exclusive("write_db")
```

---

## 5.5 Tool Use Interview Questions

**Q: How does function calling work in AutoGen?**

A: "AutoGen uses OpenAI-compatible function calling. You define Python functions
with type annotations and docstrings, then register them with `register_function`,
specifying a `caller` agent (whose LLM decides when to call the tool) and an
`executor` agent (which actually runs the function). When the caller's LLM generates
a function call, AutoGen intercepts it, routes it to the executor, executes the
function, and returns the result to the caller. This separation of concerns is
important -- the LLM-equipped agent reasons about WHEN to use tools, while the
executor handles the actual execution, which can include safety checks."

**Q: How would you handle a tool that's slow or unreliable?**

A: "Several strategies:
1. **Timeout**: Set execution timeouts so slow tools don't block the agent
2. **Retry with backoff**: Implement exponential backoff for transient failures
3. **Caching**: Cache results for deterministic tools to avoid redundant calls
4. **Fallback tools**: Register alternative tools that provide similar capability
5. **Circuit breaker**: If a tool fails N times, temporarily disable it and notify
6. **Async execution**: Run slow tools asynchronously while the agent continues
7. **Rate limiting**: Prevent agents from overwhelming external APIs"

---

# ============================================================
# SECTION 6: ERROR HANDLING AND RELIABILITY
# ============================================================

## 6.1 Fallback Strategies

```python
# Fallback Pattern: Try primary, fall back to alternatives
class FallbackChain:
    """Try a sequence of strategies until one succeeds."""

    def __init__(self):
        self.strategies = []

    def add_strategy(self, name: str, func: callable, condition: callable = None):
        """Add a fallback strategy.
        condition: Optional function that returns True if this strategy should be tried.
        """
        self.strategies.append({
            "name": name,
            "func": func,
            "condition": condition or (lambda: True),
        })

    def execute(self, *args, **kwargs):
        errors = []
        for strategy in self.strategies:
            if not strategy["condition"]():
                continue
            try:
                result = strategy["func"](*args, **kwargs)
                return {"result": result, "strategy_used": strategy["name"]}
            except Exception as e:
                errors.append({"strategy": strategy["name"], "error": str(e)})

        raise RuntimeError(f"All strategies failed: {errors}")

# Usage
fallback = FallbackChain()
fallback.add_strategy("gpt4", lambda msg: call_gpt4(msg))
fallback.add_strategy("gpt35", lambda msg: call_gpt35(msg))
fallback.add_strategy("local_model", lambda msg: call_local_model(msg))
fallback.add_strategy("cached_response", lambda msg: get_cached(msg))

result = fallback.execute("Analyze this data...")

# AutoGen has built-in model fallback:
llm_config = {
    "config_list": [
        {"model": "gpt-4", "api_key": "key1"},         # Try first
        {"model": "gpt-3.5-turbo", "api_key": "key2"}, # Fallback
    ],
    "temperature": 0.7,
}
# AutoGen automatically tries the next model if the first fails
```

---

## 6.2 Retry Mechanisms

```python
import time
import random
from functools import wraps

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    if attempt == max_retries:
                        raise
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    if jitter:
                        delay *= (0.5 + random.random())
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
        return wrapper
    return decorator

# Usage with agent tool calls
@retry_with_backoff(max_retries=3, retryable_exceptions=(TimeoutError, ConnectionError))
def call_external_api(query: str) -> str:
    """Call an external API with retry logic."""
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=10)
    response.raise_for_status()
    return response.json()

# AutoGen agent-level retry
class RetryableAgent(ConversableAgent):
    """Agent that retries on LLM failures."""

    def __init__(self, *args, max_retries=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries

    def generate_reply(self, messages=None, sender=None, **kwargs):
        for attempt in range(self.max_retries):
            try:
                reply = super().generate_reply(messages, sender, **kwargs)
                if reply:  # Validate response
                    return reply
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"Failed after {self.max_retries} attempts: {e}"
                time.sleep(2 ** attempt)
        return "Unable to generate a response."
```

---

## 6.3 Circuit Breaker Pattern

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker for agent tool calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0

    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Wait {self.recovery_timeout}s"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # 3 successes to fully close
                self.state = CircuitState.CLOSED
                self.success_count = 0

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class CircuitBreakerOpenError(Exception):
    pass

# Usage with agents
api_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

def safe_api_call(query: str) -> str:
    """API call protected by circuit breaker."""
    try:
        return api_breaker.call(call_external_api, query)
    except CircuitBreakerOpenError:
        return "API is temporarily unavailable. Using cached results."
    except Exception as e:
        return f"API call failed: {e}"
```

---

## 6.4 Graceful Degradation

```python
class ResilientAgent(ConversableAgent):
    """Agent with graceful degradation capabilities."""

    def __init__(self, *args, degradation_levels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.degradation_levels = degradation_levels or [
            {
                "name": "full",
                "description": "Full capability with GPT-4 and all tools",
                "model": "gpt-4",
                "tools_enabled": True,
                "max_tokens": 4096,
            },
            {
                "name": "reduced",
                "description": "Reduced capability with GPT-3.5",
                "model": "gpt-3.5-turbo",
                "tools_enabled": True,
                "max_tokens": 2048,
            },
            {
                "name": "minimal",
                "description": "Minimal capability, no tools",
                "model": "gpt-3.5-turbo",
                "tools_enabled": False,
                "max_tokens": 1024,
            },
            {
                "name": "cached",
                "description": "Cached responses only",
                "model": None,
                "tools_enabled": False,
                "max_tokens": 0,
            },
        ]
        self.current_level = 0

    def degrade(self):
        """Move to the next degradation level."""
        if self.current_level < len(self.degradation_levels) - 1:
            self.current_level += 1
            level = self.degradation_levels[self.current_level]
            print(f"Degrading to level: {level['name']} - {level['description']}")
            # Update agent configuration
            if level["model"]:
                self.llm_config["config_list"][0]["model"] = level["model"]

    def generate_reply(self, messages=None, sender=None, **kwargs):
        try:
            reply = super().generate_reply(messages, sender, **kwargs)
            return reply
        except Exception as e:
            self.degrade()
            if self.current_level >= len(self.degradation_levels):
                return "System is experiencing issues. Please try again later."
            return self.generate_reply(messages, sender, **kwargs)
```

---

## 6.5 Multi-Agent Error Handling Patterns

```python
# Pattern 1: Supervisor-based error recovery
supervisor = ConversableAgent(
    name="Supervisor",
    system_message="""You monitor other agents for errors.
    When an agent reports an error:
    1. Analyze the error type
    2. If transient (timeout, rate limit): retry the task
    3. If input error: reformulate the input and retry
    4. If capability error: reassign to a different agent
    5. If critical: escalate to human
    Always explain what went wrong and what you're doing about it.""",
    llm_config=llm_config,
)

# Pattern 2: Error recovery with checkpointing
class CheckpointedWorkflow:
    """Workflow that saves progress and can resume from checkpoints."""

    def __init__(self):
        self.checkpoints = {}
        self.current_step = 0

    def save_checkpoint(self, step: int, state: dict):
        self.checkpoints[step] = {
            "state": state,
            "timestamp": time.time(),
        }

    def restore_checkpoint(self, step: int) -> dict:
        if step in self.checkpoints:
            return self.checkpoints[step]["state"]
        raise ValueError(f"No checkpoint at step {step}")

    def resume_from_last(self) -> tuple[int, dict]:
        if not self.checkpoints:
            return 0, {}
        last_step = max(self.checkpoints.keys())
        return last_step, self.checkpoints[last_step]["state"]

# Pattern 3: Consensus-based error detection
# Multiple agents perform the same task; if results disagree, flag an error
class ConsensusChecker:
    """Detect errors by comparing outputs from multiple agents."""

    def __init__(self, agents: list, threshold: float = 0.7):
        self.agents = agents
        self.threshold = threshold

    def check(self, task: str) -> dict:
        """Run task on all agents and check consensus."""
        results = []
        for agent in self.agents:
            try:
                result = agent.generate_reply(
                    messages=[{"role": "user", "content": task}]
                )
                results.append(result)
            except Exception as e:
                results.append(None)

        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return {"consensus": False, "error": "All agents failed"}

        # Check agreement (simplified -- in production use semantic similarity)
        agreement = self._calculate_agreement(valid_results)
        return {
            "consensus": agreement >= self.threshold,
            "agreement_score": agreement,
            "results": results,
            "majority_answer": self._get_majority(valid_results),
        }
```

---

## 6.6 Error Handling Interview Questions

**Q: How do you handle LLM hallucinations in a multi-agent system?**

A: "Multiple strategies:
1. **Verification agent**: A dedicated agent that fact-checks other agents' outputs
   against known data sources or tools (search, database lookups).
2. **Consensus**: Run the same query through multiple agents with different
   temperatures or models. If they agree, confidence is higher.
3. **Grounding**: Force agents to cite sources and use RAG to ground responses
   in actual documents.
4. **Self-consistency**: Ask the same agent to verify its own response from
   a different angle.
5. **Structured output**: Force agents to output structured JSON with explicit
   confidence scores and source references.
6. **Human-in-the-loop**: For high-stakes decisions, require human verification."

**Q: What happens when an agent in a multi-agent pipeline fails?**

A: "The failure mode depends on the architecture:
- **Sequential pipeline**: The entire pipeline stops. Solution: checkpointing at
  each stage so you can retry from the last successful step.
- **Orchestrator-worker**: The orchestrator detects the failure and can reassign
  the task to another worker or retry with modified parameters.
- **Peer-to-peer**: Other agents may not notice the failure. Solution: heartbeat
  monitoring and timeout detection.

My recommended approach is implementing a supervisor pattern with:
1. Timeouts on every agent call
2. Retry with exponential backoff (3 attempts)
3. Circuit breaker to avoid hammering failed services
4. Graceful degradation (fall back to simpler model or cached response)
5. Dead letter queue for failed tasks that need human review
6. Checkpointing to enable resume from last known good state"

**Q: How do you test multi-agent systems?**

A: "Testing at multiple levels:
1. **Unit tests**: Test each agent in isolation with mocked LLM responses
2. **Integration tests**: Test agent pairs (e.g., assistant + executor)
3. **Scenario tests**: Run complete multi-agent workflows with predefined inputs
   and expected outcomes
4. **Adversarial tests**: Feed agents malicious or edge-case inputs
5. **Regression tests**: Record and replay past conversations
6. **Cost tests**: Ensure token usage stays within budget
7. **Performance tests**: Measure latency, throughput, and scalability
8. **Determinism tests**: Run same input multiple times, check consistency

Tools: pytest with fixtures for agents, LLM response mocking, conversation recording."
