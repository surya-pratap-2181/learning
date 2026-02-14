# Multi-Agent AI Systems - Complete Interview Guide (Part 1)
# Architecture Patterns, AG2/AutoGen Deep Dive, Communication Patterns
# For AI Engineers 2025-2026

---

## TABLE OF CONTENTS - PART 1

1. Multi-Agent Architecture Patterns
2. AG2/AutoGen Framework Deep Dive
3. Agent Communication Patterns

---

# ============================================================
# SECTION 1: MULTI-AGENT ARCHITECTURE PATTERNS
# ============================================================

## 1.1 What Are Multi-Agent Systems?

A Multi-Agent System (MAS) consists of multiple autonomous AI agents that interact,
collaborate, or compete to solve complex tasks that would be difficult or impossible
for a single agent. Each agent has its own capabilities, knowledge, and objectives.

**Key Properties:**
- **Autonomy**: Each agent operates independently with its own decision-making
- **Social Ability**: Agents communicate with each other via defined protocols
- **Reactivity**: Agents perceive and respond to changes in their environment
- **Proactiveness**: Agents take initiative to achieve their goals

---

## 1.2 Architecture Pattern: Orchestrator-Worker

**How It Works:**
A central orchestrator agent receives a complex task, decomposes it into subtasks,
delegates them to specialized worker agents, collects results, and synthesizes
a final output.

**When to Use:**
- Complex tasks that can be decomposed into independent subtasks
- When you need centralized control and monitoring
- When subtasks require different specialized capabilities
- When you need deterministic workflow execution

**When NOT to Use:**
- Simple tasks that don't need decomposition
- When the orchestrator becomes a bottleneck
- When tasks require real-time peer-to-peer negotiation

```python
# Orchestrator-Worker Pattern - Conceptual Implementation
from autogen import ConversableAgent, GroupChat, GroupChatManager

# Orchestrator decomposes and delegates
orchestrator = ConversableAgent(
    name="Orchestrator",
    system_message="""You are a task orchestrator. When given a complex task:
    1. Break it into subtasks
    2. Assign each subtask to the most appropriate specialist
    3. Collect and synthesize results
    4. Provide the final consolidated answer""",
    llm_config={"config_list": config_list}
)

# Specialized workers
researcher = ConversableAgent(
    name="Researcher",
    system_message="You are a research specialist. Gather and analyze information.",
    llm_config={"config_list": config_list}
)

coder = ConversableAgent(
    name="Coder",
    system_message="You are a coding specialist. Write clean, tested code.",
    llm_config={"config_list": config_list}
)

reviewer = ConversableAgent(
    name="Reviewer",
    system_message="You review work from other agents for quality and correctness.",
    llm_config={"config_list": config_list}
)
```

**Interview Answer Template:**
"The orchestrator-worker pattern uses a central coordinator that decomposes complex
tasks and delegates subtasks to specialized workers. It's ideal when you need
centralized control, task decomposition, and result aggregation. The tradeoff is
that the orchestrator can become a single point of failure and a bottleneck."

---

## 1.3 Architecture Pattern: Hierarchical

**How It Works:**
Agents are organized in a tree-like hierarchy. Higher-level agents manage and
coordinate lower-level agents. Each level can have its own sub-orchestrators
managing their own teams of workers.

**When to Use:**
- Very large, complex systems with many agents
- When tasks have natural hierarchical decomposition
- Enterprise workflows with approval chains
- When you need different levels of abstraction

```python
# Hierarchical Pattern - Multi-Level Agent Structure

# Top-level manager
project_manager = ConversableAgent(
    name="ProjectManager",
    system_message="""You manage the entire project. Delegate to team leads:
    - Frontend Lead: for UI/UX tasks
    - Backend Lead: for API and database tasks
    - QA Lead: for testing tasks
    Synthesize results from all teams.""",
    llm_config=llm_config
)

# Mid-level team leads (each manages their own sub-agents)
frontend_lead = ConversableAgent(
    name="FrontendLead",
    system_message="""You lead the frontend team. Break frontend tasks into:
    - Component design
    - State management
    - Styling and responsiveness
    Coordinate with your team members and report back.""",
    llm_config=llm_config
)

backend_lead = ConversableAgent(
    name="BackendLead",
    system_message="You lead the backend team. Manage API design, database, services.",
    llm_config=llm_config
)

# Worker-level agents
react_developer = ConversableAgent(
    name="ReactDeveloper",
    system_message="You are a React specialist. Build React components.",
    llm_config=llm_config
)

api_developer = ConversableAgent(
    name="APIDeveloper",
    system_message="You design and implement REST/GraphQL APIs.",
    llm_config=llm_config
)

# In AutoGen, this can be implemented with nested GroupChats
# Each team lead has its own GroupChat with its workers
frontend_group = GroupChat(
    agents=[frontend_lead, react_developer],
    messages=[],
    max_round=5
)

backend_group = GroupChat(
    agents=[backend_lead, api_developer],
    messages=[],
    max_round=5
)

# Top-level group coordinates team leads
top_group = GroupChat(
    agents=[project_manager, frontend_lead, backend_lead],
    messages=[],
    max_round=10
)
```

**Key Difference from Orchestrator-Worker:**
Hierarchical adds multiple levels of management. An orchestrator-worker is flat
(one level), while hierarchical can be N levels deep.

---

## 1.4 Architecture Pattern: Peer-to-Peer (Decentralized)

**How It Works:**
All agents are equal peers with no central coordinator. Each agent can communicate
directly with any other agent. Agents negotiate, share information, and
collaboratively arrive at solutions.

**When to Use:**
- When no single agent has enough context to coordinate
- Brainstorming and creative problem-solving
- When agents need to negotiate or reach consensus
- Distributed systems where centralization is impractical

**When NOT to Use:**
- When you need deterministic, reproducible workflows
- When clear task decomposition is possible
- When accountability and audit trails are important

```python
# Peer-to-Peer Pattern using AutoGen GroupChat
# All agents speak freely, no designated orchestrator

agent_alice = ConversableAgent(
    name="Alice",
    system_message="You are an AI ethics expert. Contribute your perspective.",
    llm_config=llm_config
)

agent_bob = ConversableAgent(
    name="Bob",
    system_message="You are a technology strategist. Contribute your perspective.",
    llm_config=llm_config
)

agent_carol = ConversableAgent(
    name="Carol",
    system_message="You are an economist. Contribute your perspective.",
    llm_config=llm_config
)

# GroupChat with "auto" speaker selection enables peer-to-peer
peer_chat = GroupChat(
    agents=[agent_alice, agent_bob, agent_carol],
    messages=[],
    max_round=15,
    speaker_selection_method="auto"  # LLM decides who speaks next
)

manager = GroupChatManager(groupchat=peer_chat, llm_config=llm_config)
agent_alice.initiate_chat(manager, message="Let's discuss AI regulation impacts.")
```

---

## 1.5 Architecture Pattern: Sequential (Pipeline)

**How It Works:**
Agents are arranged in a linear pipeline. The output of one agent becomes the
input of the next. Each agent performs a specific transformation or enrichment
step.

**When to Use:**
- When tasks have natural sequential stages
- Data processing pipelines
- Content creation workflows (research -> write -> edit -> review)
- When each step depends on the previous step's output

```python
# Sequential Pipeline Pattern

researcher = ConversableAgent(
    name="Researcher",
    system_message="Research the given topic thoroughly. Output structured findings.",
    llm_config=llm_config
)

writer = ConversableAgent(
    name="Writer",
    system_message="Take research findings and write a polished article.",
    llm_config=llm_config
)

editor = ConversableAgent(
    name="Editor",
    system_message="Edit the article for clarity, grammar, and style.",
    llm_config=llm_config
)

fact_checker = ConversableAgent(
    name="FactChecker",
    system_message="Verify all claims in the article. Flag inaccuracies.",
    llm_config=llm_config
)

# Sequential execution using GroupChat with round_robin
sequential_chat = GroupChat(
    agents=[researcher, writer, editor, fact_checker],
    messages=[],
    max_round=4,
    speaker_selection_method="round_robin"  # Forces sequential order
)

manager = GroupChatManager(groupchat=sequential_chat, llm_config=llm_config)
researcher.initiate_chat(manager, message="Write an article about quantum computing.")

# Alternative: Using AutoGen's sequential chat feature
researcher.initiate_chats([
    {"recipient": writer, "message": "Research findings: ...", "max_turns": 2},
    {"recipient": editor, "message": "Please edit this article", "max_turns": 2},
    {"recipient": fact_checker, "message": "Please verify facts", "max_turns": 2},
])
```

---

## 1.6 Architecture Pattern: Round-Robin

**How It Works:**
Agents take turns speaking in a fixed, rotating order. Each agent gets equal
opportunity to contribute. After all agents have spoken, the cycle repeats.

**When to Use:**
- When all agents' perspectives are equally important
- Brainstorming sessions
- Review processes where every reviewer must weigh in
- Ensuring no agent dominates the conversation

```python
# Round-Robin Pattern
group_chat = GroupChat(
    agents=[agent1, agent2, agent3, agent4],
    messages=[],
    max_round=12,  # 3 full cycles of 4 agents
    speaker_selection_method="round_robin"
)
```

**Difference from Sequential:**
Round-robin cycles repeatedly through all agents, while sequential is typically
a single pass through the pipeline.

---

## 1.7 Architecture Pattern: Debate / Adversarial

**How It Works:**
Two or more agents take opposing positions and debate. A judge agent evaluates
the arguments and determines the best solution. This leverages adversarial
dynamics to stress-test ideas and find weaknesses.

**When to Use:**
- Decision-making under uncertainty
- When you need to evaluate tradeoffs
- Red-teaming and security analysis
- Code review with devil's advocate
- When you want to reduce hallucination through cross-verification

```python
# Debate/Adversarial Pattern

proposer = ConversableAgent(
    name="Proposer",
    system_message="""You propose solutions and defend them with evidence.
    Make the strongest case possible for your approach.
    Address counterarguments directly.""",
    llm_config=llm_config
)

critic = ConversableAgent(
    name="Critic",
    system_message="""You critically evaluate proposed solutions.
    Find weaknesses, edge cases, and potential failures.
    Challenge assumptions and demand evidence.
    Play devil's advocate constructively.""",
    llm_config=llm_config
)

judge = ConversableAgent(
    name="Judge",
    system_message="""You are an impartial judge. Listen to both the proposer
    and critic. Evaluate the strength of arguments. Synthesize the best
    solution incorporating valid points from both sides.
    When you're ready, say FINAL_DECISION: followed by your ruling.""",
    llm_config=llm_config
)

def is_final_decision(msg):
    return "FINAL_DECISION:" in msg.get("content", "")

debate_chat = GroupChat(
    agents=[proposer, critic, judge],
    messages=[],
    max_round=10,
    speaker_selection_method="auto"
)

manager = GroupChatManager(groupchat=debate_chat, llm_config=llm_config,
                           is_termination_msg=is_final_decision)
proposer.initiate_chat(manager, message="Propose: We should use microservices for the new platform.")
```

---

## 1.8 Architecture Pattern: Supervisor

**How It Works:**
A supervisor agent monitors all worker agents, handles exceptions, reassigns
tasks when agents fail, and ensures quality standards. Unlike an orchestrator
that mainly delegates, a supervisor actively monitors execution.

**When to Use:**
- Production systems requiring high reliability
- When agent outputs need quality gates
- When tasks may fail and need reassignment
- When you need real-time monitoring and intervention

```python
# Supervisor Pattern

supervisor = ConversableAgent(
    name="Supervisor",
    system_message="""You are a supervisor agent. Your responsibilities:
    1. Monitor output quality from all workers
    2. If a worker's output is below quality, request a redo
    3. If a worker fails repeatedly, reassign the task to another worker
    4. Ensure all tasks complete within the time budget
    5. Escalate to human if critical failures occur

    Quality criteria:
    - Code must include error handling
    - Research must cite sources
    - All outputs must be factually accurate""",
    llm_config=llm_config
)

# LangGraph Supervisor Pattern (more explicit control flow)
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent

def supervisor_node(state: MessagesState):
    """Supervisor decides which worker to call next or to finish."""
    response = model.invoke([
        {"role": "system", "content": "Route to: researcher, coder, or FINISH"},
        *state["messages"]
    ])
    return {"next": response.content}

# Build the graph
workflow = StateGraph(MessagesState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("coder", coder_agent)
workflow.add_conditional_edges("supervisor", route_function)
```

---

## 1.9 Architecture Comparison Matrix

```
| Pattern           | Centralized? | Complexity | Fault Tolerance | Best For                    |
|-------------------|-------------|------------|-----------------|------------------------------|
| Orchestrator      | Yes         | Medium     | Low (SPOF)      | Task decomposition           |
| Hierarchical      | Yes (multi) | High       | Medium          | Large enterprise systems     |
| Peer-to-Peer      | No          | High       | High            | Brainstorming, negotiation   |
| Sequential        | No          | Low        | Low             | Pipelines, ETL               |
| Round-Robin       | No          | Low        | Medium          | Equal-voice discussions      |
| Debate/Adversarial| No          | Medium     | Medium          | Decision-making, red-teaming |
| Supervisor        | Yes         | High       | High            | Production reliability       |
```

---

## 1.10 Interview Questions on Architecture Patterns

**Q: How would you choose between an orchestrator and a supervisor pattern?**

A: "An orchestrator primarily handles task decomposition and delegation -- it breaks
a complex task into subtasks and assigns them. A supervisor goes further by actively
monitoring execution quality, handling failures, and reassigning work. I'd choose
an orchestrator for straightforward workflows where tasks are well-defined, and a
supervisor for production systems where reliability is critical and agents might fail.
The supervisor adds overhead but provides quality gates and fault tolerance."

**Q: What are the failure modes of each architecture pattern?**

A: "Orchestrator/Hierarchical: Single point of failure at the coordinator level.
If the orchestrator goes down, the entire system stalls. Mitigation: redundant
orchestrators or checkpointing.

Sequential: A failure in any stage blocks the entire pipeline. Mitigation: retry
logic, dead letter queues, checkpointing between stages.

Peer-to-Peer: Can lead to infinite loops, circular reasoning, or agents talking
past each other. Mitigation: max rounds, convergence detection, moderator agent.

Debate: Agents may get stuck in unproductive disagreement. Mitigation: judge
agent with authority to end debate, time/round limits."

**Q: How do you prevent infinite loops in multi-agent conversations?**

A: "Multiple strategies:
1. `max_round` parameter in GroupChat to cap conversation turns
2. `is_termination_msg` callback that detects termination keywords
3. Token/cost budgets that halt execution when exceeded
4. Convergence detection -- if agents repeat the same content, terminate
5. Timeout mechanisms at the infrastructure level
6. A supervisor agent that can forcibly end conversations"

---

# ============================================================
# SECTION 2: AG2/AUTOGEN FRAMEWORK DEEP DIVE
# ============================================================

## 2.1 AutoGen Overview and History

**AutoGen** was originally developed by Microsoft Research. In late 2024, the project
transitioned to the open-source **AG2** community (ag2ai.github.io). The framework
enables building multi-agent conversational AI systems.

**Key Philosophy:**
- Agents are conversable -- they communicate via natural language messages
- Multi-agent conversations are the primary abstraction
- Supports human-in-the-loop at any point
- Code execution is a first-class capability
- Highly customizable through subclassing

**Two Versions (Important for Interviews):**
- **AutoGen 0.2.x** (stable): The widely-used version with ConversableAgent
- **AutoGen 0.4.x / AG2**: Major rewrite with an asynchronous, event-driven architecture
- Know both -- interviewers may ask about either

---

## 2.2 ConversableAgent (The Foundation)

`ConversableAgent` is the base class for ALL agents in AutoGen. Every agent type
inherits from it.

**Key Attributes:**
- `name`: Unique identifier for the agent
- `system_message`: The system prompt defining agent behavior
- `llm_config`: LLM configuration (model, API key, temperature, etc.)
- `human_input_mode`: "ALWAYS", "TERMINATE", or "NEVER"
- `max_consecutive_auto_reply`: Limits auto-replies before requesting human input
- `code_execution_config`: Configuration for executing code blocks
- `function_map`: Mapping of function names to callable functions

```python
from autogen import ConversableAgent

agent = ConversableAgent(
    name="MyAgent",
    system_message="You are a helpful AI assistant specialized in Python.",
    llm_config={
        "config_list": [
            {
                "model": "gpt-4",
                "api_key": "your-api-key",
            }
        ],
        "temperature": 0.7,
        "timeout": 120,
        "cache_seed": 42,  # For reproducibility; set to None to disable
    },
    human_input_mode="NEVER",         # No human intervention
    max_consecutive_auto_reply=10,     # Max auto-replies
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
)
```

**Critical Interview Detail -- Message Flow:**
When agent A sends a message to agent B:
1. A's `generate_reply` creates a message
2. The message is added to A's `_oai_messages[B]` (A's chat history with B)
3. B receives the message and it's added to B's `_oai_messages[A]`
4. B's `generate_reply` creates a response
5. The response flows back to A

---

## 2.3 AssistantAgent

`AssistantAgent` is a subclass of `ConversableAgent` pre-configured for LLM-based
assistance. Default settings:

```python
from autogen import AssistantAgent

assistant = AssistantAgent(
    name="Assistant",
    system_message="""You are a helpful AI assistant.
    Solve tasks using your coding and language skills.
    In the following cases, suggest python code (in a python coding block)
    or shell script (in a sh coding block) for the user to execute.
    ...
    Reply "TERMINATE" when the task is done.""",
    llm_config=llm_config,
    # Defaults:
    # human_input_mode="NEVER"
    # code_execution_config=False (does NOT execute code itself)
)
```

**Key Characteristics:**
- Has LLM capabilities (llm_config is required)
- Does NOT execute code by default
- Default system message encourages suggesting code for others to execute
- Designed to work with a UserProxyAgent that executes the code

---

## 2.4 UserProxyAgent

`UserProxyAgent` is a subclass of `ConversableAgent` that acts as a proxy for
a human user. It can execute code and solicit human input.

```python
from autogen import UserProxyAgent

user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="TERMINATE",  # Ask human only at termination
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    code_execution_config={
        "work_dir": "coding_workspace",
        "use_docker": True,  # IMPORTANT: Use Docker for safety
        # "use_docker": "python:3.11",  # Specific Docker image
    },
    llm_config=False,  # No LLM -- just executes code and relays human input
)
```

**human_input_mode Options:**
- `"ALWAYS"`: Always asks for human input before auto-replying
- `"TERMINATE"`: Asks for human input only when a termination condition is met
- `"NEVER"`: Never asks for human input; fully autonomous

**Code Execution Flow:**
1. AssistantAgent suggests code in a code block (```python ... ```)
2. UserProxyAgent detects the code block
3. UserProxyAgent executes it (in Docker or locally)
4. UserProxyAgent sends execution result back to AssistantAgent
5. AssistantAgent analyzes the result and iterates if needed

```python
# Complete two-agent code generation example
assistant = AssistantAgent(name="Assistant", llm_config=llm_config)

user_proxy = UserProxyAgent(
    name="UserProxy",
    code_execution_config={"work_dir": "workspace", "use_docker": True},
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
)

# Initiate the conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function that finds the nth Fibonacci number using memoization. Test it with n=50."
)
```

---

## 2.5 GroupChat and GroupChatManager

**GroupChat** enables multi-agent conversations with 3+ agents.

```python
from autogen import GroupChat, GroupChatManager

# Create agents
planner = AssistantAgent(name="Planner", system_message="...", llm_config=llm_config)
coder = AssistantAgent(name="Coder", system_message="...", llm_config=llm_config)
tester = AssistantAgent(name="Tester", system_message="...", llm_config=llm_config)
executor = UserProxyAgent(name="Executor", code_execution_config={...})

group_chat = GroupChat(
    agents=[planner, coder, tester, executor],
    messages=[],
    max_round=20,
    speaker_selection_method="auto",     # LLM picks next speaker
    # speaker_selection_method="round_robin",  # Fixed rotation
    # speaker_selection_method="random",       # Random selection
    # speaker_selection_method=custom_func,    # Your own function
    allow_repeat_speaker=False,  # Prevent same agent speaking twice in a row
    # allowed_or_disallowed_speaker_transitions={  # Control who can speak after whom
    #     planner: [coder, tester],
    #     coder: [tester, executor],
    #     tester: [coder, planner],
    #     executor: [tester, coder],
    # },
    # speaker_transitions_type="allowed",
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
)

# Start the conversation
planner.initiate_chat(manager, message="Build a web scraper for news articles.")
```

**Speaker Selection Methods (Critical for Interviews):**

1. **"auto"**: LLM analyzes conversation context and selects the most appropriate
   next speaker. Most flexible but adds LLM call overhead.

2. **"round_robin"**: Cycles through agents in order. Predictable, deterministic.

3. **"random"**: Randomly selects the next speaker. Useful for brainstorming.

4. **Custom function**: You provide a function that receives the last speaker,
   group chat, and returns the next speaker.

```python
def custom_speaker_selection(last_speaker, groupchat):
    """Custom logic for selecting the next speaker."""
    messages = groupchat.messages
    last_message = messages[-1]["content"] if messages else ""

    # Route based on content
    if "error" in last_message.lower() or "bug" in last_message.lower():
        return debugger_agent
    elif "```python" in last_message:
        return executor_agent
    elif "test" in last_message.lower():
        return tester_agent
    else:
        return planner_agent

group_chat = GroupChat(
    agents=[planner, coder, debugger, tester, executor],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection,
)
```

**Speaker Transitions (Graph-based Control):**

```python
# Define allowed transitions as a directed graph
allowed_transitions = {
    planner: [coder],           # Planner can only delegate to Coder
    coder: [executor, tester],  # Coder can delegate to Executor or Tester
    executor: [coder, tester],  # Executor sends results to Coder or Tester
    tester: [coder, planner],   # Tester can request fixes or report to Planner
}

group_chat = GroupChat(
    agents=[planner, coder, executor, tester],
    messages=[],
    max_round=20,
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed",  # "allowed" or "disallowed"
    speaker_selection_method="auto",
)
```

---

## 2.6 Nested Chats

Nested chats allow an agent to trigger a separate multi-agent conversation as
part of processing a message. This enables hierarchical architectures.

```python
# Nested Chat Example: When the writer receives a message,
# it triggers a sub-conversation with a researcher and fact-checker

writer = ConversableAgent(
    name="Writer",
    system_message="You write articles based on research provided to you.",
    llm_config=llm_config,
)

researcher = ConversableAgent(
    name="Researcher",
    system_message="You research topics and provide detailed findings.",
    llm_config=llm_config,
)

fact_checker = ConversableAgent(
    name="FactChecker",
    system_message="You verify factual claims and flag inaccuracies.",
    llm_config=llm_config,
)

# Register nested chats for the writer
writer.register_nested_chats(
    trigger=user_proxy,  # When Writer gets a message from UserProxy
    chat_queue=[
        {
            "recipient": researcher,
            "message": "Please research the following topic in depth.",
            "max_turns": 3,
            "summary_method": "reflection_with_llm",
        },
        {
            "recipient": fact_checker,
            "message": "Please verify the facts in the following research.",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
    ],
)

# When user_proxy sends a message to writer, it automatically
# triggers the nested research -> fact-check pipeline first
user_proxy.initiate_chat(writer, message="Write about quantum computing breakthroughs in 2025.")
```

**Summary Methods for Nested Chats:**
- `"last_msg"`: Uses the last message as the summary
- `"reflection_with_llm"`: Uses an LLM to summarize the conversation
- Custom function: You provide a function that takes the conversation and returns a summary

---

## 2.7 Teachability (Learning Agents)

Teachability enables agents to learn from conversations and remember across sessions
using a vector database (by default, ChromaDB).

```python
from autogen.agentchat.contrib.capabilities.teachability import Teachability

# Create a teachable agent
teachable_agent = ConversableAgent(
    name="TeachableAgent",
    system_message="You are a helpful assistant that learns from conversations.",
    llm_config=llm_config,
)

# Add teachability capability
teachability = Teachability(
    verbosity=0,
    reset_db=False,         # Set True to clear memory
    path_to_db_dir="./teach_db",
    recall_threshold=1.5,   # Lower = stricter matching
)
teachability.add_to_agent(teachable_agent)

# Session 1: Teach something
user_proxy.initiate_chat(
    teachable_agent,
    message="Remember this: Our company's fiscal year starts in April, not January."
)

# Session 2 (later): Agent remembers!
user_proxy.initiate_chat(
    teachable_agent,
    message="When does our fiscal year start?"
)
# Agent will recall: "Your company's fiscal year starts in April."
```

**How Teachability Works Internally:**
1. After each conversation turn, the agent analyzes if new information was taught
2. If yes, it extracts the key fact and stores it as a vector embedding in ChromaDB
3. Before generating each reply, it retrieves relevant memories via semantic search
4. Retrieved memories are injected into the context as additional information

---

## 2.8 Tool Use and Function Calling in AutoGen

AutoGen supports OpenAI-style function calling for tool use.

```python
# Method 1: Using @register_function decorator (AutoGen 0.2+)
from autogen import register_function
import yfinance as yf

def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")["Close"].iloc[-1]
    return f"The current price of {ticker} is ${price:.2f}"

def calculate_portfolio_value(holdings: dict) -> str:
    """Calculate total portfolio value given holdings {ticker: shares}."""
    total = 0
    for ticker, shares in holdings.items():
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]
        total += price * shares
    return f"Total portfolio value: ${total:,.2f}"

# Create agents
assistant = AssistantAgent(
    name="FinanceAssistant",
    system_message="You help with financial analysis. Use tools when needed.",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config=False,  # No code execution, only tool calls
)

# Register tools with both agents
# The assistant needs to know about tools (for LLM function calling)
# The user_proxy needs to execute them
register_function(
    get_stock_price,
    caller=assistant,     # The agent that calls the function (via LLM)
    executor=user_proxy,  # The agent that executes the function
    name="get_stock_price",
    description="Get the current stock price for a ticker symbol",
)

register_function(
    calculate_portfolio_value,
    caller=assistant,
    executor=user_proxy,
    name="calculate_portfolio_value",
    description="Calculate total portfolio value",
)

user_proxy.initiate_chat(assistant, message="What's the current price of AAPL and MSFT?")
```

```python
# Method 2: Using llm_config with functions (lower-level)
llm_config_with_tools = {
    "config_list": config_list,
    "functions": [
        {
            "name": "get_stock_price",
            "description": "Get current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    ]
}

assistant = AssistantAgent(
    name="Assistant",
    llm_config=llm_config_with_tools,
)

# Register the actual function implementation
user_proxy = UserProxyAgent(
    name="User",
    function_map={"get_stock_price": get_stock_price},
    human_input_mode="NEVER",
)
```

---

## 2.9 Code Execution in AutoGen

**Local Execution:**
```python
user_proxy = UserProxyAgent(
    name="Executor",
    code_execution_config={
        "work_dir": "coding_output",
        "use_docker": False,       # Executes locally (SECURITY RISK!)
        "timeout": 60,             # Timeout in seconds
        "last_n_messages": 3,      # Only look at last 3 messages for code
    },
)
```

**Docker Execution (Recommended for Production):**
```python
user_proxy = UserProxyAgent(
    name="Executor",
    code_execution_config={
        "work_dir": "coding_output",
        "use_docker": "python:3.11-slim",  # Specific Docker image
        "timeout": 120,
    },
)
```

**Custom Code Executor (AutoGen 0.2.27+):**
```python
from autogen.coding import LocalCommandLineCodeExecutor, DockerCommandLineCodeExecutor

# Local executor with more control
local_executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="./workspace",
)

# Docker executor
docker_executor = DockerCommandLineCodeExecutor(
    image="python:3.11-slim",
    timeout=120,
    work_dir="./workspace",
)

user_proxy = UserProxyAgent(
    name="Executor",
    code_execution_config={"executor": docker_executor},
)
```

---

## 2.10 Custom Agents

You can create custom agents by subclassing `ConversableAgent`:

```python
from autogen import ConversableAgent
from typing import Optional, Dict, List, Union

class DataAnalystAgent(ConversableAgent):
    """Custom agent specialized for data analysis tasks."""

    DEFAULT_SYSTEM_MESSAGE = """You are a senior data analyst. You:
    1. Always start by understanding the data schema
    2. Write pandas code for analysis
    3. Create visualizations with matplotlib/seaborn
    4. Provide statistical insights
    5. Suggest follow-up analyses"""

    def __init__(
        self,
        name="DataAnalyst",
        system_message=None,
        llm_config=None,
        data_sources=None,
        **kwargs
    ):
        super().__init__(
            name=name,
            system_message=system_message or self.DEFAULT_SYSTEM_MESSAGE,
            llm_config=llm_config,
            **kwargs
        )
        self.data_sources = data_sources or {}

    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional["Agent"] = None,
        **kwargs
    ) -> Union[str, Dict, None]:
        """Override to add custom pre-processing logic."""

        # Add data context to messages before generating reply
        if messages and self.data_sources:
            context = f"\nAvailable data sources: {list(self.data_sources.keys())}"
            enhanced_messages = messages.copy()
            enhanced_messages[-1] = {
                **enhanced_messages[-1],
                "content": enhanced_messages[-1]["content"] + context
            }
            return super().generate_reply(enhanced_messages, sender, **kwargs)

        return super().generate_reply(messages, sender, **kwargs)

    def load_data(self, source_name: str):
        """Custom method to load data from registered sources."""
        if source_name in self.data_sources:
            return self.data_sources[source_name]
        raise ValueError(f"Unknown data source: {source_name}")

# Usage
analyst = DataAnalystAgent(
    llm_config=llm_config,
    data_sources={
        "sales": "s3://bucket/sales.parquet",
        "customers": "s3://bucket/customers.parquet",
    }
)
```

---

## 2.11 Termination Conditions

Termination is critical -- without proper conditions, agents can chat forever.

```python
# Method 1: is_termination_msg (most common)
agent = ConversableAgent(
    name="Agent",
    is_termination_msg=lambda msg: any(
        keyword in msg.get("content", "").upper()
        for keyword in ["TERMINATE", "DONE", "COMPLETE", "EXIT"]
    ),
    llm_config=llm_config,
)

# Method 2: max_consecutive_auto_reply
agent = ConversableAgent(
    name="Agent",
    max_consecutive_auto_reply=5,  # Stops after 5 auto-replies
    llm_config=llm_config,
)

# Method 3: max_round in GroupChat
group_chat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=15,  # Stops after 15 total messages
)

# Method 4: Custom termination in generate_reply
class TerminatingAgent(ConversableAgent):
    def __init__(self, *args, budget_limit=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_cost = 0
        self.budget_limit = budget_limit

    def generate_reply(self, messages=None, sender=None, **kwargs):
        if self.total_cost >= self.budget_limit:
            return "Budget exceeded. TERMINATE"
        reply = super().generate_reply(messages, sender, **kwargs)
        # Track cost here
        return reply

# Method 5: Multiple conditions combined
def complex_termination(msg):
    content = msg.get("content", "") if msg else ""
    # Terminate if keyword found OR if message contains final answer format
    return (
        "TERMINATE" in content or
        content.startswith("FINAL ANSWER:") or
        len(content) == 0
    )
```

---

## 2.12 Human-in-the-Loop Patterns

```python
# Pattern 1: Always ask human (full human control)
human_agent = UserProxyAgent(
    name="Human",
    human_input_mode="ALWAYS",
    # Every response requires human approval/modification
)

# Pattern 2: Ask at termination only (semi-autonomous)
semi_auto = UserProxyAgent(
    name="SemiAuto",
    human_input_mode="TERMINATE",
    # Runs autonomously until an agent says TERMINATE,
    # then asks human to confirm/continue
)

# Pattern 3: Fully autonomous (no human)
auto = UserProxyAgent(
    name="Auto",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=20,
)

# Pattern 4: Approval gates using nested chats
approval_agent = ConversableAgent(
    name="ApprovalGate",
    system_message="""Review the proposed action.
    If it involves: deleting data, sending emails, making payments,
    or accessing production systems -- ask for human approval.
    Otherwise, auto-approve.""",
    human_input_mode="ALWAYS",  # Forces human review
    llm_config=llm_config,
)

# Pattern 5: Conditional human-in-the-loop
class ConditionalHumanAgent(UserProxyAgent):
    """Only asks for human input on high-stakes decisions."""

    HIGH_RISK_KEYWORDS = ["delete", "payment", "production", "deploy", "email"]

    def get_human_input(self, prompt):
        last_message = self._oai_messages[list(self._oai_messages.keys())[-1]][-1]
        content = last_message.get("content", "").lower()

        if any(keyword in content for keyword in self.HIGH_RISK_KEYWORDS):
            return super().get_human_input(
                f"HIGH RISK ACTION DETECTED!\n{prompt}"
            )
        return ""  # Auto-approve low-risk actions
```

---

## 2.13 AutoGen 0.4 / AG2 New Architecture (Important for 2025-2026)

AutoGen 0.4 introduced a completely new architecture:

**Key Changes:**
- **Asynchronous by default**: All agents run asynchronously
- **Event-driven messaging**: Agents communicate via events, not synchronous calls
- **Runtime**: A central runtime manages agent lifecycle and message routing
- **Typed messages**: Messages have explicit types, not just strings
- **Better isolation**: Agents are more isolated, reducing coupling

```python
# AutoGen 0.4 / AG2 Style (new API)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# New model client interface
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Agents in 0.4
agent1 = AssistantAgent(
    name="PrimaryAssistant",
    model_client=model_client,
    system_message="You are a helpful assistant.",
)

agent2 = AssistantAgent(
    name="CriticAssistant",
    model_client=model_client,
    system_message="You review and critique responses.",
)

# Termination condition
termination = TextMentionTermination("APPROVE")

# Team (replaces GroupChat)
team = RoundRobinGroupChat(
    participants=[agent1, agent2],
    termination_condition=termination,
)

# Async execution
import asyncio

async def main():
    result = await team.run(task="Write a haiku about programming.")
    print(result)

asyncio.run(main())
```

**New Team Types in AG2 0.4:**
- `RoundRobinGroupChat`: Agents take turns in order
- `SelectorGroupChat`: LLM or custom logic selects next speaker
- `Swarm`: OpenAI-swarm inspired handoff-based teams
- `MagenticOneGroupChat`: Microsoft's Magentic-One architecture

---

# ============================================================
# SECTION 3: AGENT COMMUNICATION PATTERNS
# ============================================================

## 3.1 Message Passing

The most fundamental communication pattern. Agents send discrete messages to
each other. This is the primary pattern used by AutoGen.

**Direct Message Passing:**
```python
# Agent A sends directly to Agent B
agent_a.initiate_chat(agent_b, message="Hello, please analyze this data.")

# The conversation continues as a back-and-forth
# A -> B -> A -> B -> ... until termination
```

**Broadcast Message Passing:**
```python
# In GroupChat, messages are broadcast to all agents
group_chat = GroupChat(agents=[a, b, c, d], messages=[])
# When agent A speaks, all agents B, C, D see the message
# This is broadcast messaging with selective response
```

**Key Characteristics:**
- Asynchronous or synchronous
- Point-to-point or broadcast
- Ordered (FIFO) or unordered
- Can be reliable (guaranteed delivery) or best-effort

---

## 3.2 Shared State Pattern

Agents communicate by reading and writing to a shared state object.
LangGraph uses this pattern heavily.

```python
# LangGraph Shared State Pattern
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class SharedState(TypedDict):
    messages: Annotated[list, add]         # Append-only message list
    research_notes: str                     # Shared research notes
    code_output: str                        # Shared code output
    review_status: str                      # Current review status
    iteration_count: int                    # Track iterations

def researcher_node(state: SharedState) -> dict:
    """Researcher reads shared state, does research, updates state."""
    messages = state["messages"]
    # ... do research based on messages ...
    return {
        "messages": [{"role": "assistant", "content": "Research findings..."}],
        "research_notes": "Key finding: ...",
    }

def coder_node(state: SharedState) -> dict:
    """Coder reads research notes from shared state, writes code."""
    research = state["research_notes"]
    # ... generate code based on research ...
    return {
        "messages": [{"role": "assistant", "content": "```python\n...\n```"}],
        "code_output": "def solution(): ...",
    }

def reviewer_node(state: SharedState) -> dict:
    """Reviewer reads code from shared state, provides review."""
    code = state["code_output"]
    # ... review code ...
    return {
        "messages": [{"role": "assistant", "content": "Review: LGTM"}],
        "review_status": "approved",
        "iteration_count": state["iteration_count"] + 1,
    }

# Build the graph
workflow = StateGraph(SharedState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "coder")
workflow.add_edge("coder", "reviewer")

def should_continue(state: SharedState):
    if state["review_status"] == "approved" or state["iteration_count"] >= 3:
        return END
    return "researcher"  # Loop back for revision

workflow.add_conditional_edges("reviewer", should_continue)
graph = workflow.compile()
```

**Advantages:**
- All agents have access to the full context
- Easy to debug (inspect state at any point)
- Natural for workflows with shared data

**Disadvantages:**
- State can grow large
- Concurrent writes need conflict resolution
- Tight coupling between agents through shared schema

---

## 3.3 Blackboard Pattern

A shared knowledge base (blackboard) that all agents can read from and write to.
Similar to shared state but more structured -- the blackboard organizes knowledge
by topic/category.

```python
# Blackboard Pattern Implementation
class Blackboard:
    """Shared knowledge base for multi-agent collaboration."""

    def __init__(self):
        self._knowledge = {}
        self._subscribers = {}
        self._lock = threading.Lock()

    def write(self, category: str, key: str, value: any, author: str):
        """Write knowledge to the blackboard."""
        with self._lock:
            if category not in self._knowledge:
                self._knowledge[category] = {}
            self._knowledge[category][key] = {
                "value": value,
                "author": author,
                "timestamp": time.time(),
            }
        # Notify subscribers
        self._notify(category, key)

    def read(self, category: str, key: str = None):
        """Read from the blackboard."""
        if key:
            return self._knowledge.get(category, {}).get(key, {}).get("value")
        return {k: v["value"] for k, v in self._knowledge.get(category, {}).items()}

    def subscribe(self, category: str, callback):
        """Subscribe to changes in a category."""
        if category not in self._subscribers:
            self._subscribers[category] = []
        self._subscribers[category].append(callback)

    def _notify(self, category, key):
        for callback in self._subscribers.get(category, []):
            callback(category, key)

# Usage
blackboard = Blackboard()

# Research agent writes findings
blackboard.write("research", "market_size", "$5.2B", author="ResearchAgent")
blackboard.write("research", "growth_rate", "15% CAGR", author="ResearchAgent")

# Analysis agent reads research and writes analysis
market_size = blackboard.read("research", "market_size")
blackboard.write("analysis", "opportunity_score", 0.85, author="AnalysisAgent")

# Decision agent reads analysis
score = blackboard.read("analysis", "opportunity_score")
```

---

## 3.4 Event-Driven Communication

Agents communicate by publishing and subscribing to events.
AutoGen 0.4 uses this pattern natively.

```python
# Event-Driven Pattern
from dataclasses import dataclass
from typing import List, Callable
import asyncio

@dataclass
class Event:
    type: str
    source: str
    data: dict

class EventBus:
    """Central event bus for agent communication."""

    def __init__(self):
        self._handlers: dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def publish(self, event: Event):
        handlers = self._handlers.get(event.type, [])
        await asyncio.gather(*[h(event) for h in handlers])

class EventDrivenAgent:
    def __init__(self, name: str, bus: EventBus):
        self.name = name
        self.bus = bus

    async def publish(self, event_type: str, data: dict):
        await self.bus.publish(Event(type=event_type, source=self.name, data=data))

# Usage
bus = EventBus()

research_agent = EventDrivenAgent("Researcher", bus)
writer_agent = EventDrivenAgent("Writer", bus)

# Writer subscribes to research completion events
async def on_research_complete(event: Event):
    findings = event.data["findings"]
    # Writer starts writing based on research
    article = f"Article based on: {findings}"
    await writer_agent.publish("article_draft_complete", {"article": article})

bus.subscribe("research_complete", on_research_complete)

# Research agent publishes when done
await research_agent.publish("research_complete", {"findings": "Key findings..."})
```

---

## 3.5 Pub-Sub (Publish-Subscribe) Pattern

A more formal version of event-driven communication with topics/channels.

```python
# Pub-Sub Pattern with Topics
class PubSubBroker:
    """Message broker with topic-based pub-sub."""

    def __init__(self):
        self.topics: dict[str, list] = {}
        self.message_queue: dict[str, list] = {}

    def create_topic(self, topic: str):
        self.topics[topic] = []
        self.message_queue[topic] = []

    def subscribe(self, topic: str, agent_id: str):
        if topic in self.topics:
            self.topics[topic].append(agent_id)

    def publish(self, topic: str, message: dict, publisher: str):
        if topic in self.topics:
            self.message_queue[topic].append({
                "publisher": publisher,
                "message": message,
                "timestamp": time.time(),
            })
            # Deliver to all subscribers
            for subscriber_id in self.topics[topic]:
                if subscriber_id != publisher:  # Don't send to self
                    self._deliver(subscriber_id, topic, message)

    def _deliver(self, agent_id: str, topic: str, message: dict):
        # Deliver message to agent's inbox
        pass

# Usage for multi-agent system
broker = PubSubBroker()
broker.create_topic("data_updates")
broker.create_topic("code_reviews")
broker.create_topic("deployment_events")

# Different agents subscribe to relevant topics
broker.subscribe("data_updates", "data_analyst")
broker.subscribe("data_updates", "ml_engineer")
broker.subscribe("code_reviews", "senior_dev")
broker.subscribe("code_reviews", "security_analyst")
broker.subscribe("deployment_events", "devops_agent")
broker.subscribe("deployment_events", "monitoring_agent")
```

---

## 3.6 Communication Pattern Comparison

```
| Pattern        | Coupling | Scalability | Complexity | Real-Time | Used By        |
|---------------|----------|-------------|------------|-----------|----------------|
| Message Pass  | Low      | High        | Low        | Yes       | AutoGen, CrewAI|
| Shared State  | Medium   | Medium      | Medium     | No        | LangGraph      |
| Blackboard    | Medium   | Medium      | Medium     | No        | Custom systems |
| Event-Driven  | Low      | High        | High       | Yes       | AutoGen 0.4    |
| Pub-Sub       | Very Low | Very High   | High       | Yes       | Custom systems |
```

---

## 3.7 Interview Questions on Communication Patterns

**Q: How does AutoGen handle message passing between agents?**

A: "In AutoGen, agents communicate via the `initiate_chat` and `generate_reply`
methods. When agent A initiates a chat with agent B, A sends a message that gets
added to both agents' conversation histories. B then generates a reply, which
flows back to A. Each agent maintains separate conversation histories with each
agent it talks to (`_oai_messages` dict). In GroupChat, messages are broadcast --
all agents see all messages, but only the selected speaker responds. In AutoGen 0.4,
this shifted to an event-driven model where agents publish and subscribe to typed
message events through a central runtime."

**Q: Compare shared state (LangGraph) vs message passing (AutoGen) approaches.**

A: "Shared state (LangGraph) uses a typed state object that all nodes read and
write to. It's like a shared whiteboard -- every agent sees the full picture.
This makes it easy to reason about the system but creates coupling through the
shared schema. Message passing (AutoGen) gives agents private conversations --
agent A's chat with B is separate from A's chat with C. This provides better
isolation but can make global context harder to access. For deterministic workflows
with shared data, I'd choose shared state. For dynamic conversations where agents
need autonomy, I'd choose message passing."
