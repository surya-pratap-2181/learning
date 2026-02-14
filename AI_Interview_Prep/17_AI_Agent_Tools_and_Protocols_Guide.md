# AI Agent Tools & Protocols: MCP, A2A, Composio, Function Calling (2025-2026)
## The Definitive Interview Guide for AI Engineers

---

# TABLE OF CONTENTS
1. [AI Agent Tool Use Overview](#s1)
2. [Explaining to a Layman](#s2)
3. [Function Calling Deep Dive](#s3)
4. [Model Context Protocol (MCP)](#s4)
5. [Google A2A Protocol](#s5)
6. [MCP vs A2A Comparison](#s6)
7. [Composio Framework](#s7)
8. [Tool Design Patterns](#s8)
9. [OAuth 2.0 for AI Agents](#s9)
10. [Building Custom Tools](#s10)
11. [Interview Questions (20+)](#s11)
12. [Code Examples](#s12)

---

# SECTION 1: AI AGENT TOOL USE OVERVIEW

## Evolution of Agent Tool Use

```
2022              2023              2024              2025-2026
 │                 │                 │                 │
 ▼                 ▼                 ▼                 ▼
Function       Tool Use         MCP Launched      MCP + A2A
Calling        Standardization  (Anthropic)       (Industry Standard)
(OpenAI)       (Multiple LLMs)  Composio grows    Agent Protocols
```

**Why agents need tools:**
- LLMs can only generate text - they can't take actions
- Tools give agents the ability to: search the web, query databases, send emails, call APIs, execute code
- Tool use transforms LLMs from "text generators" to "action takers"

> 🔵 **YOUR EXPERIENCE**: At RavianAI, you implemented a dynamic capability registry and tool discovery mechanism, allowing agents to reason about available tools at runtime and select optimal execution paths. This is cutting-edge tool orchestration.

---

# SECTION 2: EXPLAINING TO A LAYMAN

> Think of AI tools like apps on a smartphone. The AI (phone) is smart, but without apps, it can only do basic things. With Gmail app, it can send emails. With Maps, it can navigate. With a calculator, it can do math.
>
> MCP is like the App Store - a standard way to install and discover new apps.
> A2A is like how your apps talk to each other - when Maps opens Uber.
> Composio is like having 800+ pre-installed apps ready to go.

---

# SECTION 3: FUNCTION CALLING DEEP DIVE

## 3.1 OpenAI Function Calling

```python
import openai

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}]

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=tools,
    tool_choice="auto"  # or "required" or {"type": "function", "function": {"name": "get_weather"}}
)

# Model returns tool_calls, you execute, then send result back
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    # Execute the function
    result = get_weather(**json.loads(tool_call.function.arguments))
    # Send result back
    messages.append({"role": "tool", "content": str(result), "tool_call_id": tool_call.id})
```

## 3.2 Anthropic (Claude) Tool Use

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }],
    messages=[{"role": "user", "content": "Weather in NYC?"}]
)
```

## 3.3 Comparison Table

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| **Syntax** | `tools` array | `tools` array | `function_declarations` |
| **Parallel calls** | Yes (multiple tool_calls) | Yes | Yes |
| **Forced call** | `tool_choice: required` | `tool_choice: any` | `tool_config` |
| **Streaming** | Yes | Yes | Yes |
| **Structured output** | JSON Schema | JSON Schema | JSON Schema |

---

# SECTION 4: MODEL CONTEXT PROTOCOL (MCP)

## 4.1 What is MCP?

MCP (Model Context Protocol) is an **open protocol** announced by Anthropic in November 2024 that creates a standard way for AI applications to connect with external data sources and tools. Think of it as "USB-C for AI" - one standard connector for everything.

**The N×M Problem MCP Solves:**
```
Before MCP:                          After MCP:
App1 ──── Tool1                      App1 ──┐
App1 ──── Tool2                      App2 ──┤── MCP ──┬── Tool1
App2 ──── Tool1  (N×M connectors)    App3 ──┘         ├── Tool2
App2 ──── Tool2                                        └── Tool3
App3 ──── Tool1                      (N+M connectors)
App3 ──── Tool2
```

## 4.2 MCP Architecture

```
┌──────────────────────────────────────────────────────┐
│                    MCP HOST                            │
│  (Claude Desktop, VS Code, IDE, Custom App)           │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │  MCP Client  │  │  MCP Client  │  ← 1 per server │
│  └──────┬───────┘  └──────┬───────┘                  │
└─────────┼──────────────────┼─────────────────────────┘
          │ JSON-RPC 2.0     │ JSON-RPC 2.0
          │ (stdio/SSE)      │ (stdio/SSE)
          ▼                  ▼
┌──────────────┐    ┌──────────────┐
│  MCP Server  │    │  MCP Server  │
│  (GitHub)    │    │  (Database)  │
│              │    │              │
│  Resources   │    │  Tools       │
│  Tools       │    │  Prompts     │
│  Prompts     │    │              │
└──────────────┘    └──────────────┘
```

## 4.3 MCP Capabilities

| Capability | Description | Example |
|-----------|-------------|---------|
| **Resources** | Read-only data sources | File contents, DB records, API responses |
| **Tools** | Actions the agent can execute | Send email, create PR, query DB |
| **Prompts** | Pre-built prompt templates | Summarize document, analyze code |

## 4.4 Building an MCP Server

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-tools")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_database",
            description="Search the company database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_database":
        results = await db.search(arguments["query"])
        return [TextContent(type="text", text=str(results))]

# Run with stdio transport
from mcp.server.stdio import stdio_server
async with stdio_server() as (read, write):
    await server.run(read, write)
```

## 4.5 MCP Adoption (2025-2026)

- **Nov 2024**: Anthropic launches MCP
- **Mar 2025**: OpenAI adopts MCP across Agents SDK, Responses API, ChatGPT desktop
- **Apr 2025**: Google DeepMind confirms MCP support for Gemini
- **Nov 2025**: Major spec update: async ops, statelessness, server identity, registry
- **Dec 2025**: Anthropic donates MCP to Linux Foundation (AAIF)

> 🔵 **YOUR EXPERIENCE**: At RavianAI, your dynamic capability registry maps directly to MCP server patterns. You can discuss how MCP standardizes what you built custom.

---

# SECTION 5: GOOGLE A2A PROTOCOL

## 5.1 What is A2A?

Agent-to-Agent (A2A) is Google's protocol for **agent-to-agent communication** (announced April 2025). While MCP connects agents to tools/data (vertical), A2A connects agents to other agents (horizontal).

## 5.2 Key Concepts

| Concept | Description |
|---------|-------------|
| **Agent Card** | JSON metadata describing an agent's capabilities |
| **Task** | Unit of work with states: submitted → working → input-required → completed/failed |
| **Message** | Communication between agents |
| **Artifact** | Output produced by a task (file, data, etc.) |

## 5.3 A2A vs MCP

```
MCP:  Agent ←──→ Tools/Data (vertical - agent connects to resources)
A2A:  Agent ←──→ Agent      (horizontal - agents collaborate)
```

| Aspect | MCP | A2A |
|--------|-----|-----|
| **Purpose** | Agent ↔ Tools/Data | Agent ↔ Agent |
| **Direction** | Vertical (down to resources) | Horizontal (peer-to-peer) |
| **Protocol** | JSON-RPC 2.0 (stdio/SSE) | JSON-RPC 2.0 over HTTP(S) |
| **Discovery** | Server manifest | Agent Cards |
| **State** | Stateful sessions | Task-based state machine |
| **Use Case** | Access databases, APIs, files | Multi-agent collaboration |
| **Launched** | Nov 2024 (Anthropic) | Apr 2025 (Google) |
| **Backers** | OpenAI, Google, Microsoft | Atlassian, Salesforce, PayPal |

**Key Insight: They are complementary, not competing.**
- MCP: "How do agents use tools?"
- A2A: "How do agents talk to each other?"
- Use both: Agent uses MCP to access tools, A2A to collaborate with other agents.

> 🔵 **YOUR EXPERIENCE**: At RavianAI, your cross-application orchestration (Gmail, GitHub, Slack, etc.) involves both tool access (MCP-like) and agent coordination (A2A-like).

---

# SECTION 6: MCP vs A2A COMPARISON

Best practice in 2025-2026: **Use both together**

```
┌─────────────────────────────────────────────────┐
│             MULTI-AGENT SYSTEM                    │
│                                                   │
│  Agent A ←──A2A──→ Agent B ←──A2A──→ Agent C    │
│     │                  │                  │       │
│    MCP                MCP                MCP      │
│     │                  │                  │       │
│  ┌──▼──┐          ┌──▼──┐          ┌──▼──┐     │
│  │Gmail│          │GitHub│          │Slack │     │
│  │Tool │          │Tool  │          │Tool  │     │
│  └─────┘          └──────┘          └──────┘     │
└─────────────────────────────────────────────────┘
```

---

# SECTION 7: COMPOSIO FRAMEWORK

## 7.1 What is Composio?

Composio provides **800+ pre-built tool integrations** for AI agents. Instead of building custom connectors, you plug in Composio and get instant access to Gmail, GitHub, Slack, Jira, etc.

## 7.2 Key Features

| Feature | Description |
|---------|-------------|
| **800+ Tools** | Pre-built connectors for popular APIs |
| **Auth Management** | OAuth, API keys, JWT handling per user |
| **Framework Agnostic** | Works with LangChain, CrewAI, AutoGen, OpenAI |
| **MCP Support** | Can serve tools via MCP protocol |
| **Sandboxed Execution** | Safe tool execution environment |
| **Observability** | Request/response logging, trace IDs |

## 7.3 Using Composio with Autogen

```python
from composio_autogen import ComposioToolSet, Action
from autogen import ConversableAgent

toolset = ComposioToolSet()

# Get Gmail tools
gmail_tools = toolset.get_tools(actions=[
    Action.GMAIL_SEND_EMAIL,
    Action.GMAIL_LIST_EMAILS,
    Action.GMAIL_READ_EMAIL
])

assistant = ConversableAgent(
    name="EmailAssistant",
    system_message="You manage emails using Gmail tools.",
    llm_config=llm_config
)

# Register tools with the agent
toolset.register_tools(gmail_tools, caller=assistant)
```

> 🔵 **YOUR EXPERIENCE**: At RavianAI, you use Composio for tool integration in your agentic platform. You can discuss production experience with auth management across multiple third-party apps (Gmail, Microsoft 365, GitHub, Slack, Notion).

---

# SECTION 8: TOOL DESIGN PATTERNS

## 8.1 Dynamic Tool Discovery

```python
class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, func, description, permissions):
        self._tools[name] = {
            "function": func,
            "description": description,
            "permissions": permissions
        }

    def get_available_tools(self, user_context):
        """Return tools available based on context and permissions."""
        return [
            tool for name, tool in self._tools.items()
            if self._check_permissions(tool["permissions"], user_context)
        ]
```

> 🔵 **YOUR EXPERIENCE**: This is exactly what you built at RavianAI - a dynamic capability registry allowing agents to reason about available tools at runtime.

## 8.2 Tool Authorization Patterns

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **API Key per tool** | Simple integrations | Store in env vars |
| **OAuth 2.0 per user** | User-specific access | Token store with refresh |
| **Scoped permissions** | Least privilege | Permission checks before execution |
| **Sandboxed execution** | Untrusted tools | Docker/VM isolation |

---

# SECTION 9: OAUTH 2.0 FOR AI AGENTS

## 9.1 OAuth Flow for Agent Tool Access

```
User → Agent → "I need to access your Gmail"
         │
         ▼
┌─────────────────────────────┐
│  1. Redirect user to Google │
│  2. User grants permission  │
│  3. Get authorization code  │
│  4. Exchange for tokens     │
│  5. Store tokens securely   │
│  6. Agent uses access token │
│  7. Refresh when expired    │
└─────────────────────────────┘
```

```python
from authlib.integrations.httpx_client import AsyncOAuth2Client

class AgentOAuthManager:
    def __init__(self):
        self.token_store = {}  # In production: encrypted DB

    async def get_token(self, user_id: str, provider: str):
        token = self.token_store.get(f"{user_id}:{provider}")
        if token and token["expires_at"] < time.time():
            token = await self.refresh_token(token)
        return token

    async def refresh_token(self, token):
        client = AsyncOAuth2Client(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        new_token = await client.refresh_token(TOKEN_URL, refresh_token=token["refresh_token"])
        return new_token
```

> 🔵 **YOUR EXPERIENCE**: At RavianAI, you integrated OAuth 2.0 and JWT for secure, privacy-preserving authentication across multiple third-party apps.

---

# SECTION 10: BUILDING CUSTOM TOOLS

## For LangChain
```python
from langchain_core.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the company database for relevant records."""
    results = db.search(query, limit=limit)
    return json.dumps(results)
```

## For AutoGen/AG2
```python
from autogen import register_function

def search_database(query: str, limit: int = 10) -> str:
    results = db.search(query, limit=limit)
    return json.dumps(results)

register_function(
    search_database,
    caller=assistant,
    executor=user_proxy,
    description="Search the company database"
)
```

---

# SECTION 11: INTERVIEW QUESTIONS (20+)

**Q1: What is MCP and why was it created?**
MCP (Model Context Protocol) by Anthropic is an open standard for connecting AI apps to external data/tools. Solves the N×M integration problem - instead of custom connectors per app-tool pair, one standard protocol. Uses JSON-RPC 2.0 with stdio/SSE transport.

**Q2: Explain MCP architecture (hosts, clients, servers).**
Host = the AI app (Claude Desktop, IDE). Client = runs inside host, connects to one server. Server = exposes tools/resources/prompts. Client-server use JSON-RPC 2.0. One client per server, can have multiple clients.

**Q3: What is A2A and how does it differ from MCP?**
A2A (Agent-to-Agent) by Google is for agent-to-agent communication. MCP = vertical (agent to tools). A2A = horizontal (agent to agent). They're complementary. A2A uses Agent Cards for discovery, Tasks for state management.

**Q4: How does function calling work in LLMs?**
LLM receives tool definitions as JSON schema. When it determines a tool is needed, it returns a structured tool_call with function name and arguments. The application executes the function and sends the result back. LLM then incorporates the result into its response.

**Q5: What is Composio and when would you use it?**
Composio provides 800+ pre-built tool integrations for AI agents with auth management. Use when you need many third-party integrations (Gmail, GitHub, Slack) without building custom connectors. Framework-agnostic.

**Q6: How do you handle authentication for agent tools?**
Depends on the tool: API keys for simple integrations, OAuth 2.0 for user-specific access (with token refresh), JWT for internal services. Store tokens encrypted. Use Composio for managed auth.

**Q7: How do you handle tool errors in agent workflows?**
Retry with exponential backoff, fallback to alternative tools, graceful degradation, error context in agent state, max retry limits, human escalation for critical failures.

**Q8: What is a dynamic tool registry?**
A runtime system where agents discover available tools based on context, permissions, and task complexity. Instead of static tool lists, agents query the registry to find the right tool for the current situation.

**Q9: How do you secure tool execution?**
Sandboxed execution (Docker), permission scoping, rate limiting, input validation, output filtering for sensitive data, audit logging, least-privilege access.

**Q10: What is the difference between MCP resources, tools, and prompts?**
Resources = read-only data (files, DB records). Tools = executable actions (send email, create PR). Prompts = reusable prompt templates. Resources are for context, tools are for action, prompts are for interaction patterns.

**Q11: How would you design a tool orchestration system?**
Registry for tool discovery, permission layer for auth, execution engine with sandboxing, result caching, retry/fallback logic, observability (logging, tracing), rate limiting per user/tool.

**Q12: Parallel vs sequential tool calls - when to use each?**
Parallel when tools are independent (check weather + check calendar). Sequential when output of one is input to another (search → summarize results). LLMs can request parallel calls natively.

**Q13: How do you handle tool versioning?**
Semantic versioning for tool APIs, backward compatibility, gradual migration, A/B testing new versions, deprecation warnings, tool registry tracks versions.

**Q14: What happens when a tool takes too long?**
Timeout mechanisms, async execution with callbacks, streaming partial results, agent informs user about delay, fallback to cached/approximate results.

**Q15: How do you test agent tool integrations?**
Mock tool responses in unit tests, integration tests with sandbox APIs, load testing for concurrent tool calls, chaos testing (random failures), end-to-end with real APIs in staging.

**Q16: How does MCP handle security?**
Transport security (TLS), auth per server, capability negotiation, content filtering, sandboxed execution. Nov 2025 update added server identity verification.

**Q17: What is an Agent Card in A2A?**
JSON metadata describing an agent: name, description, capabilities, supported protocols, authentication requirements. Used for agent discovery - like a business card for AI agents.

**Q18: How does Composio handle multi-tenant auth?**
Per-user OAuth tokens, isolated credential storage, per-user permission scoping. When agent acts on behalf of User A, it uses User A's tokens only.

**Q19: How would you migrate from custom tool integrations to MCP?**
Wrap existing tools as MCP servers, maintain backward compatibility during transition, test MCP versions alongside existing, gradual rollout per tool.

**Q20: What is the future of AI agent protocols?**
MCP + A2A becoming industry standards. Convergence expected: MCP for tool access, A2A for agent coordination. Standardized registries for tool/agent discovery. Security frameworks maturing.

---

# SECTION 12: CODE EXAMPLES

## MCP Server for Custom Database

```python
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource

server = Server("company-db")

@server.list_resources()
async def list_resources():
    return [
        Resource(uri="db://customers", name="Customer Database", description="All customer records")
    ]

@server.list_tools()
async def list_tools():
    return [
        Tool(name="query_customers", description="Query customer database",
             inputSchema={"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}),
        Tool(name="update_customer", description="Update customer record",
             inputSchema={"type": "object", "properties": {"id": {"type": "string"}, "data": {"type": "object"}}, "required": ["id", "data"]})
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "query_customers":
        results = await db.execute(arguments["sql"])
        return [TextContent(type="text", text=json.dumps(results))]
    elif name == "update_customer":
        await db.update("customers", arguments["id"], arguments["data"])
        return [TextContent(type="text", text="Updated successfully")]
```

---

## Sources
- [Anthropic MCP Announcement](https://www.anthropic.com/news/model-context-protocol)
- [MCP Specification](https://modelcontextprotocol.io/specification/2025-11-25)
- [Google A2A Protocol](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [MCP vs A2A: Koyeb](https://www.koyeb.com/blog/a2a-and-mcp-start-of-the-ai-agent-protocol-wars)
- [Composio Docs](https://docs.composio.dev/docs)
- [Composio GitHub](https://github.com/ComposioHQ/composio)
