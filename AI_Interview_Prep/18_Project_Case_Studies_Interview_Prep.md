---
title: "Project Case Studies"
layout: default
parent: "Interview Practice & Career"
nav_order: 5
---

# Project Case Studies: "Tell Me About Your Project" Interview Prep
## Detailed Architecture Breakdowns for AI Engineer Interviews (2025-2026)

---

# TABLE OF CONTENTS
1. How to Answer "Tell Me About Your Project"
2. RavianAI - Agentic AI Platform (Platform Architect)
3. MARS - Multi-Agent Analytics Platform (MathCo)
4. Stellantis - Global Data Insights (MathCo)
5. AbbVie - Data Pipeline & Orchestration (MathCo)
6. Takeda - Cloud Migration & Re-Architecture (MathCo)
7. Fealty Technologies - Production Web Applications
8. AG2AI Autogen - Open Source Contribution
9. GDG Talk - "Rethinking AI Agents"
10. Common Follow-Up Questions & How to Answer

---

# SECTION 1: HOW TO ANSWER "TELL ME ABOUT YOUR PROJECT"

## The STAR+ Framework for Technical Projects

```
S - Situation:  What was the business problem? (1-2 sentences)
T - Task:       What was your specific role and responsibility?
A - Action:     What did you build? (architecture, tech decisions, challenges)
R - Result:     What was the measurable impact?
+ - Lessons:    What did you learn? What would you do differently?
```

**Tips:**
- Lead with IMPACT, not technology ("Built a platform that reduced manual analysis by 60%")
- Be specific about YOUR contribution vs team contribution
- Have 3 levels of depth: 30-second, 2-minute, 5-minute versions
- Prepare for "why did you choose X over Y?" follow-ups
- Always connect technology choices to business requirements

---

# SECTION 2: RAVIANAI - AGENTIC AI PLATFORM

## 2.1 The 30-Second Version

"At RavianAI, I'm the Platform Architect for a device-native Agentic AI platform that runs on macOS and Windows. It lets users automate workflows across Gmail, Slack, GitHub, Notion, and Microsoft 365 using natural language. I designed the entire agentic architecture -- from intent recognition to task execution -- using AG2 Autogen for multi-agent orchestration, FastAPI with WebSocket for real-time communication, and packaged it with Nuitka + Electron for desktop distribution."

## 2.2 Full Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────┐
│                    ELECTRON SHELL                        │
│  ┌──────────────┐  ┌──────────────────────────────────┐ │
│  │  React UI    │  │  Nuitka-Compiled Python Core     │ │
│  │  (Frontend)  │◄─┤                                  │ │
│  │              │  │  ┌─────────────────────────────┐ │ │
│  │  - Chat UI   │  │  │    FastAPI + WebSocket       │ │ │
│  │  - Workflow   │  │  │    (Real-time Agent Comms)   │ │ │
│  │    Builder    │  │  │                             │ │ │
│  │  - Settings   │  │  │  ┌──────────────────────┐  │ │ │
│  └──────────────┘  │  │  │  AG2 Autogen Engine   │  │ │ │
│                    │  │  │  ┌─────┐ ┌─────┐      │  │ │ │
│                    │  │  │  │Email│ │Code │      │  │ │ │
│                    │  │  │  │Agent│ │Agent│ ...   │  │ │ │
│                    │  │  │  └──┬──┘ └──┬──┘      │  │ │ │
│                    │  │  └────┼────────┼─────────┘  │ │ │
│                    │  │       │        │            │ │ │
│                    │  │  ┌────┴────────┴─────────┐  │ │ │
│                    │  │  │ Dynamic Tool Registry  │  │ │ │
│                    │  │  │ (Composio 800+ Tools)  │  │ │ │
│                    │  │  └───────────┬───────────┘  │ │ │
│                    │  └─────────────┼───────────────┘ │ │
│                    └────────────────┼─────────────────┘ │
└─────────────────────────────────────┼───────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │         AWS INFRASTRUCTURE         │
                    │  ┌──────┐ ┌──────┐ ┌──────┐      │
                    │  │ EC2  │ │  S3  │ │ RDS  │      │
                    │  └──────┘ └──────┘ └──────┘      │
                    │  ┌──────────────────────────┐     │
                    │  │ Docker + Nginx (LB)      │     │
                    │  └──────────────────────────┘     │
                    └───────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │      THIRD-PARTY INTEGRATIONS      │
                    │  Gmail │ Slack │ GitHub │ Notion   │
                    │  MS 365│ Calendar │ ...            │
                    │  (via Composio + OAuth 2.0)        │
                    └───────────────────────────────────┘
```

## 2.3 Key Technical Decisions & Why

### Decision 1: AG2 Autogen over LangGraph for Multi-Agent
**Why:** AG2's conversation-based paradigm was ideal for our use case -- agents need to dynamically collaborate based on user intent, not follow a fixed graph. The GroupChat pattern with custom speaker selection let us route to specialized agents (email, code, scheduling) based on context.

**Follow-up ready:** "LangGraph would have been better if we had a fixed, deterministic workflow. But our platform needed dynamic agent selection based on ambiguous natural language input, which maps naturally to AG2's conversation model."

### Decision 2: FastAPI + WebSocket over REST polling
**Why:** Real-time agent communication requires streaming token-by-token responses and status updates. WebSocket provides bidirectional, persistent connections. REST polling would add latency and waste bandwidth.

```python
# Connection Manager for multi-session WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_sessions: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_sessions[session_id] = websocket

    async def stream_agent_response(self, session_id: str, agent_stream):
        ws = self.active_sessions.get(session_id)
        if ws:
            async for token in agent_stream:
                await ws.send_json({"type": "token", "content": token})
            await ws.send_json({"type": "done"})
```

### Decision 3: Nuitka + Electron for Desktop
**Why:** We needed to ship a cross-platform desktop app with a Python AI core. Nuitka compiles Python to C, producing a secure binary (protects IP). Electron wraps it with a polished UI. The alternative was pure Electron + API calls, but local execution was required for privacy and offline capability.

### Decision 4: Composio for Tool Integration
**Why:** Building custom integrations for Gmail, Slack, GitHub, Notion, MS 365 individually would take months. Composio provides 800+ pre-built integrations with auth management (OAuth 2.0), letting agents discover and use tools at runtime.

### Decision 5: Context Persistence Architecture
**Why:** Multi-step workflows span sessions. We combine:
- **Short-term memory:** Conversation context within a session
- **Structured task state:** JSON-based task tracking for multi-step workflows
- **Long-term memory:** Persistent storage for user preferences and past interactions

```python
class ContextManager:
    def __init__(self, session_id: str):
        self.short_term = ConversationBuffer(max_tokens=4000)
        self.task_state = TaskStateTracker(session_id)  # Redis-backed
        self.long_term = VectorMemory(session_id)       # For retrieval

    async def get_context(self, user_message: str) -> dict:
        return {
            "conversation": self.short_term.get_recent(k=10),
            "active_tasks": self.task_state.get_pending(),
            "relevant_history": await self.long_term.search(user_message, k=3)
        }
```

## 2.4 Challenges & How You Solved Them

| Challenge | Solution | Impact |
|-----------|----------|--------|
| Agent failures mid-workflow | Error-handling + fallback strategy: retry with different model, graceful degradation, user notification | Critical workflows maintained continuity |
| Cross-platform packaging | Nuitka for Python compilation + Electron wrapper + code signing/notarization for macOS/Windows | Secure distribution on both platforms |
| OAuth token management | Composio handles OAuth flows per integration; tokens stored encrypted; auto-refresh | Seamless auth for 800+ tools |
| Concurrent sessions | FastAPI async + WebSocket ConnectionManager + session isolation | Multiple users/workflows simultaneously |
| Prompt reliability | Few-shot examples, structured JSON output, role-based design, iterative optimization | Improved agent tool accuracy significantly |

## 2.5 Metrics to Mention
- Platform supports Gmail, Microsoft 365, GitHub, Slack, Notion integrations
- Multi-agent system with specialized agents for different domains
- Real-time streaming responses via WebSocket
- Cross-platform: macOS + Windows distribution
- Secure: Code signed, notarized, OAuth 2.0, JWT authentication

---

# SECTION 3: MARS - MULTI-AGENT ANALYTICS PLATFORM (MathCo)

## 3.1 The 30-Second Version

"At MathCo, I led the design of MARS, a multi-agent analytics platform that transformed how our analytics teams worked with data. Users upload datasets, and the platform automatically performs EDA, engineers features, generates insights, and enables model training -- all orchestrated by multiple AI agents. Built with Autogen, FastAPI, and a React frontend, it significantly reduced manual analysis effort."

## 3.2 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  MARS ANALYTICS PLATFORM                  │
│                                                          │
│  ┌──────────────┐         ┌───────────────────────────┐  │
│  │  React UI    │◄──API──►│    FastAPI Backend         │  │
│  │              │         │                           │  │
│  │  - Upload    │         │  ┌─────────────────────┐  │  │
│  │  - Dashboard │         │  │  Autogen Multi-Agent │  │  │
│  │  - Insights  │         │  │                     │  │  │
│  │  - Training  │         │  │  ┌──────┐ ┌──────┐ │  │  │
│  └──────────────┘         │  │  │ EDA  │ │Feature│ │  │  │
│                           │  │  │Agent │ │Agent  │ │  │  │
│                           │  │  └──────┘ └──────┘ │  │  │
│                           │  │  ┌──────┐ ┌──────┐ │  │  │
│                           │  │  │Insight│ │Model │ │  │  │
│                           │  │  │Agent  │ │Agent │ │  │  │
│                           │  │  └──────┘ └──────┘ │  │  │
│                           │  └─────────────────────┘  │  │
│                           │           │               │  │
│                           │  ┌────────┴────────────┐  │  │
│                           │  │  LLM APIs           │  │  │
│                           │  │  (Multiple Providers)│  │  │
│                           │  └─────────────────────┘  │  │
│                           └───────────────────────────┘  │
│                                      │                   │
│                           ┌──────────┴────────────┐      │
│                           │   AWS (EC2, S3, RDS)  │      │
│                           └───────────────────────┘      │
└──────────────────────────────────────────────────────────┘
```

## 3.3 Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Agent Framework | Autogen | Dynamic agent conversations for data analysis tasks |
| Backend | FastAPI | Async support for concurrent analysis sessions |
| Frontend | ReactJS | Interactive dashboards, real-time updates |
| Cloud | AWS | Existing MathCo infrastructure |
| Multi-LLM | Multiple APIs | Different models for different tasks (cheap for EDA, powerful for insights) |

## 3.4 Impact
- Reduced manual data analysis effort significantly
- Accelerated experimentation and decision-making across teams
- Unified platform replacing multiple disconnected tools
- Self-service analytics for non-technical team members

---

# SECTION 4: STELLANTIS - GLOBAL DATA INSIGHTS (MathCo)

## 4.1 The 30-Second Version

"For Stellantis at MathCo, I developed a centralized cross-region customer data platform by consolidating consent, engagement, and interaction data. I integrated a LangChain-powered conversational AI chatbot for natural language querying and designed ML-ready data pipelines in Snowflake and Databricks to standardize datasets into unified feature layers for predictive analytics."

## 4.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│              STELLANTIS DATA PLATFORM                    │
│                                                          │
│  Data Sources (Multi-Region)                             │
│  ┌─────────┐ ┌──────────┐ ┌────────────┐               │
│  │Consent  │ │Engagement│ │Interaction │               │
│  │Data     │ │Data      │ │Data        │               │
│  └────┬────┘ └─────┬────┘ └─────┬──────┘               │
│       └────────────┼────────────┘                       │
│                    ▼                                    │
│  ┌─────────────────────────────────────┐                │
│  │  ETL Pipeline (PySpark + Databricks) │                │
│  │  - Consolidation & Deduplication    │                │
│  │  - Schema Standardization           │                │
│  │  - Feature Engineering              │                │
│  └──────────────┬──────────────────────┘                │
│                 ▼                                       │
│  ┌──────────────────────────┐                           │
│  │ Snowflake (Data Warehouse)│                           │
│  │ - Unified Feature Layers  │                           │
│  │ - ML-Ready Datasets       │                           │
│  └───────────┬──────────────┘                           │
│              ▼                                          │
│  ┌──────────────────────────┐  ┌─────────────────────┐  │
│  │ LangChain Chatbot        │  │ Predictive Analytics │  │
│  │ - NL Querying            │  │ - Personalization    │  │
│  │ - Real-time Insights     │  │ - Customer Segments  │  │
│  └──────────────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 4.3 Key Contributions
- Designed cross-region data consolidation pipeline
- Built LangChain RAG chatbot for NL querying over customer data
- Created unified feature layers in Snowflake for ML pipelines
- Standardized PySpark + Databricks data transformation workflows

---

# SECTION 5: ABBVIE - DATA PIPELINE & ORCHESTRATION (MathCo)

## 5.1 The 30-Second Version

"For AbbVie at MathCo, I built scalable data pipelines transforming raw pharmaceutical data into enterprise-ready insights. I engineered an advanced orchestration system that achieved a **70% improvement in runtime performance**. I used PySpark, Dataiku, and Snowflake, managing end-to-end monthly production refresh cycles."

## 5.2 The 70% Runtime Improvement Story

**Situation:** Monthly production refresh cycles for pharmaceutical data were taking too long, causing delays in analytics and reporting.

**What I did:**
1. **Profiled existing pipelines** -- identified bottlenecks in sequential processing, redundant data reads, and suboptimal joins
2. **Redesigned orchestration** -- parallelized independent tasks, implemented smart dependency tracking
3. **Optimized PySpark jobs** -- broadcast joins for small tables, partition pruning, caching intermediate results
4. **Leveraged EMR efficiently** -- right-sized cluster configurations, spot instances for non-critical jobs

**Result:** 70% runtime improvement, reliable monthly delivery across the organization.

**How to articulate this in an interview:**
"I profiled the entire pipeline and found three main bottlenecks: sequential task execution where parallelization was possible, redundant data reads across stages, and suboptimal PySpark joins. By redesigning the orchestration layer to track dependencies and parallelize independent tasks, optimizing PySpark operations like broadcast joins and partition pruning, and right-sizing our EMR clusters, we achieved a 70% improvement in runtime. This was recognized with the MathCo Excellence Award."

## 5.3 Technical Details
- **Data flow:** Snowflake → Business Logic → CDL Infrastructure
- **Automated data quality:** Proactive validation frameworks ensuring data integrity
- **Tools:** Dataiku for visual workflow design, PySpark for heavy transformations
- **Scale:** Enterprise-scale pharmaceutical data with regulatory compliance

---

# SECTION 6: TAKEDA - CLOUD MIGRATION (MathCo)

## 6.1 The 30-Second Version

"I led the migration of a production application from Azure to AWS with zero downtime. I re-architected cloud infrastructure, services, and CI/CD pipelines, and created comprehensive documentation for long-term maintainability."

## 6.2 Key Points to Emphasize
- **Zero-downtime migration** -- planned cutover strategy with blue-green deployment
- **Service mapping:** Azure Functions → AWS Lambda, Azure Blob → S3, Azure SQL → RDS
- **CI/CD pipeline migration:** Rebuilt pipelines for AWS-native deployment
- **Documentation:** Architecture decisions, deployment workflows, best practices

---

# SECTION 7: FEALTY TECHNOLOGIES - PRODUCTION WEB APPLICATIONS

## 7.1 The 30-Second Version

"At Fealty Technologies, I led development of production web applications across e-commerce, healthcare, and enterprise domains. I built backends with Django, FastAPI, and Flask, designed PostgreSQL and MongoDB schemas with optimization, and managed Docker/Nginx deployments on AWS."

## 7.2 Key Points for Interviews
- **Full-stack backend:** Django, FastAPI, Flask -- know when to use each
- **Database design:** PostgreSQL schemas, indexing strategies, MongoDB for semi-structured data
- **DevOps:** Docker containerization, Nginx reverse proxy/load balancing
- **Mentoring:** Trained junior developers on best practices
- **Breadth:** E-commerce, healthcare, enterprise -- adaptable to any domain

---

# SECTION 8: AG2AI AUTOGEN - OPEN SOURCE CONTRIBUTION

## 8.1 The 30-Second Version

"I contributed to AG2AI Autogen, an open-source multi-agent framework. I built a real-time WebSocket UI with a FastAPI backend for interactive agent debugging and live LLM communication monitoring. The PR is at github.com/ag2ai/ag2/pull/2062."

## 8.2 What You Built
```
┌──────────────────────────────────────┐
│     WebSocket Debugging UI           │
│  ┌───────────┐    ┌───────────────┐  │
│  │ Agent Chat │    │ LLM Monitor   │  │
│  │ Debugger   │    │ (Live Comms)  │  │
│  └─────┬─────┘    └───────┬───────┘  │
│        └──────┬───────────┘          │
│               ▼                      │
│  ┌──────────────────────────┐        │
│  │  FastAPI + WebSocket      │        │
│  │  - Session management     │        │
│  │  - Streaming responses    │        │
│  │  - Message formatting     │        │
│  │  - User Proxy agent input │        │
│  └──────────────────────────┘        │
└──────────────────────────────────────┘
```

## 8.3 Why This Matters for Interviews
- Shows you contribute to the open-source community
- Demonstrates deep understanding of AG2/Autogen internals
- WebSocket + FastAPI is the same stack you use at RavianAI
- Practical tool that improved developer experience for the entire AG2 community

---

# SECTION 9: GDG TALK - "RETHINKING AI AGENTS"

## 9.1 How to Reference This

"I was invited to speak at AI Manthan: Agents at Work, a GDG Cloud New Delhi event, on 'Rethinking AI Agents: Building and Aligning Autonomous Workflows with an Artificial General Agent.' I discussed how AI agents can autonomously plan, reason, and execute workflows across domains."

**Why this matters:** Shows thought leadership, public speaking ability, and deep expertise in agentic AI. Interviewers value candidates who can communicate complex technical ideas.

---

# SECTION 10: COMMON FOLLOW-UP QUESTIONS & HOW TO ANSWER

## About RavianAI

**Q: How do you handle agent failures in production?**
"We implemented a three-tier fallback strategy: (1) Retry with adjusted parameters, (2) Fall back to a simpler model or approach, (3) Graceful degradation with user notification. Critical workflows have circuit breakers that prevent cascading failures."

**Q: How do agents decide which tool to use?**
"We built a dynamic capability registry where agents reason about available tools at runtime. Each tool has metadata describing its capabilities, required permissions, and context. The agent matches task requirements to tool capabilities, considering context and permissions."

**Q: How do you handle security with third-party integrations?**
"OAuth 2.0 for all third-party auth, with Composio managing token refresh. JWT for internal authentication. All tokens encrypted at rest. Permissions are scoped per-integration. Users control which integrations are active."

**Q: Why Nuitka instead of PyInstaller?**
"Nuitka compiles Python to C and then to native binary, providing both performance improvement and IP protection. PyInstaller just bundles the interpreter with bytecode, which is easier to reverse-engineer. For a commercial desktop product, Nuitka's compilation approach was essential for security."

## About MARS

**Q: How did you handle multiple LLM providers?**
"We built an abstraction layer that routes to different models based on task complexity. Simple EDA tasks go to cheaper, faster models. Complex insight generation uses more powerful models. This optimizes both cost and quality."

**Q: How did agents coordinate on the MARS platform?**
"Using Autogen's GroupChat with custom speaker selection. The orchestrator agent decomposes the user's request, and specialized agents (EDA, feature engineering, insight generation, model training) are invoked based on the task pipeline."

## About AbbVie

**Q: Walk me through the 70% performance improvement.**
"I identified three bottlenecks: (1) Sequential task execution where 40% of tasks were independent and could run in parallel, (2) Redundant data reads -- the same Snowflake tables were read multiple times across pipeline stages, so I added caching, (3) Suboptimal PySpark joins on large tables -- I switched to broadcast joins where one table was small, added partition pruning, and right-sized EMR clusters. The combined effect was 70% faster end-to-end runtime."

## General Technical Questions

**Q: How do you choose between AG2/Autogen and LangGraph?**
"It depends on the workflow: AG2 excels when agents need dynamic, conversation-style collaboration -- like MARS where agents discuss data findings. LangGraph is better for deterministic workflows with precise state management and conditional routing. At RavianAI, we use AG2 for the core agent orchestration but could use LangGraph for specific deterministic sub-workflows."

**Q: What's your approach to prompt engineering for agents?**
"I use structured prompts with clear sections: role definition, available tools with descriptions, decision framework, output format, and constraints. I iterate using few-shot examples of successful tool calls, structured JSON output schemas, and role-based design where each agent has a specific persona and expertise."

**Q: How do you handle data quality in pipelines?**
"At AbbVie, I implemented automated data quality frameworks with proactive validation -- schema validation, null checks, range checks, statistical distribution monitoring, and cross-source reconciliation. Issues trigger alerts before downstream consumption."

---

## Sources
- Resume of Surya Pratap Singh Rathore
- [AG2AI Autogen GitHub PR #2062](https://github.com/ag2ai/ag2/pull/2062)
