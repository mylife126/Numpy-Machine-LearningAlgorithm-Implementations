# High level system design

我会按你要的 Airbnb ML system design 模版 来出：

	1.	High-level consolidated summary（脑图 / 先建立全局）
	2.	Technical breakdowns（逐层拆系统）
	3.	English narrative（你面试时可直接说）
	4.	Likely follow-ups（下一层深挖）

这题我们固定业务场景为：

```
A new PM takes over a delayed project, wants to understand project status, find the right docs, identify blockers, and generate a recovery plan.

这个场景和 Gaia 非常匹配，因为文档强调 Gaia 的核心是把 isolated AI experiments 升级成跨人类与 AI 的 coordinated workflows，并且持续优化 how work gets done。 ￼  ￼
```
⸻

## 1. High-level consolidated summary

1.1 Problem framing

这不是一个“问答助手”题，而是一个 case-based human-AI workflow orchestration 题。

用户表面请求是：

	•	帮我理解项目
	•	找关键文档
	•	看当前 blocker
	•	给 recovery plan

但系统真正要做的是：

	•	把这个请求变成一个 Case
	•	维护跨步骤、跨 session 的 persistent state
	•	协调 retrieval / summarization / planning / validation / human review
	•	在知识冲突或 plan 不可信时，主动 escalate / abstain / recover
	•	从执行结果中学习，持续优化 team-specific workflow policy

⸻

1.2 Core design principle

我会把系统分成两层：

Layer A: Task-level assistant

负责单步能力：

	•	intent understanding
	•	retrieval / RAG
	•	summarization
	•	role-aware output generation

Layer B: Workflow-level operating system

负责 Gaia 真正核心：

	•	case creation
	•	durable workflow graph
	•	case file / shared blackboard
	•	multi-agent + human routing
	•	validation / recovery
	•	optimization loop

这和文档里强调的 dynamic orchestrator、shared workspace、continuous optimization、task executor 完全一致。 ￼  ￼

⸻

1.3 End-to-end flow（脑图）

```
User request
  ↓
User Interaction Layer
  ↓
Intent + role + ambiguity detection
  ↓
Create / update Case
  ↓
Orchestrator
  ├─ select workflow template
  ├─ decompose into steps
  ├─ choose AI / human routing
  └─ decide required context bundle
  ↓
Knowledge Management System
  ├─ retrieve docs / tickets / meeting notes / code / dashboards
  ├─ resolve entity aliases
  ├─ detect contradictory claims
  └─ attach provenance + confidence
  ↓
Case File / Blackboard
  ├─ task intent
  ├─ assumptions
  ├─ intermediate results
  ├─ decision rationale
  ├─ validation outcomes
  └─ next-step state
  ↓
Execution steps
  ├─ project summary
  ├─ blocker diagnosis
  ├─ recovery plan draft
  ├─ human review if high-stakes
  └─ task execution / update external systems
  ↓
Optimization loop
  ├─ monitor rework / escalation / latency / acceptance
  ├─ reason about root cause
  └─ adjust routing / context / workflow policy
```

⸻

1.4 What success looks like

不是“回答得像不像”，而是：

	•	新 PM 更快理解项目
	•	找文档更快
	•	blocker diagnosis 更准
	•	recovery plan 被接受率更高
	•	rework / repeated clarification 下降
	•	workflow 能在冲突和失败下继续推进，而不是重来

文档里强调 Gaia 的目标是让系统 not just finish a task, but confirm it solved the problem，并持续改进 execution cycle。 ￼

⸻

## 2. Technical breakdowns

⸻

### 2.1 User Interaction Layer

Goal

接收自然语言、多模态输入，并把模糊的需求转成结构化 case。

What I would build

输入：

	•	text / voice / docs / pasted tickets / screenshots

输出：

	•	structured intent
	•	user role
	•	urgency
	•	ambiguity flags
	•	whether this is a new case or follow-up to an existing case

Example: 

```
用户说：

“I’m taking over Atlas. Help me understand what’s blocked and what we should do next.”

解析成：

{
  "case_type": "project_takeover",
  "subtasks": ["project_summary", "blocker_diagnosis", "recovery_plan"],
  "role": "PM",
  "risk_level": "medium",
  "need_clarification": false
}
```
#### Why this matters

**因为 Gaia 文档明确说 interaction layer 不只是被动接收输入，还会在 intent 不清楚时主动提问，并根据任务选择最合适的输出形式** ￼

⸻

### 2.2 Case creation and Case File / Blackboard

Core idea

这是 Gaia 和普通 agent 最大差异点。

我不会只维护 chat history，而会在用户第一次请求时创建一个 Case，并维护一个持久化的 Case File。

What goes into Case File

	•	case objective
	•	current project / entities
	•	stakeholders / owners
	•	retrieved evidence and provenance
	•	assumptions
	•	blockers and hypotheses
	•	generated artifacts (summary / plan drafts)
	•	validation results
	•	approval state
	•	execution status

Why

文档明确说 long-horizon orchestration 要靠 shared Case File / blackboard，跟踪 intent 和 progress；workflow 需要能被 inspect、resume、replay，并在变化发生时只重跑受影响步骤。 ￼

My framing

你可以把它理解成：
	•	一半是 workflow state store
	•	一半是 reasoning trace
	•	一半是 recoverable execution log

⸻

### 2.3 Orchestrator

Role

Gaia 的 brain，不是简单 tool router，而是 dynamic planner + coordinator。

What it does
	1.	识别当前是新 case 还是 existing case continuation
	2.	选择 workflow template
	3.	把 case 分解成多步子任务
	4.	给每一步选 executor：AI agent / human / API / external system
	5.	为每一步准备 context bundle
	6.	读取 validator 结果，决定继续、重试、改路由还是 escalate

For this use case

我会设计一个 workflow template：

Step 1: collect relevant project context
Step 2: generate project summary
Step 3: identify blocker candidates
Step 4: resolve conflicting evidence
Step 5: propose recovery plan
Step 6: human review if needed
Step 7: publish / execute next-step recommendations

Why this matches doc

文档对 orchestrator 的定义就是 interprets intent, decomposes tasks, selects or creates workflows, routes to AI or humans, maintains shared workspace, and continuously re-evaluates next steps based on actual outcomes。 ￼

⸻

2.4 Knowledge Management System

This is not just enterprise search

如果我只说“做个 RAG”是不够的。

It needs to do 4 things
	1.	Ingest multi-modal organizational knowledge
	•	docs
	•	tickets
	•	meeting notes
	•	Slack / chat
	•	dashboards
	•	code / PRs
	•	calendars / milestones
	2.	Normalize and structure
	•	chunk text
	•	entity extraction
	•	ownership
	•	timestamp
	•	source type
	•	access control
	3.	Hybrid retrieval
	•	dense retrieval for semantic match
	•	sparse / keyword retrieval for project names, acronyms, IDs
	•	reranker for precision
	4.	Contradiction resolution
	•	claim extraction
	•	compare authority / recency / corroboration
	•	preserve provenance
	•	abstain or ask targeted clarification if unresolved

Why this is crucial

文档明确说 knowledge system 不只是 capture and retrieve，它还必须 resolve conflicting information，并在冲突无法安全解决时保留 uncertainty、升级给人。 ￼

Example
	•	doc says blocker = infra
	•	latest ticket says fixed
	•	meeting note says legal review is actual blocker

系统不能直接 collapse 成一句 summary。
必须先生成 claim graph，再根据 timestamp、owner、system-of-record、corroboration 选择更可信 interpretation。

⸻

2.5 Memory and personalization

Important distinction

这里我会刻意分两类 memory：

A. User / team personalization memory
	•	role = PM / engineer / manager
	•	preferred summary depth
	•	frequently referenced projects
	•	team-specific workflow norms

B. Workflow / case memory
	•	current intent
	•	intermediate outputs
	•	prior assumptions
	•	previously resolved conflicts
	•	approval history
	•	execution progress

Why this matters

Gaia 文档里的 personalization 不是普通“记住用户偏好”，而是 team-specific policy learning，让 workflow variant、context bundle、routing decisions 随 team 的 outcome 逐渐优化。 ￼

Storage design
	•	structured profile store：role / team / stable preferences
	•	vector memory：past relevant cases / summarized interactions
	•	case file：current workflow state

Output adaptation example

同样是 “What is the current status of Atlas?”
	•	engineer version：technical blockers, dependencies, failing checks
	•	PM version：risks, owners, timeline, escalation needs

⸻

2.6 Planner / generation layer

What generation is used for

不是直接“让 LLM 给答案”，而是用于：
	•	summarization
	•	blocker hypothesis generation
	•	recovery plan drafting
	•	targeted clarification question generation
	•	justification synthesis for user-facing explanations

Guardrails
	•	grounded generation only
	•	must cite retrieved evidence
	•	distinguish fact vs inference
	•	if evidence conflict unresolved, surface uncertainty explicitly

Why

文档很强调 visibility, inspectability, and closed-loop validation，不能 black box 地输出一个 plan。 ￼

⸻

2.7 Validators and human-in-the-loop

Validators

每个关键步骤要有 validator，不然 workflow graph 只是 fancy prompt chain。

比如：

Summary validator
	•	coverage of key entities
	•	no unsupported claims
	•	freshness check

Blocker diagnosis validator
	•	evidence coverage across systems
	•	contradiction unresolved?
	•	confidence threshold met?

Recovery plan validator
	•	plan aligns with actual blockers
	•	dependencies included
	•	owners assigned
	•	timeline realistic

Human-in-the-loop

对于高风险动作：
	•	modifying tracker
	•	notifying stakeholders
	•	creating task changes
	•	changing configs

默认 human approval。

这和文档的原则 “Escalation is a capability, not a failure” 以及 task executor 中的 confidence-based autonomy 一致。 ￼  ￼

⸻

2.8 Task Executor

Goal

turn decisions into outcomes

What it can do
	•	create/update Jira tasks
	•	notify owners
	•	generate docs or launch checklist
	•	update skill files / context files
	•	call external API / MCP
	•	route to third-party agent
	•	request human review

Infrastructure requirements
	•	idempotency
	•	rollback / compensation
	•	audit log
	•	permission checks
	•	executor registry

文档明确把执行层定义为多机制执行、授权 agent 路由、traceability、rollback、recoverability。 ￼

⸻

2.9 Optimization loop

This is one of the biggest Gaia-specific additions

不是“调 prompt”这么简单。

3-stage optimization loop

1. Observe
收集：
	•	rework rate
	•	review cycles
	•	plan acceptance
	•	escalation frequency
	•	resolution latency
	•	user override rate

2. Reason
形成 root-cause hypotheses：
	•	missing context
	•	stale knowledge
	•	wrong workflow template
	•	wrong routing
	•	inadequate examples
	•	user role mismatch

3. Act
保守地调整：
	•	context bundle selection
	•	workflow variant
	•	agent routing
	•	prompt / skill config
	•	escalation threshold

Learning style

我会用 scoped conservative policy updates
	•	team-scoped
	•	workflow-scoped
	•	bandit-style updates across a small set of alternatives

这正对应文档说的 versioned policy、observable outcomes、bandit-style learning、scoped updates。 ￼

Why this connects to your background

你可以自然映射到你做过的 modular retrieval/ranking + feedback loop + semi-real-time personalization。你简历里已经做过 centralized reusable architecture、feedback loop continuous learning、intent-aware personalization。 ￼  ￼

⸻

2.10 Evaluation

我会分三层：

A. Retrieval / knowledge metrics
	•	Recall@K
	•	Precision@K
	•	claim coverage
	•	freshness
	•	permission correctness
	•	contradiction detection rate

B. Workflow metrics
	•	case completion rate
	•	resume success after interruption
	•	partial re-execution success
	•	blocker diagnosis accuracy
	•	recovery plan acceptance rate
	•	escalation appropriateness

C. Business / user metrics
	•	time-to-context for new PM
	•	time-to-find-right-doc
	•	reduction in repeated questions
	•	lower rework
	•	faster recovery from delays
	•	trust / transparency satisfaction

D. Optimization metrics
	•	policy improvement lift
	•	lower rework after policy update
	•	lower review cycles
	•	no instability across teams

⸻

2.11 MVP and phased rollout

Phase 1

Read-heavy assistant:
	•	project understanding
	•	doc retrieval
	•	blocker summary
	•	cited recommendations

Phase 2

Case-aware workflows:
	•	persistent case file
	•	multi-step plan generation
	•	conflict resolution
	•	human review

Phase 3

Action-taking system:
	•	task creation
	•	execution coordination
	•	config updates
	•	closed-loop optimization

Why this rollout

先从 low-risk/high-value 的 understanding 开始，再到 planning，再到 write actions。这样既建立 trust，又逐步验证 recovery / execution infrastructure。

⸻

3. English narrative（面试可直接说）

I’d design Gaia for this use case as a case-based human-AI workflow system rather than a single-turn assistant.

The user request may look like a simple question — for example, a new PM wants to understand a delayed project, identify blockers, and generate a recovery plan. But in practice, this is a long-running workflow. The system needs to gather context from many sources, resolve conflicting information, generate intermediate outputs, validate them, and potentially continue execution across multiple steps and sessions.

So I would structure the system in two layers.

The first is a task-level assistant layer. This handles intent understanding, hybrid retrieval, grounded summarization, and role-aware output generation. For example, it determines whether the user needs project understanding, blocker diagnosis, or planning support, retrieves the right documents and signals, and produces grounded outputs with citations.

The second is the workflow layer, which is the Gaia-specific part. When the user initiates the request, I would create a persistent Case and maintain a shared Case File or blackboard. This stores task intent, intermediate results, assumptions, decisions, validation outcomes, and execution state. That allows the system to maintain continuity, support human handoffs, and recover or partially re-execute workflows instead of restarting from scratch.

On top of this shared state, I would build an Orchestrator that acts as a dynamic planner. It selects or creates a workflow, decomposes the task into steps such as context gathering, summary generation, blocker diagnosis, conflict resolution, and recovery planning, then routes each step to the right executor — an AI agent, a human, or an external tool. After each step, validators check whether the output is trustworthy enough to continue. If not, the orchestrator can retry with better context, switch workflows, or escalate to a human.

The knowledge layer would go beyond standard RAG. It needs to ingest documents, tickets, meeting notes, dashboards, and other organizational artifacts, preserve provenance, and explicitly resolve contradictory claims using authority, recency, and corroboration. If the system cannot safely resolve the conflict, it should preserve uncertainty and ask a targeted follow-up or escalate rather than hallucinate a confident answer.

I would also include personalization, but I would define it more broadly than user preference. There is user and role-aware personalization, such as adapting outputs differently for a PM versus an engineer, but there is also team-specific workflow personalization. I would represent important decisions such as workflow variant, routing policy, or context bundle as versioned policies, and optimize them conservatively based on downstream outcomes like rework, escalation rate, and resolution latency.

The execution layer would support actions such as drafting a recovery plan, creating follow-up tasks, notifying owners, or updating downstream systems, but I would gate higher-risk actions behind confidence thresholds and human approval. Traceability, rollback, and auditability are essential.

In terms of evaluation, I would measure retrieval and conflict-resolution quality, workflow success such as accurate blocker diagnosis and plan acceptance, and business outcomes such as time-to-context and reduced coordination overhead.

If I were prioritizing an MVP, I would start with project understanding and blocker diagnosis in a read-heavy mode, then add persistent case state and planning, and only later expand into higher-risk execution.

Overall, I would evolve the system from a retrieval-augmented assistant into a stateful operational intelligence layer that can manage long-horizon workflows, coordinate humans and AI, and improve how teams execute work over time.

⸻

4. Likely follow-ups

Q1. What is the hardest technical part?

Answer direction:
Long-horizon state management under changing evidence. Not retrieval itself, but keeping intent, reasoning trace, and partial progress consistent as workflows evolve.

⸻

Q2. Why isn’t standard RAG enough?

Answer direction:
Because organizations have conflicting, stale, and partial knowledge. Gaia needs provenance-aware claim resolution, uncertainty handling, and abstention, not just retrieval.

⸻

Q3. What makes this different from a project-tracking tool?

Answer direction:
Project tools track status and ownership. Gaia tracks decision rationale, evidence, validation, and execution flow, so it can reason, adapt, and recover.

⸻

Q4. How would you personalize safely?

Answer direction:
Use scoped policies. Learn at explicit decision boundaries, keep updates local to team/workflow, use conservative bandit-style changes, and preserve stable global defaults.

⸻

Q5. How would you connect this to your past experience?

Answer direction:
Talk about transforming fragmented pipelines into modular reusable architecture, adaptive retrieval/ranking, feedback loops, and semi-real-time persona/personalization; then say Gaia is the workflow-level extension of that same systems problem.  ￼  ￼

⸻

下一题最适合直接接：

2️⃣ Deep dive: long-horizon orchestration（case file + workflow + recovery）