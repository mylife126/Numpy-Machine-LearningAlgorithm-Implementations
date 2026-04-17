## 1.1 What long-horizon orchestration means

中文先理解

long-horizon orchestration 不是“记住更多聊天记录”，而是：

	•	任务会持续很久
	•	中间结果会改变后续计划
	•	有多个人 / 多个 agent 接力
	•	中途可能失败
	•	不能每次都从头开始

所以目标不是单次回答对，而是：

让 work progress through adjustment rather than reset

也就是出问题时能修局部、保留有效结果、继续往前走。这个正是文档对成功标准的定义。 ￼

⸻

1.2 Core design

我会把 long-horizon orchestration 设计成 5 个核心构件：

	1.	Case File / blackboard：持久化共享状态
	2.	Durable workflow graph：把任务表示成可恢复的图
	3.	Constrained skills + validators：每步有明确 I/O contract
	4.	Supervisor：读取 Case File，决定下一步、是否重规划、是否升人
	5.	Recovery engine：失败时只重跑受影响步骤，而不是整链重来

这基本就是文档给出的解法。 ￼

⸻

1.3 Business case anchor

继续沿用我们固定的 case：

新 PM 接手一个 delayed project，要理解项目、确认 blockers、生成 recovery plan，并在后续几天根据新 evidence 持续更新。

为什么这是 long-horizon：

	•	第一轮 summary 只是开始
	•	blocker 可能变化
	•	plan 会根据新的 ticket / meeting notes 变化
	•	某一步被 reviewer 否掉后要局部修改
	•	后面还要触发 owner sync / task updates

⸻

1.4 Architecture sketch
```aiignore
User / PM request
  ↓
Create or update Case
  ↓
Case File / Blackboard
  ├─ objective
  ├─ entities / owners
  ├─ evidence + provenance
  ├─ assumptions
  ├─ intermediate outputs
  ├─ validation results
  ├─ approval state
  └─ workflow status
  ↓
Supervisor
  ├─ select workflow template
  ├─ expand / modify graph
  ├─ choose next skill
  ├─ attach context bundle
  ├─ inspect validator output
  └─ decide continue / retry / replan / escalate
  ↓
Durable Workflow Graph
  ├─ retrieve context
  ├─ summarize project
  ├─ diagnose blocker
  ├─ resolve contradictions
  ├─ draft recovery plan
  ├─ human review
  └─ execute follow-up actions
  ↓
Execution log + checkpoints
  ↓
Recovery / selective re-execution

```
## 2. Technical breakdowns

⸻

2.1 Core abstraction: Case, not query

中文梗概

long-horizon orchestration 的第一步，是把用户请求从“一个 query”升级成“一个 case”。

因为 query 是一次性的；case 是持续推进的。

What I would do

当用户第一次说：

“I’m taking over Atlas. Help me understand what’s blocked and what we should do next.”

系统不会只给一个 response。
而是创建：
```aiignore
{
  "case_id": "atlas_takeover_0426",
  "goal": "understand project + identify blockers + propose recovery plan",
  "status": "active",
  "priority": "high",
  "stakeholders": ["new PM", "eng lead", "legal reviewer"],
  "workflow_version": "project_takeover_v1"
}
```

Why this matters

这样后续所有动作都挂在这个 case 上：

	•	新 evidence
	•	新 plan
	•	reviewer comments
	•	approval state
	•	partial re-execution

⸻

2.2 Case File / Blackboard

中文梗概

Case File 不是“用户画像 memory”，而是：

workflow 的共享工作台 + reasoning trace + persistent state

What it stores

我会存 6 类信息：

A. Objective and intent

	•	当前目标是什么
	•	success criteria 是什么

B. Workflow progress

	•	哪些 step 已完成
	•	哪些 step 在 pending / failed / approved

C. Evidence and provenance

	•	用了哪些 doc / ticket / dashboard / meeting note
	•	来源、时间戳、owner、authority

D. Intermediate outputs

	•	summary draft
	•	blocker hypotheses
	•	plan versions
	•	clarification questions asked

E. Assumptions and decisions

	•	为什么认定 blocker 是 infra 或 legal
	•	哪个 source 被优先采信
	•	为什么选择 workflow variant A

F. Validation and approval state

	•	哪个 validator failed
	•	哪一步需要 human review
	•	哪个输出已被接受

Why it is essential

文档里明确说 shared Case File tracks intent and progress across all steps and sessions；每个 material step 的 evidence、assumptions、validation outcomes 也要保存在 workflow state 里，方便诊断和恢复。 ￼  ￼

技术实现建议

	•	structured DB：case metadata / status / step state
	•	object store：artifacts / drafts / logs
	•	event log：step transitions / decisions / retries
	•	vector index：case-relevant evidence / semantic lookup

⸻

2.3 Durable workflow graph

中文梗概

不能把 workflow 写成固定 pipeline，因为现实会变。
但也不能完全让 LLM 每次自由发挥。
所以最稳妥的是：

durable graph + constrained skills

Graph design

每个 node 是一个 skill，例如：

	1.	collect_project_context
	2.	summarize_project_state
	3.	identify_blocker_candidates
	4.	resolve_conflicting_claims
	5.	generate_recovery_plan
	6.	validate_plan
	7.	route_for_human_review
	8.	execute_follow_up_actions

每个 node 都定义：

	•	input schema
	•	output schema
	•	required context
	•	validator
	•	retry / escalation policy

Why graph is better

	•	支持 dependency tracking
	•	支持 checkpoint
	•	支持 partial recompute
	•	支持 branching / loops
	•	支持把 human 也当成 executor node

文档里原话就是 durable workflow graph，each step implemented by constrained skill with clear input/output contracts and required validators。 ￼

⸻

2.4 Supervisor 迭代更新， call skill nodes

中文梗概

Supervisor 是真正的 orchestration brain。

它不是简单 “choose tool”，而是每一步都要看：

	•	现在 case 处于什么状态
	•	上一步结果好不好
	•	哪个假设变了
	•	是否需要加 context
	•	是否要换 workflow
	•	是否该让人介入

Decision loop

我会让 supervisor 每一轮都执行：

Step 1: read current Case File
看 intent、progress、evidence、failures、approvals

Step 2: select next valid step
根据 workflow graph 和当前状态找下一步

Step 3: construct execution package
决定：

	•	skill
	•	context bundle
	•	executor
	•	confidence threshold

Step 4: inspect validator results
如果失败，决定：

	•	retry with better context
	•	switch to another skill
	•	branch into conflict resolution
	•	escalate to human

Step 5: update Case File
记录结果、理由、下一状态

Case mock:

```aiignore


That’s a great question. I think the key idea is that the workflow graph is dependency-aware, not just a linear pipeline.

If an upstream node changes, we don’t necessarily recompute everything downstream.
Instead, we identify the minimal set of affected nodes and selectively re-execute only those parts of the graph.

Let me give a concrete example.

Suppose we have a workflow for a new PM taking over a project:

Step 1: collect project context
Step 2: generate project summary
Step 3: identify blockers
Step 4: generate recovery plan

In the first iteration, the system retrieves documents and concludes that the main blocker is infrastructure.
Based on that, it generates a recovery plan focusing on scaling fixes.

Now, in a second iteration, a new ticket or meeting note indicates that the infrastructure issue was already fixed, and the real blocker is legal approval.

In this case, we don’t need to recompute everything.
The project context node may still be valid, and the summary might still be mostly correct.

What we do instead is:
	•	invalidate the blocker diagnosis node
	•	invalidate the recovery plan node that depends on it
	•	keep earlier valid outputs such as context and summary
	•	re-run blocker identification with updated evidence
	•	then regenerate the recovery plan based on the new blocker

This allows the system to preserve valid work, maintain continuity, and avoid unnecessary recomputation.

Each node is implemented as a constrained skill with clear input/output contracts and validators, so we know exactly what depends on what, and how to safely re-execute only the affected parts.

So overall, the system progresses through adjustment rather than full reset.
```

2️⃣ node 怎么定义 dependency？

每个 node（skill）要有：

	•	input schema（输入）
	•	dependency list（依赖哪些上游结果）
	•	output artifact（产物）
	•	validator（质量检查）

例如：

blocker node：
```aiignore
{
  "inputs": ["project_summary", "tickets", "meeting_notes"],
  "outputs": ["blocker_hypothesis"],
  "validator": "evidence_coverage + contradiction_check"
}
```
3️⃣ invalidation 是怎么做的？

当某个输入变了：

Step 1：检测变化类型

	•	new evidence
	•	stale assumption
	•	conflicting claim

⸻

Step 2：标记 affected nodes

	•	找哪些 node 用了这个输入
	•	标记为 invalid

⸻

Step 3：cascade（但不是全量）

	•	只影响 dependent nodes

⸻

Step 4：partial re-execution

	•	重跑 affected subgraph
	•	更新 Case File

⸻

4️⃣ skill node 是怎么跑的？
```aiignore
inputs (from case file + retrieval)
→ context construction
→ LLM / tool execution
→ output artifact
→ validator
→ write back to case file
```

## 总结！
好，这一题我们做一个终极 consolidated answer，帮你做到两件事：

	1.	保留你刚才的理解（graph / dependency / RAG / node）
	2.	升级成 Gaia 想听的 framing（workflow execution + case state + iteration）

并且我会帮你嵌入一个完整 example（两轮迭代），你可以直接讲出来。

⸻

🧠 中文梗概（你快速过一遍）

👉 正确理解框架（你现在应该这样想）：

1️⃣ 这个系统不是 dynamic RAG

而是：

一个以 Case 为中心的 workflow execution system

⸻

2️⃣ RAG 在里面的角色

👉 只是其中一种 node（skill）：

	•	用来 fetch context
	•	用来 grounding
	•	用来支持 decision

但不负责：

	•	workflow推进
	•	dependency管理
	•	failure恢复

⸻

3️⃣ orchestration 的本质

根据 Case 的状态，决定下一步做什么 + 用什么 context + 是否需要重跑某些节点

⸻

4️⃣ graph execution 的核心能力

	•	dependency-aware
	•	selective re-execution
	•	state update（不是只更新knowledge，是更新整个case state）

⸻

5️⃣ 最关键一句话（记住）

This is not knowledge evolving — this is workflow state evolving.

⸻

🎯 你直接说（英文，可复述，含 example）

这段是可以直接用来面试的👇

⸻

I think my understanding is that this system can be partially viewed as a dynamic knowledge system, but I would frame it more accurately as a case-oriented workflow orchestration system rather than just a RAG system.

Retrieval is definitely an important component, and each node in the workflow may use RAG to gather context. But the system is not primarily about retrieving knowledge — it is about managing how a task progresses over time.

Concretely, I would model the system as a graph-based execution workflow, where each node is a skill with a well-defined input and output schema.
The inputs can come from two sources: global knowledge retrieved via RAG, and intermediate outputs from previous nodes stored in the Case File.

The key idea is that the system maintains a persistent Case state, and each node updates that state with structured artifacts, such as summaries, blocker hypotheses, plans, or validation results.

Let me walk through a simple example to illustrate how this works across iterations.

In the first iteration, suppose a new PM takes over a project and asks for help understanding its status and next steps.
The system creates a case and executes a workflow:
	•	Node 1 retrieves project context from documents, tickets, and meeting notes
	•	Node 2 generates a project summary
	•	Node 3 identifies blocker candidates
	•	Node 4 generates a recovery plan

At this point, the system concludes that the main blocker is infrastructure, and generates a plan focusing on scaling fixes.
All of these outputs are written into the Case File as structured state.

Now in a second iteration, new information comes in — for example, a recent ticket or meeting note indicates that the infrastructure issue was already fixed, and the real blocker is legal approval.

The system does not restart from scratch.
Instead, it updates the Case File with the new evidence, and the orchestrator analyzes dependencies between nodes.

It then:
	•	invalidates the blocker identification node
	•	invalidates the recovery plan node that depends on it
	•	keeps earlier valid outputs such as context and summary
	•	re-runs blocker identification with updated context
	•	regenerates the recovery plan based on the new blocker

This is a dependency-aware, selective re-execution process.
Each node reads from the updated Case state and writes back new structured outputs, so the knowledge of the system evolves together with the workflow.

So overall, I would not describe this as a dynamic RAG system.
Instead, it is a stateful workflow system where retrieval supports execution, and the orchestrator continuously updates the case state as the task evolves over time.

⸻

🔁 如果对方追问（你可以接）

❓ Where exactly is RAG used?

👉

RAG is used inside specific nodes, for example in context retrieval or evidence gathering.
But it is always controlled by the orchestrator and scoped to the current step in the workflow.

⸻

❓ What exactly is being updated over time?

👉

Not just knowledge, but the entire case state, including intermediate outputs, assumptions, decisions, and validation results.

⸻

🧠 技术讲解（帮你稳住逻辑）

⸻

1️⃣ 你原本的理解 vs 升级版本

你说的：

	•	graph
	•	dependency
	•	node 输入输出
	•	RAG fetch context

👉 全对

⸻

但你少了一层：

state + execution + iteration

⸻

2️⃣ graph execution 的本质

Node = skill
Edges = dependency
Execution = read case state → produce artifact → update case state


⸻

3️⃣ Case File 更新的是什么？

不是：

❌ “knowledge text”

而是：
```
{
  "objective": "...",
  "summary": "...",
  "blocker": "...",
  "plan": "...",
  "assumptions": "...",
  "evidence": "...",
  "validation": "...",
  "status": "..."
}
```
👉 是一个 structured evolving state

⸻

4️⃣ iteration 的本质

每一轮不是：

重新回答问题

而是：

更新 case state + 修正 workflow


⸻

5️⃣ 你这个回答最加分的点

你做对了两件非常关键的事：

✅ 用 graph + dependency 解释 system

→ 很 technical

✅ 用 example 展示 iteration

→ 很 practical

⸻

✅ 最后一条（你面试可以用的总结）

I would think of the system as executing and updating a case over time, where each node contributes structured outputs, and the orchestrator ensures the workflow adapts correctly as new information arrives.

