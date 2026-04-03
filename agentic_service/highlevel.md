# Enterprise Collaborator 
一句话定位

Gaia Intelligence = 面向企业的人机协作操作系统的大脑层。

它的目标是解决企业 agent 落地的“最后一公里”问题：不是只自动化一个 isolated step，而是让 AI 真正参与跨团队、跨系统、跨时间的 end-to-end workflow。 ￼

⸻

## 1) 它到底在解决什么问题

A. 企业里 agent 迟迟不能 scale 的根因

文档反复强调 6 个底层痛点：

	•	工作不是单步，而是长链路、多依赖、会动态变化
	•	AI 没有组织上下文，就会 output 漂、要求和设计 drift
	•	真正 workflow 永远是 human judgment + AI execution
	•	局部低效会在 network 里被放大成系统性拖累
	•	LLM 非确定、昂贵，所以必须有 visibility 和 control
	•	如果这些不解决，AI 只能做低价值边角活，不能进入核心工作流  ￼

B. 所以 Gaia 的核心价值不是“回答问题”

而是两件事：

	•	intelligent orchestration
	•	continuous optimization

也就是既要把事做完，又要越做越好。 ￼

⸻

2) 系统总架构脑图
![img.png](img.png)

```
Layer 1: User Interaction Layer

接多模态输入，主动追问，输出 summary / visualization / recommendation，并且显示 confidence / uncertainty。 ￼

Layer 2: Orchestrator

核心 planner / coordinator。负责：
	•	intent understanding
	•	task decomposition
	•	workflow selection / creation
	•	routing to AI or human
	•	根据 outcome 动态重规划  ￼

Layer 3: Knowledge Management System

不是普通 RAG，而是组织级 knowledge layer：
	•	business / org context: including goals, org structure, and optimization
	•	team operational knowledge, such as institutional context, goals, roles, and functions, and guardrails
	•	runtime context, including agent registration info, project shared contexts based on recency, 
	•	冲突知识 resolution -> should esclate to human to resolve them
	•	provenance / uncertainty preservation  ￼

Layer 4: Optimization Layer

分两部分：
	•	Insights：看 usage pattern / quality / team collaboration pattern
	•	Reasoning + Action Generation：做 root-cause diagnosis，形成 hypothesis，决定修什么、怎么修、是否自动修  ￼

Layer 5: Task Executor

真正执行动作：
	•	API / MCP
	•	trigger agent
	•	update skill files
	•	route to third-party agents
	•	human approval
	•	rollback / audit / traceability  ￼
```

## 简单的task level orchestration v.s workflow orchestration
### task level 
先给你最短版结论：

你现在想的是

一个 feature：

	•	用户提问
	•	intent understanding
	•	走 summary / guidance / generation
	•	根据任务取不同 context
	•	LLM 输出
	•	结合 memory / personalization

这个没错。这个可以叫：

task-level orchestration

### Workflow orchestration
不仅做这一轮回答，而是：

workflow-level orchestration

也就是：

	•	这个任务后面还有什么步骤
	•	下一步该谁做
	•	如果结果不对怎么办
	•	如果 context 变了要不要重跑
	•	哪些中间结果还能复用
	•	哪些决策要让人 approve

⸻

最关键区别

Task level's设计的是：

“这次回答怎么生成”

Gaia 想做的是：

“这件事怎么一路被做完，并且越做越好”

```angular2html
earlier design is valid for a task-level AI assistant.
It covers intent understanding, context retrieval, generation policies, and personalization for producing a good response or recommendation.

But Gaia goes one level higher.
It is not only deciding how to answer the current request — it is managing how the entire workflow progresses over time.

So in Gaia, orchestration is not just routing a request to the right context or prompt policy.
It also needs to maintain persistent task state, track decisions and intermediate outputs, coordinate multiple agents or humans, and decide how to recover or adapt when conditions change.

That’s why the original design is a good foundation, but not sufficient by itself for a long-horizon operational system.

```

![img_1.png](img_1.png)

## Business use case

User: I’m taking over project Atlas Or “Okay, create a launch recovery plan.”

```
→ create Case
→ build initial context
→ generate summary
→ identify blockers
→ propose plan
→ detect conflict
→ update assumptions
→ replan
→ track progress
→ learn better policy
```

所以这里很需要的是:
✅ Challenge 1: long horizon

	•	项目不是一次理解完
	•	blocker 会变
	•	plan 会变
	•	需要持续更新

⸻

✅ Challenge 2: multi-agent

	•	retrieval agent
	•	planning agent
	•	analysis agent
	•	human reviewer

⸻

✅ Challenge 3: knowledge conflict

	•	doc vs ticket vs meeting notes 不一致

⸻

✅ Challenge 4: personalization

	•	PM vs engineer 看不同东西

⸻

✅ Challenge 5: optimization

	•	哪种 plan 更有效
	•	哪种 workflow rework 少

