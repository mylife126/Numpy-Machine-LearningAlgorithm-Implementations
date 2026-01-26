## 问题： 如何增加recall？利用llm orchestration来实现 如何训练呢

好的。我用你刚才那组追问（Formulation / Training signal / Cold start / Failure mode）给你一套**“Principal级别、可直接背诵”的回答**。我会默认你用 LLM 当 planner，但会把它工程化、可评估、可回滚。

⸻

✅ Answer (Principal-style): Query → Intermediate Anchor Selection

0) Frame the problem

“In Atlassian enterprise search, the key challenge is semantic gap + tenant heterogeneity + privacy. So I treat ‘should we use an intermediate anchor / multi-hop?’ as an expected value decision: does an extra hop increase the probability of retrieving a useful doc enough to justify extra latency/cost.”

⸻

1️⃣ Formulation: how to decide “need anchor?”

I’d model it as a query difficulty / routing decision with three outputs, not just yes/no:

	•	Direct: query is self-contained → single-hop retrieval
	•	Anchor-first: query likely needs a canonical page/entity as anchor (e.g., roadmap/epic/KB page)
	•	Graph-first: query needs entity resolution (manager/team/time window) via org graph or metadata

Implementation-wise (V1 → V2):

V1 (fast, robust):

	•	A lightweight classifier (or rules + classifier) outputs {Direct, Anchor-first, Graph-first} + confidence.
	•	Features: query length, presence of relational operators (“blocking”, “owned by”, “related to”), temporal phrases, entity mentions, plus optional user context.

V2 (stronger):

	•	Use an LLM planner to produce a structured plan (JSON), but bounded:
	•	route_candidates: [Jira, Confluence, Graph]
	•	need_anchor: true/false
	•	anchor_type: roadmap|epic|project|kb|person
	•	constraints: {time, owner, status, relation}
	•	max_hops: <=2

The key is: LLM outputs a plan, not final answers.

⸻

2️⃣ Training signal: where do labels come from under privacy?

I don’t rely on human labels. I use implicit feedback + self-supervision from logs.

Define a “success” proxy at session level

For each query session, define success signals (weak → strong):

	•	click (weak)
	•	long dwell / long click (stronger)
	•	next_action (share/copy/link/cite/create ticket/comment) (strong)
	•	query reformulation count decreases (strong)

Create labels for “need anchor”

We can infer whether a query benefits from anchor using counterfactual replay / path analysis:

Labeling idea (practical):

	•	If a session shows:
	•	multiple reformulations, or
	•	low hit@K in initial retrieval, and then
	•	success happens after user clicks a Confluence “hub” doc (roadmap/epic page) before landing on target tickets
→ label as Anchor-first.

Similarly:

	•	if success requires resolving manager/time/team via filters or metadata interactions → label as Graph-first.
	•	if single-hop retrieval already yields fast success (high MRR, low reformulation) → Direct.

This is privacy-safe because we only need doc IDs + event types, not reading content.

Also: LLM-as-teacher for bootstrapping

For cold start and ambiguity, I’d use a small number of sanitized synthetic prompts (no tenant content) to generate pseudo-labels for query patterns (“blocking”, “roadmap”, “what did my manager do last week”) to seed the classifier.

⸻

3️⃣ Cold start: new tenant with sparse logs

I’d do a staged rollout:

V0

	•	Default to Direct with strict guardrails
	•	Enable Anchor-first only for high-confidence patterns (roadmap/epic/KB keywords, relational operators)
	•	Use global backbone model trained across tenants on non-UGC / safe features (query text patterns + generic behaviors), and keep tenant-specific adaptation minimal.

V1

	•	Start collecting tenant-local implicit signals (no content exposure)
	•	Enable bandit-style exploration: occasionally try anchor-first for borderline queries, within a latency budget, to learn.

V2

	•	Tenant-aware light adaptation:
	•	shared encoder/backbone
	•	lightweight tenant adapter / head (LoRA or small MLP) trained on tenant-local logs
This handles terminology differences without training a full model per tenant.

⸻

4️⃣ Failure mode: when multi-hop hurts & mitigation

Concrete failure:

	•	Query is actually specific enough for direct retrieval (e.g., ticket title-ish), but planner triggers anchor-first:
	•	adds 1–2 hops
	•	latency increases
	•	and anchor retrieval may “overshadow” and drift away, reducing recall for the actual ticket

Detection (offline + online):

	•	offline: compare Direct vs Anchor-first on:
	•	MRR@K / hit@K for historically clicked doc IDs
	•	extra hop cost: retrieval calls, tokens, latency
	•	online: monitor:
	•	p95 latency
	•	reformulation rate
	•	abandonment / zero-click sessions
	•	“hop1 low yield”: if hop1 returns low-confidence candidates frequently

Mitigation (guardrails):

	•	Budgeted planning: max_hops=2, hard latency cap
	•	Quality gate: only proceed to hop2 if hop1 yields a strong anchor (e.g., top1 score > threshold, or high agreement between lexical + embedding)
	•	Fallback: if hop1 fails or costs too much → revert to Direct retrieval in same request
	•	Cache: cache common anchors (roadmap pages) per tenant to reduce latency

⸻

How I’d close it (tie back to system)

“So QU improves recall primarily by:
(1) choosing the right retrieval space (routing),
(2) extracting constraints to reduce noise, and
(3) deciding when to restructure search via anchors (planning).
Ranking remains downstream, but it benefits because candidate quality and diversity are much better.”

⸻

如果你愿意，我可以把这套答案再压缩成一个 90 秒版本（面试最常用），以及给你一个结构化 JSON plan 示例，让你在讲“LLM planner”时更像真正在做 production。

## 问题二 设计好了第一个query intent understanding后你怎么设计routing 的policy

- “对于刚才的问题针对于第一步 query intent understanding那部分，你问了一个问题关于routing policy的制定和验证， 你距离说如果jira是0.65， confluence 0.55 该怎么办 是multirouts还是single。 你也给我一个好的回答吧”

好的。我给你一个Principal级别的回答：既回答“0.65 vs 0.55 怎么选”，也把policy 如何制定 + 如何验证讲完整，而且不把自己绕进“只能上线才能评估”的坑里。

⸻

✅ Principal Answer: Routing policy（single vs multi-route）

0) 先把问题定义清楚

“Routing policy 的目标不是最大化分类准确率，而是最大化 端到端效用：在给定 latency/cost budget 下，提高成功检索概率（hit / long click / next action），同时控制无谓的双路开销。”

所以我不会只看 P(Jira) / P(Confluence)，我会做一个budgeted decision。

⸻

1) 一个清晰可落地的决策规则（你可以直接说）

假设模型输出：
	•	pJ = 0.65
	•	pC = 0.55

我会用 两层 gating：

Layer A: 绝对置信度门槛（confidence threshold）

每个路由都有自己的阈值（通过 offline tuning 得到）：
	•	tJ for Jira route
	•	tC for Confluence route

如果 pJ >= tJ，说明 Jira route “值得做”；同理 Confluence。

这一步解决的是：模型很不确定时不要乱开路由。

Layer B: 相对差距门槛（margin / dominance）

如果两个都过阈值，还要看差距 Δ = |pJ - pC|：
	•	如果 Δ >= m（差距明显）
→ single-route 选更高的那个（减少成本）
	•	如果 Δ < m（差距不明显）
→ multi-route（cascade 优先，必要时 parallel）

在你给的例子：
	•	两个都可能过阈值，但 Δ = 0.10 属于“边界情况”
	•	这时我会倾向 cascade：先跑 Jira（高的一侧），如果早停条件不满足，再跑 Confluence。

⸻

2) 为什么优先 cascade 而不是 parallel？

Principal 的点在这里：把成本变成可控的
	•	Cascade：先走便宜/概率高的路 → 若命中就 early stop
	•	Parallel：确定更稳，但 latency/cost 是固定翻倍

所以默认策略：
	1.	cascade
	2.	只有当 query 类型显示“强 multi-source”（比如 roadmap / blocking / “what did X do”）或高价值场景，才 parallel。

⸻

3) policy 怎么“验证”？（你刚才卡住的关键点）

你可以这样回答：policy 的验证分三层：routing、retrieval、end-to-end。

A) Offline：policy 本身的“决策质量”

用日志回放（不需要读内容）：

对每个 query session，我们可以构造：
	•	单路：只跑 Jira 或只跑 Confluence 的模拟结果（用历史索引快照 + topK 结果 id）
	•	双路：模拟 cascade / parallel 的 topK

定义一个 utility（可以是你前面讲的 graded relevance / 连续目标）：
	•	click / long click / next_action → utility 更高
	•	no click / bounce / reformulation → utility 更低

然后对 policy 做对比：
	•	E[utility] - λ * latency_cost 最大的 policy 为最佳
（λ 是你对 latency 的成本权重，可以用业务目标定）

核心：不是评估 classifier accuracy，而是评估“选择哪条路”带来的 expected utility。

B) Shadow test（上线但不影响用户）

线上真实请求只返回单路结果，但后台同时跑另一条路：
	•	记录如果走双路，hit@K / MRR / utility 会不会更好
	•	同时记录额外 latency/cost

这一步能在不冒风险下验证：
“multi-route 在哪些 query 上真的值”

C) A/B：只对高不确定区间开放 multi-route

当你 offline+shadow 有把握后：
	•	只对 Δ < m 的边界 query（比如 0.65 vs 0.55 这种）启用 multi-route
	•	这能把实验成本最小化，也更符合 Principal 的“风险控制”。

⸻

4) 你可以用一句话把 0.65 vs 0.55 讲得很“高级”

“If both routes pass the confidence gate but the margin is small, I treat it as an ambiguity region and use a budgeted cascade policy with early stopping, optimized for expected utility under latency constraints, validated via offline replay and shadow evaluation before widening to A/B.”

⸻

5) 面试官如果继续追问（你也有答案）

Q: early stopping 怎么做？
	•	Jira 路 topK 如果满足：
	•	top1 score 超阈值 或
	•	topK 的 aggregate confidence 足够高 或
	•	已出现 strong signals（比如 exact key match / strong semantic score）
→ 停止，不跑第二路
否则触发 Confluence 路。

Q: 不同 tenant 会不会阈值不同？

会。做法：
	•	全局阈值作为默认
	•	tenant 级别用 bandit/校准层做轻量调整（而不是全模型重训）

⸻

如果你想，我们可以把这个 routing policy 用一个小图总结成“decision tree”，你在面试里讲 30 秒就很清楚。

## How to design a system to improve search recall? 

下面我按你说的来：先评论，再给你一个更好的参考回答（你可以直接复用）。

⸻

✅ 即时评论（你这段哪里强、哪里会被扣分）

你做得好的点

	•	抓住主矛盾：tenant privacy → 不能靠人工标注 → 需要日志/弱监督/LLM teacher。
	•	有 模块化思维：intent → parsing/orchestration → multi-hop。
	•	讲了 shared backbone + tenant heads：能对上他们“heterogeneity + privacy”故事线。

你会被扣分的点（Principal 常见坑）
```
	1.	没有先对齐“QU 的目标”
你开头说 “QE improve recall”，但没有明确：
	•	QU 要优化的是 recall? precision? latency?
	•	还是 routing / constraint / planning 各自的 KPI？
Principal 面试官会想：你在讲“做很多事”，但不知道“成功定义”。
	2.	多处“train a model / leverage replay logs”太泛
面试官会追问：
	•	label 到底长什么样？
	•	用什么指标做 offline 评估？
	•	replay 怎么做 counterfactual（不然就是只能在线 A/B）？
你没落到“可执行的定义”。
	3.	你把 intent、parsing、planning 混在一起了
你说两大模块，但第二大模块里面又是 parsing、planning、rewriting、anchor…
Principal 需要更清晰的职责边界（否则像在堆 buzzwords）。
	4.	你过早把“tenant heads”放在 routing 上
routing（Jira vs Confluence）通常更像跨租户共性；tenant 异质性更明显的往往是：
	•	synonym / jargon（QU embedding、rewrite、entity linking）
	•	anchor selection / multi-hop价值判断
所以你这段会让人觉得你把最强武器用在了不最需要的地方。
```
⸻

✅ 一个更强、更像 Principal 的回答（可直接背）

我给你一个 2 分钟结构化版本，满足：目标清晰、模块边界清晰、隐私约束下可落地、讲清楚如何提升 recall。

```
Reference Answer (Principal-style)

“Great. I’ll frame Query Understanding as a pre-retrieval decision system that improves recall under tenant privacy by doing three things: routing, constraint extraction, and plan selection. The key is: we don’t rely on human labels; we use implicit feedback and bounded LLM planning.”

1) Objectives & success metrics
	•	Goal: increase successful sessions under latency budget, especially for vague queries.
	•	Offline KPIs:
	•	Recall proxies: hit@K / MRR@K against clicked doc IDs
	•	User effort: reformulation rate, time-to-first-success
	•	Cost: added hops, p95 latency, token cost

2) Module A — Intent/Routing (what corpus to search)
	•	Output: {Jira, Confluence, Both} + confidence.
	•	Start with cheap classifier (rules + small encoder).
	•	Why it helps recall: prevents “searching the wrong universe”.

Privacy-safe labels: from logs — if the successful clicked doc type is Jira vs Confluence, that’s the weak label.

3) Module B — Constraint Extraction (what filters to apply)
	•	Extract structured constraints:
	•	entity: project, team, person
	•	time window: last week / Q3
	•	relation operators: blocking, owned-by, mentioned-in
	•	Implementation:
	•	V1: lightweight model + rule patterns (temporal phrases, relation keywords)
	•	V2: bounded LLM that outputs JSON constraints (no free-form)

Why it helps recall: converts vague text into retrieval-ready structure (filters + normalized terms).

4) Module C — Planning / Anchor decision (direct vs multi-hop)
	•	Decision: direct retrieval vs anchor-first vs graph-first.
	•	Anchor-first example:
	•	query: “tickets blocking Q3 roadmap”
	•	step1: retrieve canonical “Q3 roadmap” page (Confluence)
	•	step2: use that page as anchor to retrieve related tickets (Jira) with relation=blocking constraint
	•	Key guardrails:
	•	max hops ≤ 2
	•	early-stop if hop1 yields high-confidence results
	•	fallback to direct retrieval if hop1 low yield

How to train without human labels:
Use session patterns to infer when multi-hop was needed:
	•	many reformulations + eventual success after clicking a hub page → anchor-first label
	•	success directly from query retrieval → direct label

5) Handling tenant heterogeneity
	•	For semantic understanding, use:
	•	shared backbone trained on global non-sensitive patterns + logs
	•	tenant lightweight adapters/heads for terminology shifts (synonyms, project names)
	•	We avoid seeing content by training on:
	•	query text + event sequences + doc IDs + metadata (doc type, timestamps)

Close:
“This QU layer improves recall by routing to the right corpus, turning text into constraints, and deciding when multi-hop/anchors are worth the cost — all validated via offline replay + shadow evaluation before A/B.”
```
⸻

你这段回答下一次怎么改（最小改动版）

你不用全部推倒重来，只要加三件事，你就“像 Principal”很多：
	1.	开头加一句：我的 QU 负责三件事：routing / constraints / planning
	2.	每个模块都给一个：输出是什么（结构化） + label 从哪里来（日志） + 为什么提升 recall
	3.	少说 “train a model”，多说：decision boundary + guardrail + evaluation

⸻

### Updated self answer
How will we answer this question again? 
So thanks for asking this question. 
Basically, you're asking how to design a system or Q-system to improve the recall of the retrieval or the search before the prior retrieval stage. 

So I think what I'm going to do is I'm going to frame the query understanding as a decision system that improves the recall on the tenant privacy by doing three major things. 
One is the routing design, policy design. The other is the semantic constraint extraction for the retrievals filtering, and then the planning selection or the orchestration. 
The key is we do not really rely on the human labels, but we actually use the implicit feedbacks for building such systems. 

First of all, what we need to design is actually the objectives and the success metrics. I think our goal is to increase the recall, which further increases successful session under the latency budget, especially for those very vague queries. 
So I think very important KPIs there are. For recall proxies, we can do HIT@K and MRR@K against the conterfactual replay, the clicked docs. 

And then for the actual user experiences-wise, we can track the reformulation rate, whether based on the new system we built, customers are actually reducing their reformulation. 
And then we also need to monitor the P95 latency. The more complicated the system is, the more latency it will be. So we want to find a balance between the actual HIT rate versus the latency. 
So that's the key KPIs that we need to measure, and that's the key KPIs that can be mapped to evaluate our objective. 

So first of all, I will talk about the module A, which is the intent or routing system, which basically is to understand which corpus to be searched. 
I assume that in Atlassian, we will have three different routes. One is JIRA only, second one is CONFERENCE only, the other is both. 
And then the model is nothing but. We start with a simple classifier. We classify whether it's going to be a JIRA CONFERENCE or both. 
So it's actually multi-class classification, and it will come with a model confidence score. 
Now, how do we do the privacy-safe labels? Basically, we still train from logs. Basically, we can see from the previous replay logs, given the search query, what's actually the doc type being actually clicked, or even better, being long-timely developed on. 
Then this routing system will do nothing but understand whether this is going to be a single route or it can be a cascade route. 
And then this policy can be also designed on a rule-based algorithm by analyzing the trade-off between latency and then the actual hit rate to judge whether this new policy using the model output is actually accurate. 

And then the module B is nothing but the constraint extraction. In this module, it does nothing but extract the structure constraints from the customer input queries to project the constraints such as project name, team name, person name, time windows, relationships such as blocking, owned by, mentioned in. 
To do this, what we can do is to build a rule-based pattern by identifying the key major patterns such that we can use entity recognition, or for v2, we can do a small bound to the LLM that outputs the JSON constraints by tuning a pre-trained lightweight model to reduce the latency. 
So what it helps to recall, it actually converts the vague text into a retrieval-ready structure with filters and normalized terms, which will further help with the post-retrieval stage. 

And now it comes to module C. It takes in the information already gained from module A and B to plan, make an orchestration to make an anchor decision. 
First of all, shall we need to reformulate the queries into a better representation? For example, if the customer searches something really disordered, a query such as Q3 roadmap, Jonathan, Wednesday. 
And if the model, and then we can reformulate it into a better representation for the queries, because each corpus embedded in space is trained on different representation of the syntax, then we can reformulate it into a better way. 
Second of all, decision on directed retrieval versus the direct search, anchor-first retrieval or graph-first retrieval, which is essentially the directed retrieval versus a multi-hop retrieval. 

For anchor-first retrieval, for example, how customer can search ```tickets blocking Q3 roadmap```, then what we can do is step one, retrieve the intermediate Q3 roadmap conference page as the actual anchor context to enrich the anchor context and to retrieve further on the GR tickets, 
applying the blocking constraint. On the other hand, if the query is simple enough, then we can directly do the query to tickets retrieval directly, 
but the key guardrails needs to be the hop steps, whether it's going to be two or three, and it's also going to be measured by the actual tradeoff between the hit rate MRR versus the actual latency. 
So the key idea is to balance out the latency and the actual performance. 

And how to train this is actually simple. We will still use the session pattern to infer whether it's going to be hop, multi-hop needed, or single hop needed. For example, if we see in the session, this customer has suffered with multiple reformulations, multiple clicks, and finally went to the actual tickets, 
then it's telling me that this needs to be a multiple anchor-first solution. On the other hand, if we see the customer has less reformulation frustration, it directly leads to the actual document that we know, based on this query, it can be a direct retrieval stage. 

So finally, I'll talk about how we handle the tenants heterogeneity issue. So basically, for semantic understanding, what we can do is for each stage, we train a similar model structure with the shared backbone on the global non-sensitive patterns and logs. 
And each tenant that adapters have can be also connected with the shared backbones. And this layer learns nothing but the tenant universal, a tenant-specific representation to either make classification or make a regression, things like that. But as we move to the LLM era, especially for the third one, which is the orchestration part, we can actually train or actually directly use an LLM to basically train specifically on the prompt part to be adapted to each tenant representation for a better performance.

### Condensed Short cut

```aiignore
🔹 Principal-level Answer (Condensed)

“I frame Query Understanding as a pre-retrieval decision system whose goal is to improve recall for vague enterprise queries under tenant privacy and latency constraints.”

1. Objectives & metrics
	•	Goal: increase successful sessions (finding the right doc with less reformulation).
	•	Offline:
	•	Recall proxies: Hit@K / MRR@K against clicked doc IDs
	•	User friction: reformulation rate, time-to-first-success
	•	Guardrails: p95 latency, added hops, cost

⸻

2. Module A — Routing (what corpus)
	•	Output: {Jira, Confluence} scores, not a hard decision
	•	Model: cheap classifier (shared backbone)
	•	Policy layer decides:
	•	single-route
	•	cascade
	•	parallel (rare, cost-bounded)

Labels (privacy-safe):
	•	From logs: final successful doc type + dwell / next-action
	•	No content inspection needed

Offline evaluation:
	•	Counterfactual replay:
“If we had routed only Jira vs both, would the clicked doc still be reachable @K?”

⸻

3. Module B — Constraint extraction (what filters)
	•	Extract structured constraints:
	•	entities (project, person)
	•	time windows
	•	relations (blocking, owned-by)
	•	V1: rules + NER
	•	V2: bounded LLM → JSON schema

Why it helps recall:
Turns vague text into retrieval-ready structure, reducing semantic drift.

⸻

4. Module C — Planning (direct vs multi-hop)
	•	Decision: direct retrieval vs anchor-first vs graph-first
	•	Example:
	•	“tickets blocking Q3 roadmap”
	•	hop1: retrieve canonical roadmap page
	•	hop2: retrieve Jira tickets linked as blocking

Training signal (no human labels):
	•	Session patterns:
	•	many reformulations → anchor-first beneficial
	•	direct success → single-hop

Guardrails:
	•	max hops ≤ 2
	•	early stop if hop1 confidence high
	•	fallback to direct search

⸻

5. Tenant heterogeneity
	•	Shared backbone for general language patterns
	•	Tenant adapters mainly on:
	•	semantic understanding
	•	constraint interpretation
	•	Routing remains mostly global

“This QU layer improves recall by deciding where to search, how to constrain, and when multi-hop is worth the cost, all learned from implicit signals under privacy constraints.”

```

## 3 个module里哪个更适合Agentic 哪个更适合 Shared backbone model？
```aiignore
这个问题问得非常好，而且非常“Principal”。你现在卡住的点，其实正是 Central AI / Agentic Search 面试里最容易混淆、但最关键的一层。

我分 5 个层次把它彻底理清，你看完之后，这三个问题都会同时解掉：
	•	Backbone 到底用在哪
	•	LLM 出现后 backbone 还要不要
	•	deterministic model vs agentic 的本质区别

⸻

一、先给你一个结论版（先稳住）

LLM ≠ 不需要 backbone
Agentic ≠ 一定是 LLM
Backbone 是“长期稳定语义”，LLM 是“即时推理与规划”

在 Atlassian 这种 enterprise + tenant heterogeneity + latency 约束 的场景里：
	•	Backbone 模型 = 规模化、可控、低成本的“感知层”
	•	LLM = 高不确定性 query 的“规划 / 推理层”
	•	Agentic 的本质是：动态决策 + 多步执行 + 反馈回路，不是“是不是 LLM”

⸻

二、把 QU 的「三部曲」重新映射到技术栈

你现在的 QU 三部曲是完全对的：

1️⃣ Intent / Routing
2️⃣ Semantic Constraint Extraction
3️⃣ Planning / Orchestration

关键是：每一层不需要用同一种技术。

🔹 Step 1：Intent / Routing（最适合 backbone）

这是 backbone 最重要、最合理的地方

为什么？
	•	输入简单：query text + 少量 metadata
	•	输出稳定：Jira / Confluence / Both
	•	QPS 高，必须便宜
	•	tenant heterogeneity 主要体现在：
	•	vocabulary
	•	query patterns
	•	不需要“推理”，只需要“识别”

典型实现
	•	Shared encoder backbone（small transformer / bi-encoder）
	•		•	tenant adapter / head
	•	输出 score，不做决策

👉 这里用 LLM 是浪费
👉 这里不用 backbone 才奇怪

⸻

🔹 Step 2：Semantic Constraint Extraction（backbone + LLM 混合）

这一层是 heterogeneity 开始变明显 的地方。

能用 backbone 的部分
	•	NER（人名、项目名、时间）
	•	pattern-like 关系（blocking / owned-by）
	•	历史频繁结构

👉 这些可以：
	•	backbone encoder + CRF
	•	或小 encoder + rule hybrid

LLM 更适合的部分
	•	模糊表达
	•	组合关系
	•	query 很短但含义复杂

例如：

“things that might slow down the Q3 push”

这种 弱结构、强语义 的内容，LLM 更稳。

👉 结论：
	•	backbone = default path
	•	LLM = fallback / augmentation

⸻

🔹 Step 3：Planning / Orchestration（这里才是 agentic）

你刚才问的关键问题在这里：

“如果我用 deterministic model，这还算 agentic 吗？”

答案是：可以算，但不 fully agentic

我们先定义 agentic 的三个必要条件：

条件	是否必须 LLM
动态决策（run-time choose path）	❌
多步执行（multi-hop）	❌
基于中间结果调整计划	❌
开放式 reasoning / decomposition	✅


⸻

三、deterministic planning vs LLM planning 的本质差异

🧠 Deterministic Planner（仍然是 agent-like）

例如：

IF (route == both AND constraint contains "blocking")
→ anchor-first
ELSE
→ direct

特点：
	•	可控
	•	可 debug
	•	latency 低
	•	很适合 80% query

👉 这是 agentic-lite

⸻

🤖 LLM-based Planner（full agentic）

例如：

{
  "step1": "retrieve roadmap doc",
  "step2": "retrieve tickets linked as blocking",
  "confidence": 0.78
}

特点：
	•	处理 long-tail
	•	处理 unseen composition
	•	可 reasoning + explanation
	•	成本高，需 guardrail

👉 这是 agentic-heavy

⸻

🟡 Atlassian 最合理的真实形态（非常重要）

Hybrid Planner

default → deterministic planner
uncertain / high-entropy → LLM planner

这正是 Central AI 在 search / agentic search 的主流设计方向。

⸻

四、那 backbone 在 orchestration 里还有用吗？

有，但不是“直接做 planning”

Backbone 在 orchestration 的角色是：

1️⃣ 提供低成本 uncertainty signals
	•	query embedding entropy
	•	similarity dispersion
	•	route score margin

👉 这些 signal 决定：
	•	要不要启用 LLM planner
	•	要不要 multi-hop

⸻

2️⃣ 为 LLM 提供 structured input

而不是把 raw query 丢给 LLM

例如：

{
  "route_scores": {"jira": 0.62, "conf": 0.58},
  "constraints": ["blocking", "Q3"],
  "history": "3 reformulations"
}

👉 这是 enterprise LLM 的关键
👉 LLM 不是感知层，是决策层

⸻

五、回到你最关键的三个问题（逐一回答）

❓1. backbone 用在哪？
	•	Routing（最重要）
	•	Constraint extraction（部分）
	•	uncertainty estimation（planner gating）

⸻

❓2. 有了 LLM 还需要 backbone 吗？

必须要。否则：
	•	成本爆炸
	•	latency 不可控
	•	tenant 难 scale
	•	无法解释 & debug

👉 LLM 不能承担 always-on perception layer

⸻

❓3. 不用 LLM 的 planner 算 agentic 吗？

算 agentic-lite
	•	agentic 的本质：动态决策 + 多步执行
	•	LLM 只是实现方式之一
	•	Atlassian 实际上 一定是 hybrid

⸻

六、为下一个问题铺路（你下一问会更强）

你下一问马上就会是：

“Which QU decision gives the most marginal recall gain per millisecond?”

现在你已经具备回答它的结构了：
	•	Routing：便宜，但 recall 提升有限
	•	Constraint extraction：中等成本，中等收益
	•	Planning（是否 multi-hop）：
👉 最高 recall gain / ms

👉 这就是下一轮我们要打穿的核心。

你已经完全在 Principal MLE 的轨道上了。
```

### 那么我想到了一个case，你说的一点很重要 对于最后的orchestration部分，传统model局限于对于tail的未知性，例如query就是一个 Q3， 这个就很难判断，可能anchor or direct的model 分数都很低，但是llm的好处是 可以动态的决定输入的信息， 面对Q3 这种极端vague的query 可以引入该客户过往24小时内的输入做分析 然后再做query rewrite 然后再做multihop or single hop， 是这个意思么？
```aiignore
对，你这个 case 抓得非常准，而且这是一个“面试官会点头”的理解。我帮你把这件事精确化、工程化一下，避免听起来像“LLM 很聪明所以都交给它”。

⸻

一句话确认你的理解（先给结论）

是的 👍
你描述的正是 LLM 在 orchestration / planning 阶段相对于传统模型的“不可替代价值”：

当 query 极端模糊（如 "Q3"）且当前 query 本身信息熵过低时，
LLM 可以在 run-time 主动拉取、组合、解释额外上下文（尤其是 session context），
再决定是 rewrite / multihop / direct，这是传统模型几乎做不到的。

⸻

为什么传统模型在这个 case 里会“天然失败”

我们把 "Q3" 这个 query 拆开看。

对 routing / planning model 来说：
	•	query embedding 几乎是：
	•	高度泛化
	•	无 domain anchor
	•	无 action intent
	•	结果就是：
	•	direct retrieval score 低
	•	anchor-first score 也低
	•	model 只能给你一个 “我不知道”

👉 这不是模型没训练好，而是输入信息论上就不够

⸻

LLM 在这里到底“多做了什么”（不是魔法）

你说的这句话非常关键：

“LLM 可以动态决定输入的信息”

这是本质区别，我们拆成 三层能力。

⸻

1️⃣ LLM 能“决定要不要扩展上下文”（传统模型做不到）

对于 "Q3"，LLM 可以做的是：

我现在不够信息，我需要更多 signal

例如：
	•	最近 24h / 当前 session：
	•	用户刚看过：
	•	Q3 roadmap
	•	launch checklist
	•	或刚搜过：
	•	Q3 blocking issues
	•	Jonathan roadmap

👉 这是 context acquisition decision
传统模型只能“吃你给的 input”，LLM 可以说“我还要这些”。

⸻

2️⃣ LLM 能解释上下文，而不是简单拼接

这点非常重要，不然会被面试官 challenge。

LLM不是简单做：

"Q3" + last_queries

而是做 semantic interpretation：

“用户现在说 Q3，很可能是一个 shorthand，指代刚才 session 里的 Q3 roadmap 主题”

这一步包含：
	•	指代消解（anaphora）
	•	topic continuity
	•	implicit intent completion

👉 这是 reasoning over context，不是 feature engineering。

⸻

3️⃣ 基于“补全后的 query”，再做 planning 决策

补全后，LLM 现在面对的不是：

"Q3"

而是类似：

"tickets blocking Q3 roadmap launch"

这时候：
	•	LLM 可以判断：
	•	direct retrieval ❌（ticket 和 roadmap 语义距离远）
	•	anchor-first ✅
	•	于是给出 plan：
	1.	找 Q3 roadmap page
	2.	用 page embedding 找 blocking tickets

👉 这一步是 planning，不是 retrieval

⸻

用一句“非常像 Principal”的话怎么说（你可以直接用）

你在面试里可以这样总结这一点（这是高分回答）：

For extremely under-specified queries like “Q3”, the limitation is not model capacity but input entropy.
A deterministic planner or classifier has no reliable signal to decide between direct or anchor-first retrieval.
What LLM enables here is dynamic context acquisition and interpretation — it can decide to pull session-level signals, infer implicit intent continuity, rewrite the query into a retrieval-ready form, and only then choose between single-hop or multi-hop execution.
That’s why we scope LLM strictly to the orchestration layer, gated by uncertainty signals from cheaper models.

⸻

再帮你防一个“追问陷阱”

面试官可能会问：

“那是不是所有 query 都交给 LLM？”

你可以立刻回答（非常重要）：

No — this only triggers when the query entropy or route margin is below a threshold.
For the majority of queries, deterministic routing and planning are both cheaper and more stable.
LLM is reserved for cases where the marginal recall gain justifies the latency and cost.

👉 这句话直接把你和“LLM first”的人区分开。

⸻

最后一句评价你现在的状态（实话）

你现在已经不是在“理解 agentic search”，
你是在自己推导为什么 agentic search 在 enterprise search 里是必要的。

下一步我们可以顺着这个 case，直接回答你提到的那个 killer question：

Which QU decision gives the most marginal recall gain per millisecond, and how would you prove it?

那一问，是Principal 的分水岭。
```


