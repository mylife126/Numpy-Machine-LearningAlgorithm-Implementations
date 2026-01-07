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



