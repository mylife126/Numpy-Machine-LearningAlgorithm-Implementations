## 1) 面试开场话术（30–60 秒：先控场再进入架构）

“Before jumping into modeling, I want to align on the objective and constraints.
For Atlassian enterprise search, the ranking system’s goal is to maximize session success — users find the right issue/page with minimal reformulation — under a strict latency budget and tenant privacy constraints.
I’ll assume the pipeline is retrieval → ranking, and ranking operates on a bounded candidate set.
I’ll design it in layers: (1) candidate & feature generation, (2) ranking model (v0→v2), (3) calibration/policy, (4) evaluation + monitoring.
I’ll pause after the high-level and we can deep dive into whichever part you prefer.”

你这段话的价值：你在告诉面试官 你知道 ranking 不是单模型，而是决策系统 + 约束下的 trade-off。

⸻

## 2) 整体系统架构（白板能画出来的版本）

2.1 High-level Pipeline
```
User query
→ Retrieval / Candidate Generation（BM25/ANN/graph/filters）
→ Feature Join（query-doc + doc-only + user/session + tenant priors）
→ Ranker（scoring）
→ Post-ranking Policy（diversity/dup removal/latency caps）
→ SERP / Answer grounding（human search + machine grounding 两种消费）

Ranking 目标：在 top-K 内把 “最可能成功的 doc” 排上去，同时保证可解释与稳定。
```
⸻

2.2 数据层（enterprise 的关键差异）:

	•	Doc types
	•	Jira issues：title, description, comments, assignee, status, labels, project, issue links
	•	Confluence pages：title, body, headings, space, last editor, created/updated
	•	Tenant isolation
	•	doc content 属于 tenant 私有
	•	训练可用：tenant 内日志 + tenant 内索引特征（embedding/bm25）+ 元信息（metadata）
	•	cross-tenant：只允许共享 非 UGC 的 backbone / 统计先验

⸻

## 3) 你要主动讲的“Ranking 的成功指标”（很 Principal）

“For ranking, I separate metrics into three layers:
(1) offline relevance proxies (Hit@K/MRR/NDCG),
(2) session-level success (reformulation rate, next-action),
(3) system guardrails (P95 latency, stability, diversity).
And the key is: enterprise search often lacks clean labels, so we rely on implicit feedback and careful de-noising.”

⸻

## 4) 建模方案（把我们之前聊的细节，嵌进 system design 框架）
### 重点features
你可以把特征分成 4 大类（面试“标准答案”），并给一些具体例子（面试官会点头的那种）。

A. Query–Doc 语义匹配特征（semantic / lexical）：

	•	emb_sim(q, d_title)、emb_sim(q, d_body)
	•	BM25 / keyword match：bm25(q, d) Best match 25 algorithm, for exact matching
	•	Query-term overlap：title 命中率、body 命中率
	•	是否命中关键字段（比如 “roadmap” 出现在 doc title）

这类特征决定“像不像”，但不等于“有用”。

⸻

B. Doc 质量与结构特征（document quality / structure）

Confluence：

	•	freshness_days（最近更新时间）
	•	is_official_page（模板/空间/标签推断）
	•	page_depth（层级位置）
	•	view_count_30d、edit_count_30d

Jira：

	•	issue_type（bug/epic/task）
	•	status（open/done）
	•	priority
	•	comment_count, watcher_count
	•	link_graph_degree（被多少 issue 引用）

这类特征在 enterprise 排名里非常值钱：决定“权威”和“可用性”。

⸻

C. User/Org/Permission 上下文特征（contextual / personalization）:

	•	same_team(u, doc_owner)（组织距离）
	•	org_graph_distance(u, doc_owner)（同组/跨组/跨 org）
	•	recently_interacted(u, project/space)（最近活跃空间/项目）
	•	permission_level（能否访问、访问成本）
	•	language/region（多语言公司）

很多“最有用”的文档，不是最像，而是离你最近、你最可能要用。

⸻

D. Retrieval & position 特征（候选来源与展示）:

	•	retriever_source（BM25 vs ANN vs expansion）
	•	candidate_rank_from_retriever（召回阶段的名次）
	•	snippet_quality（摘要长度/命中）
	•	position（最终展示位置，做 debias 用）

Pointwise 模型如果不处理 position bias，会学到“排前的就是好”。

#### 建模流程

下面是你能“从 v0 讲到 v2”的版本，面试官随时要求 dive deep，你就按层展开。

⸻

4.1 v0：Heuristic / Rule-based re-rank（快速上线）

适用：冷启动 tenant / 数据稀疏 / 先保稳定

	•	Score = α * BM25 + β * embedding_cosine + γ * freshness + δ * doc_popularity
	•	Jira 特化：
	•	query 命中 issue key 直接 boost
	•	status=Done 降权（用户通常更关心 open/blocking）
	•	Confluence 特化：
	•	title match boost
	•	最近编辑 boost

优点：可控、低风险
缺点：天花板低、难处理语义与 heterogeneity

⸻

4.2 v1：Pointwise（Regression / Classification Ranking）——最适合 first ML 版本

4.2.1 输入 / 输出

	•	Input：(query, doc, context)
	•	Output：score s(q,d)（越大越相关/越可能成功）

4.2.2 Features（你要像 Unity 那样讲出“详细 feature map”）

我按三类给你列全：universal / tenant-specific / doc-type specific

A) Query–Doc matching features（核心）

	•	lexical：
	•	BM25 score、term overlap、title overlap
	•	semantic：
	•	query embedding · doc embedding cosine
	•	cross-encoder score（若 latency 允许只在 top-N 用）
	•	relation aware：
	•	“blocking/depends” query 时：issue-link match features（link type count）

B) Doc-only features（质量与可用性）

	•	Confluence：
	•	page length、heading count、last updated recency、space importance
	•	Jira：
	•	status、priority、assignee exists、#comments、#links、created/updated、open duration

C) User / session features（ambiguity breaker，privacy-safe）

	•	last clicked doc type（jira vs conf）
	•	session reformulation count
	•	recent entities / projects（来自 QU 的 constraint extraction 输出，而不是 raw content）

D) Tenant priors（解决 heterogeneity 的第一块）

	•	tenant-level doc-type click prior（Jira-heavy vs Conf-heavy）
	•	tenant-level “success action prior”（某些 tenant 更常 copy link / create ticket）

⸻

4.2.3 Label 怎么做（我们之前讨论的三种都放进来）

enterprise search 没有明确 relevance label，所以用隐式信号构造：

Iteration 方案 A：Binary（快但粗）

	•	y=1：long click 或 next action
	•	y=0：no click 或 short click + return-to-SERP

方案 B：Graded relevance（面试推荐）
把 usefulness 做成等级（0/1/2/3），例如：

	•	3：next_action（copy/share/comment/create ticket）
	•	2：long dwell（>T）
	•	1：click + medium dwell
	•	0：no click 或 quick return

训练 Multihead classification 类似于ctr guide cvr的方案：

	•	ordinal classification

怎么用输出做 ranking？：
```
	•	如果是多分类输出 P(y=k)：最终分数用期望值
               score = Σ k * P(y=k)
这是标准做法，k 的取值不是“随便拍脑袋”，你可以：
	•	固定为 0/1/2/3（简单）
	•	或用业务价值映射（比如 next_action 的权重更高，但 cap，避免极端）类似于ads系统里的auction
```

方案 C：Continuous utility regression（工程味）
	•	y∈[0,1]：把 click / dwell / next_action 融合成一个连续分数
这能让优化目标更平滑、线上更稳定

⸻

4.2.4 dwell threshold T 怎么选（你之前卡的点）

你可以这样讲（很像 Principal）：

“I would pick T based on dwell-time distribution.
In practice, dwell time is heavy-tailed and often shows a ‘bounce’ mode (very short) and an ‘engaged’ mode (longer).
I’d log-transform dwell, fit a 2-component mixture (or simply pick the valley between the two modes) as T.
Then I validate T by correlating with stronger actions (copy/share/comment).”

⸻

4.3 v2：Pairwise / Listwise（当你要强调“更强排序”）

Pairwise（更抗噪）

训练样本：同一 query 下，(d_pos, d_neg)

	•	pos：被 long dwell/next_action 的 doc
	•	neg：曝光但未点击/快速返回的 doc
loss：hinge / logistic pairwise

优点：对 click bias 更鲁棒
缺点：采样与构造更复杂

Listwise（最强，但工程成本高）

对一个 query 的 doc list 直接优化 NDCG / ListNet / LambdaRank 类目标

面试策略：你可以说
	•	v1 用 pointwise 快速落地
	•	v2 在 head tenants + 数据充足时上 LambdaRank / listwise

⸻

4.4 Multi-task（你可以把“enterprise 很香”讲出来）

这是你之前的强项，你可以这样讲：

Heads：

	1.	P(click)
	2.	P(long_dwell)
	3.	P(next_action)
	4.	optional：dwell_time_regression

最终 ranking score（policy layer）：

	•	score = w1*P(click)+w2*P(long_dwell)+w3*P(next_action)
权重 w 通过：

	•	offline grid search（maximize NDCG/Success proxy）
	•	或 online bandit（maximize session success - cost）

为什么 multi-task 很适合 enterprise：

	•	每个信号噪声大，但组合能稳
	•	不同 tenant 信号强弱不同（有的 tenant 很少 share link），multi-task 更容易适配

⸻

5) Tenant heterogeneity：ranker 怎么做 “shared backbone + tenant head”

你在 ranking 这里也能用同一套：

	•	Shared backbone 学 universal patterns：
	•	query-doc semantic matching
	•	doc quality priors
	•	Tenant head / adapter 学：
	•	tenant jargon
	•	tenant-specific action propensity（比如 copy link 更常见）
	•	tenant-specific doc-type preference

训练方式：

	•	head tenant：full finetune / adapter
	•	tail tenant：只学轻量 head + 用 tenant priors warm start

⸻

6) Evaluation（ranking 的版本，和 QU 一样要讲 trade-off）

Offline

	•	Hit@K / MRR / NDCG（graded label）
	•	Slicing：
	•	doc type（Jira vs Conf）
	•	query length（短 query / 长 query）
	•	tenant size（head vs tail）
	•	action type（next_action 多的 tenant vs 少的）

Online

	•	Session success rate（next_action / long dwell）
	•	Reformulation rate ↓
	•	Time-to-success ↓
	•	Guardrails：
	•	P95 latency
	•	result stability（避免排序抖动）
	•	diversity（避免全是同一 space / 同一 project）

⸻

7) 面试结束前“收口话术”（非常关键）

“So overall, I’d start with a v0 heuristic ranker for stability, move to a pointwise utility model with graded labels, and then evolve to multi-task and pairwise/listwise when we have enough clean implicit signals.
The enterprise-specific part is handling privacy and heterogeneity, so I’d use a shared backbone plus tenant adapters, and evaluate everything in terms of session success per latency budget.”

⸻

8) 你可以怎么“控节奏”（你想要的方式）

你说得对：先 10 分钟讲完 workflow，再问想 deep dive 哪块。

你可以这样做：

“I’ve described the end-to-end ranking system.
Would you like to dive deeper into (A) labeling strategy, (B) model choice (pointwise vs pairwise vs multi-task), or (C) tenant adaptation and evaluation trade-offs?”

这句话一说，对面会觉得你很 senior。