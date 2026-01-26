## 1) 面试开场话术（先把 retrieval 的目标讲清楚）

“I’ll treat sourcing as a semantic retrieval layer that returns a bounded high-recall candidate set for downstream rankers.
The objective is not top-1 accuracy, but recall@K per millisecond under tenant isolation.
In Atlassian, documents are heterogeneous: Jira issues are short/structured with graph edges, while Confluence pages are long/unstructured. So I’d design a multi-index retrieval with shared embedding backbone and doc-type-specific heads, plus tenant adapters when allowed.”

⸻

## 2) 系统架构（Post-QU → Retrieval）

Input：QU 输出的结构化信息

	•	route：Jira / Confluence / both
	•	constraints：project, assignee, time window, relation(“blocking”)
	•	plan：direct vs multi-hop anchor-first

Retrieval stack（通常是多路并行）

	1.	Lexical retrieval：BM25 / keyword / filters（高精确、低成本）
	2.	Semantic ANN：two-tower embedding → vector DB（高召回）
	3.	Graph retrieval（Jira 特有）：issue links / project graph / dependency expansion
	4.	Fusion & prune：把多路候选合并去重，出一个 bounded set（比如 500～2k）

Retrieval 的“Principal点”就在 fusion 和 boundedness：你要让下游 ranker 能吃得下。

⸻

## 3) Atlassian 的 two-tower 该怎么建（和 Amazon 的对应关系）

3.1 Tower 定义（Query tower vs Doc tower）

Query tower 输入

	•	query text（可能很短：Q3）
	•	QU 的 constraints（结构化 token：project=X, time=last_week, relation=blocking）
	•	optional：session context（最近点击 doc type / project）
	•	输出：e_q

Doc tower 输入（分 doc type）

Confluence doc

	•	title + headings + first N chars（长文要做 chunk 或 summary）
	•	metadata：space, last_updated, author
	•	输出：e_doc

Jira issue

	•	title + description（truncate） + key fields（status/priority/assignee/labels）
	•	graph-aware features：link types counts / parent epic id（可选）
	•	输出：e_issue

你可以说：doc tower 不是一个——至少是 shared backbone + doc-type adapters（Jira vs Conf），否则表示能力会被长文/短文冲突拖垮。

⸻

3.2 多索引（Multi-index）是 Atlassian retrieval 的核心差异

你在 Amazon often 是 “product index 一套”。Atlassian 更像：

	•	Confluence index（page/chunk）
	•	Jira index（issue）
	•	optional：People / Org index（manager queries）
	•	optional：Project/Space index

QU routing 决定 搜哪个 index，planning 决定 是否 multi-hop。

⸻

## 4) 训练数据怎么来（enterprise 的关键：全靠 implicit feedback）

你可以按三类样本构造，特别像你之前 ranking 的 label 思路，但 retrieval 要更强调“正例覆盖”和“hard negatives”。

4.1 正例（positive pairs）

对每个 query session：

	•	最强：发生 next_action 的 doc
	•	次强：long dwell doc
	•	弱：clicked doc

构造：
	•	(q, d_pos) 作为正例

4.2 负例（negatives）

Retrieval 训练里负例策略非常关键（Principal点）：

	•	In-batch negatives：同 batch 其它 doc
	•	Hard negatives from retrieval：BM25/旧 embedding topN 但没被点击 / quick return
	•	Same-tenant hard negatives（非常重要）：同一个 project/space 内相似 doc（否则模型学不到区分能力）

enterprise 里最怕“随机负例太容易”，训练出来的 embedding 只有粗粒度语义，召回可以但 precision 很差。

⸻

5) loss 怎么选（对标你熟的 two-tower）

5.1 InfoNCE / contrastive（最标准）

	•	score = cos(e_q, e_d)
	•	softmax over positives vs negatives

适合：训练稳定，召回效果强。

5.2 Pairwise hinge / logistic（也可）

	•	让 score(q, pos) > score(q, neg) + margin

适合：你已经有 hard negatives 时。

5.3 Multi-task retrieval（Atlassian 很香）

heads：
	•	click-retrieval head
	•	long-dwell head
	•	next-action head

让 embedding 同时服务不同强度的 “relevance” 定义，最后融合或用 strongest signal 训练。

⸻

6) Tenant 隔离下怎么做（你问的核心）

6.1 如果允许共享非UGC backbone

做法跟你之前 multi-head 一样：

	•	Shared backbone：学通用语言匹配能力（不看具体 tenant content）
	•	Tenant adapter/head：学 tenant jargon / doc format / action propensity

6.2 Tail tenant 数据稀疏怎么办

	•	先用 shared backbone + doc-type adapter
	•	tenant 只训练很轻的 bias / temperature / projection layer
	•	或者用 QU constraints + lexical retrieval 补 recall（hybrid）

你可以一句话总结：
“For tail tenants, hybrid retrieval is the safety net; for head tenants, tenant adapters unlock the next level.”

⸻

7) Confluence 长文怎么 embed（Atlassian 特有）

你要提前准备这一点，否则面试官会追问 “page 很长你怎么做？”

三种主流方案：

	1.	Chunking：把 page 分段 embed，检索 chunk，再聚合到 page
	2.	Structured fields：title/headings 单独 embed + body chunk
	3.	Offline summary embedding：先生成 page summary（privacy-safe within tenant），summary 做 embedding，chunk 做 recall 补充

面试推荐说：

	•	v0：title+headings embed（快）
	•	v1：chunk ANN + page aggregation（稳）
	•	v2：summary + chunk hybrid（最好）

⸻

8) Retrieval evaluation（跟 ranking 不一样：更偏 recall/cost）

Offline

	•	Recall@K / Hit@K（clicked / long-dwell / next-action doc 是否被召回）
	•	MRR@K（可选）
	•	Δrecall per ms：不同策略下的边际收益（你之前那题的延伸）

Online

	•	session success ↑
	•	reformulation ↓
	•	time-to-success ↓
	•	guardrails：
	•	P95 latency
	•	ANN QPS/cost
	•	multi-hop rate（过高说明 planner 太激进）

⸻

9) 你问“这不就和 ranking 很像吗？”——面试里的标准回答

你可以这样一句话把差别讲清楚：

“They are aligned but optimized for different objectives: retrieval maximizes recall under latency, ranking maximizes top-K usefulness.
Retrieval is embedding + ANN with contrastive learning and hard negatives; ranking is supervised utility prediction with richer features and calibration.”

⸻

10) 一段你可以直接背的“2分钟完整回答”

“I’d build semantic sourcing as a two-tower retrieval system with ANN, but Atlassian needs multi-index because Jira issues and Confluence pages have very different structures.
The query tower takes query text plus QU constraints; the doc tower is shared backbone with doc-type adapters, and optionally tenant adapters for head tenants.
Training uses implicit feedback: positives from next-action/long-dwell/click, and hard negatives mined from BM25 or prior retrieval within the same tenant/project. Loss is contrastive InfoNCE.
At serving time I run hybrid retrieval: BM25 + ANN + graph expansion for Jira, then fuse and prune into a bounded candidate set for rankers.
Evaluation focuses on recall@K per millisecond, and online I track session success, reformulation, time-to-success with P95 latency guardrails.”
