## QU for Atalassian
好，我们直接进 Query Understanding（QU）技术细节，专门针对 Atlassian（Jira/Confluence + 3P connectors + tenant/privacy）这个场景来讲：能怎么做、限制是什么、方法优劣、如何演进、怎么衡量、它和 retrieval/ranking 的关系。

⸻

0) 先把 QU 在 Atlassian 里“是什么”讲清楚

在电商里，QU 很多是把 query 映射到「商品属性」+「购买意图」。
在 Atlassian enterprise search 里，QU 更像是：

把一个“模糊的工作问题/任务”翻译成“可检索的结构化检索计划”
并且要适配：文档类型（Jira issue / Confluence page / 评论 / PR / Slack msg）+ 权限 + tenant 语言习惯 + 组织结构。

典型 query（你会经常遇到）：

	•	“Q3 roadmap”
	•	“oncall handoff”
	•	“postmortem payment outage”
	•	“OKR for search quality”
	•	“how to request access to prod”
	•	“tickets related to latency spike last week”
	•	“design doc for agentic search”

这些 query 的共同点：

	•	极短、缺少实体（哪个 team？哪个项目？）
	•	隐含上下文（我是谁、我在哪个 org、我最近在做啥）
	•	答案往往是多个文档组合（不是一个“商品”）

⸻

1) QU 在系统里到底贡献给谁？

你问得非常对：QU 是给 sourcing/retrieval 还是 ranker？

✅ 主贡献对象：Retrieval / Sourcing（上游）

QU 主要做三件事：

A. Query expansion（召回增强）

把 “Q3 roadmap” 变成可检索关键词/同义词/别名：

	•	“roadmap”, “planning”, “FY25 Q3”, “Q3 priorities”
	•	team/project 别名（如 “Robo”, “Central AI”, “Search quality”）
	•	文档类型/空间（Confluence space key / Jira project key）

B. Entity resolution（实体消歧）

把 query 里隐含实体补出来：

	•	“my manager” → org graph
	•	“payments outage” → service name / incident taxonomy
	•	“last week” → 时间范围

C. Vertical routing（检索路由）

决定去哪些索引查：

	•	Jira issues vs Confluence pages vs Slack vs Git
	•	先查高 precision 的 structured store（Jira）还是先查文本（Confluence）

✅ Ranker 是下游（但 QU 也会喂给 ranker）

你之前做的 ranker 是 QU 的下游之一。QU 输出的：

	•	intent / entity
	•	query rewrite
	•	filters（time, project, doc type）
	•	query embedding
都会成为 ranker 的 features 或候选集质量保障。

一句话：

QU 决定“你能不能把对的东西召回来”，Ranker 决定“召回里谁排前面”。

⸻

2) Atlassian QU 的限制/困难点（为什么比 Amazon 难）

2.1 tenant heterogeneity（语言/习惯差异）

同样是 “roadmap”：

	•	A 公司叫 “plan”
	•	B 公司叫 “FY planning”
	•	C 公司叫 “north star doc”
同义词和命名体系高度 tenant-specific。

2.2 privacy / observability 限制

你不能人工看 tenant 文档内容做规则；失败案例也难 replay。导致：
	•	不能靠人工维护一堆 synonym 表
	•	不能靠人工标注大量 query intent

2.3 冷启动 + 稀疏行为

很多 tenant 搜索量不大，且行为弱（不点、不分享），QU 的监督信号更稀疏。

2.4 文档形态复杂（半结构化 + 图结构）

Jira issue 有强结构（project/status/assignee/links）
Confluence 是长文本
Slack 是短对话
Git 是 code + PR
QU 必须能处理跨源异构。

⸻

3) QU 能怎么做：方法谱系（从传统到 LLM）

我给你一条完整“技术路线图”，每个方法都有优劣/适用场景。

⸻

方法 1：Lexical / Query rewriting（传统、稳、快）

做什么
	•	拼写纠错、分词、同义词扩展
	•	acronym expansion（SLA / RCA / SEV2）
	•	基于统计的 phrase mining（从文档/标题中挖高频短语）

优点
	•	低延迟、强可控、容易 debug
	•	对 Jira 这类结构化字段很好用（project key、status）

缺点
	•	语义泛化弱
	•	tenant-specific 词表维护困难（privacy + 成本）

适用
	•	V0/V1 baseline
	•	高 precision routing（如 Jira project key 检测）

⸻

方法 2：Embedding-based QU（语义召回友好）

做什么
	•	query embedding
	•	生成 query expansion candidates（用 embedding 找相近词/短语）
	•	近似做“概念对齐”，比如 “postmortem”≈“RCA”≈“incident review”

优点
	•	泛化强（跨表达方式）
	•	对短 query 很有效

缺点
	•	容易漂（semantic drift）
	•	难解释、难控（尤其在 enterprise）

适用
	•	提升 recall
	•	给 RAG / agentic search 做 grounding

⸻

方法 3：LLM-based QU（意图识别 + 结构化检索计划）

这是 enterprise QU 的“核心升级路线”。

3.1 LLM 做 intent + entity + constraints 抽取

把 query 解析成结构化 JSON（bounded output）：

{
  "intent": "find_document",
  "entities": ["Q3", "roadmap"],
  "doc_types": ["confluence_page", "jira_epic"],
  "time_range": null,
  "filters": {"project": null, "space": null},
  "rewrites": ["Q3 roadmap", "FY25 Q3 roadmap", "Q3 planning priorities"]
}

3.2 LLM 做 query decomposition

“tickets related to latency spike last week”：
	•	子查询1：latency spike + last week + incident/postmortem
	•	子查询2：Jira issues tagged latency/perf + last week
	•	子查询3：dashboards/alerts

优点
	•	解决“短+模糊+多意图”
	•	天然适合 agentic search（生成 search plan）
	•	可以做 bounded outputs，便于安全可控

缺点（Atlassian 场景最关键）
	•	需要防 hallucination：不能编 project key/space
	•	tenant-specific 术语仍是难点（需要 grounding/adapter）
	•	成本/延迟（不过他们说 2–3s budget，在线可用一部分）

适用
	•	V2/V3：真正把 QU 做成“检索计划生成器”
	•	支撑 agentic search

⸻

4) QU 的演进路线（为什么这么 evolve）

你要的不是“面试框架”，而是技术上“为什么要从 A→B”。

V0：规则 + lexical（保证不犯错）
	•	拼写纠错、停用词、关键词扩展
	•	project/space/status 规则识别
	•	目的：可控、低风险

V1：语义 embedding + hybrid retrieval（补 recall）
	•	用 embedding 做 query expansion & semantic recall
	•	与 BM25 / keyword 混合
	•	目的：解决表达差异

V2：LLM 结构化 QU（提升可解释可控的理解）
	•	LLM 输出 JSON（intent/entities/filters/rewrites）
	•	与 tenant metadata（org graph / project list）做 grounding 校验
	•	目的：把“理解”显式化，可 debug，可控

V3：Tenant-aware adaptation（解决 heterogeneity）
	•	共享 backbone（通用 QU 能力）
	•	tenant-specific adapter/head（术语、简称、内部命名）
	•	或检索增强：tenant glossary index（仅该 tenant 内）
	•	目的：个性化/异质性

V4：Agentic QU（复杂问题 → 多步检索）
	•	QU 不只是 rewrite，而是生成 “tool calls plan”
	•	搜索作为工具多轮调用，逐步 narrowing
	•	目的：回答复杂工作问题

⸻

5) QU 怎么衡量（很关键）

QU 的指标要分层，否则很难归因。

5.1 QU 本身的“解析质量”
	•	intent accuracy（若有弱标注）
	•	entity extraction precision/recall
	•	rewrite quality：query-to-doc recall@K 提升（用 offline replay）
	•	invalid filter rate：生成了不存在的 project/space 的比例（很关键）

5.2 对 retrieval 的提升
	•	recall@K（尤其是 high-value doc：被 share/cite 的）
	•	MRR / NDCG（可以在候选集层面看）
	•	coverage：能产生有效 rewrites/filters 的 query 占比

5.3 端到端业务指标（更贴 Atlassian）
	•	search success rate（是否发生 long click / next action）
	•	time-to-first-useful-click
	•	next_action rate（copy/cite/create issue）
	•	agentic：task completion rate、tool-call efficiency

5.4 tenant heterogeneity 下的指标设计
	•	按 tenant size 分 bucket（large/medium/small）
	•	按 data source 分 bucket（1P vs 3P）
	•	看 uplift 分布而不是平均值（防大 tenant 淹没）

⸻

6) 回答你最后一句：Ranker 是 QU 的下游吗？

✅ 是的，ranker 是 QU 的下游之一。
更准确地说：
	•	QU → retrieval candidates 质量（最重要）
	•	QU → features for ranker（次重要）
	•	QU → agentic planner 的输入（未来更重要）

如果 QU 做不好，你的 ranker 再强也只能在“错的候选集”里排序。


## 深入问题1

- “有几个问题：
1. 你说Atlassian的 tenant heterogeneity问题，是不是可以理解为 在amazon 我用一个model 做的semantic retrieval就可以解决所有的 query to product的recall。 但是在Atlassian它有不同的公司，terminology不一样，一个model就很难解决多个syntax的问题？ 所以optimally， 如果可以做tenant aware 的multihead就会更好？

2. 然后在亚麻QU 就很简单， 例如map 一个 query to catalog， product type， attribute等等， 最主要的是Q2Asin 来做推荐的结合ANN，那么 Atlassian 也是类似的么？ 然后又该怎么建模呢，数据怎么来的都是？ 我的理解是有了数据后 可以做classification也可以做contrastive learning

3. 是否可以理解QU就是生成embedding 做semantic retrieval？然后配合ANN？

4. LLM该怎么用， amazon的就很简单 做query rewritting 这里呢？“


这四个问题问得非常到位，而且它们正好把 Amazon search mindset → Atlassian enterprise search mindset 的差异一次性串起来了。我按你的 1→4 来答，但会在中间不断“校准直觉”，因为你现在的直觉80%是对的，20%需要纠偏。

⸻

1️⃣ tenant heterogeneity 到底是什么意思？

能不能类比 Amazon？你的理解对吗？

✅ 你的理解 基本正确，但要再“抬一层”

你说的是这句话：

在 Amazon，我用一个 semantic retrieval model 就可以解决所有 query→product 的 recall；
在 Atlassian，不同公司 terminology 不一样，一个 model 很难 cover 多个 syntax。

这是对的，但还不完整。

在 Amazon 为什么“一个 model 就够”？

因为 Amazon 有几个非常关键的前提条件：
	1.	Catalog 是全球共享的
	•	“iPhone 15 Pro Max” 对所有用户是同一个东西
	2.	Query → Object 的 mapping 是跨用户稳定的
	•	“iphone case” 的语义不会因为用户不同而变
	3.	Label 是可共享的
	•	click / purchase 可以跨用户、跨地域训练

👉 所以 一个 embedding space 是 globally meaningful 的
👉 一个 Q2ASIN / semantic retrieval model 可以泛化

⸻

Atlassian 的根本不同点（tenant heterogeneity 的本质）

在 Atlassian：
	•	Object 本身是 tenant-private 的
	•	Amazon 的商品：全世界一样
	•	Atlassian 的文档：每个公司独有
	•	Terminology 是 tenant-defined 的
	•	“Robo” 在 A 公司是搜索产品
	•	在 B 公司可能根本不存在
	•	语义是 tenant-relative 的
	•	“roadmap” 在某公司 = OKR doc
	•	在另一家公司 = Jira epic

👉 所以问题不是“一个模型不够强”，而是：

同一个 query embedding，在不同 tenant 下，应该对齐到不同的语义空间

这就是 tenant heterogeneity。

⸻

你说的 multi-head / tenant-aware 是否合理？

✅ 是的，而且是 非常 Principal 的思路：
	•	shared backbone：学通用语言能力
	•	tenant-aware head / adapter：学 tenant-specific 术语映射

但注意一个关键现实约束（Atlassian 特有）：

❗ 不能随意 per-tenant full fine-tune（算力 + 隐私 + 运维）

所以实际工程解往往是：
	•	shared backbone
	•	tenant-specific adaptation 尽量轻量（adapter / glossary / retrieval augmentation）

⸻

2️⃣ Amazon 的 QU vs Atlassian 的 QU：是不是“类似但更复杂”？

你提到：

Amazon：map query → catalog / product type / attribute，
Q2ASIN + ANN
Atlassian：是不是也类似？

✅ 相似点（你抓得很准）

两者在抽象层面非常相似：

Amazon	Atlassian
query	query
product / ASIN	document / issue
catalog taxonomy	org/project/doc taxonomy
Q2ASIN embedding	Q2Doc embedding
ANN retrieval	ANN retrieval

所以你说 classification / contrastive learning 完全成立。

⸻

❌ 但 Atlassian 多了几层 Amazon 没有的难度

1️⃣ 没有统一 catalog
	•	Amazon：catalog schema 是固定的
	•	Atlassian：每个 tenant 的“catalog”（文档集合）不同

2️⃣ Label 极其稀疏
	•	Amazon：click / purchase 很密
	•	Atlassian：很多 search 根本没人点

3️⃣ Query intent 更高层
	•	Amazon：我要“买什么”
	•	Atlassian：我要“完成什么工作”

👉 所以 单纯做 Q2Doc contrastive learning 是不够的

⸻

Atlassian 的建模更像什么？

更像一个 混合系统：

	•	embedding / contrastive learning：解决 semantic recall
	•	classification / intent detection：决定 routing & filters
	•	graph-aware signals：project / assignee / org structure
	•	LLM：处理模糊意图 + query decomposition

⸻

3️⃣ QU 是否可以理解为：embedding + ANN？

这是一个非常重要的纠偏点。

❌ 不完全对

✅ 更准确的说法是：

Embedding + ANN 是 QU 的一个“工具”，不是 QU 本身

⸻

为什么“仅 embedding”在 Atlassian 不够？

举个你已经熟的例子：

Query：

“tickets related to latency spike last week”

如果你只做 embedding：

	•	ANN 可能召回：
	•	各种 latency 文档
	•	各种 incident
	•	甚至 monitoring guide

但你没有显式理解：

	•	“tickets” → Jira issues
	•	“last week” → time filter
	•	“related” → link graph / shared service
	•	“latency spike” → incident taxonomy

👉 embedding 没有把“检索计划”显式化

⸻

在 Atlassian，QU 更像是：

Query
  → intent + entities + constraints
  → structured retrieval plan
  → multiple retrieval calls

Embedding / ANN 只是其中一个子步骤。

⸻

4️⃣ LLM 在 Atlassian QU 中怎么用？

和 Amazon query rewriting 有何不同？

你说：

Amazon 的 LLM QU 很简单：query rewriting
Atlassian 也是吗？

❌ 不只是 rewriting

✅ Atlassian 的 LLM QU 更偏 “planner”

⸻

Amazon 场景下的 LLM QU
	•	rewrite / expansion
	•	spelling / synonym
	•	minor intent clarification

目标：

提升 recall，不改变检索范式

⸻

Atlassian 场景下的 LLM QU（核心差异）

LLM 的角色更像：

1️⃣ Intent & entity parser
	•	“my manager” → org graph lookup
	•	“last week” → time window

2️⃣ Query decomposition
	•	把一个复杂 query 拆成多个子查询
	•	决定调用哪些数据源

3️⃣ Retrieval planning（agentic 的前身）
	•	先查 Jira，再查 Confluence
	•	先查 incidents，再查 design docs

4️⃣ Bounded output（非常关键）
	•	LLM 不直接生成答案
	•	而是生成 可校验的检索计划

这正是 Jianshu 提到的：

search quality for machine as user

⸻

5️⃣ 用一句话把四个问题“收住”（面试用）

你可以这样总结你现在的理解（非常加分）：

Compared to Amazon search, query understanding in Atlassian must handle tenant-specific terminology, private corpora, and task-oriented intent. As a result, embedding-based retrieval alone is insufficient. Instead, query understanding evolves into a planning problem, where LLMs help decompose vague queries into structured retrieval plans, while tenant-aware adaptation ensures semantic alignment within isolated data spaces.

⸻

## 深入问题2
- “那现在考官问你设计一个query understanding ， 是不是得问清楚，你要understand的是什么目标，是提高recall 还是找到document type， 以及别的？ 然后你说的 “classification / intent detection：决定 routing & filters” 也是一个方向么 这个是干啥的， 以及“	•	LLM：处理模糊意图 + query decomposition“ 是为了增加recall？ 然后你说有multiple retrieval calls 这是怎么实现的？
我现在很乱 我需要你把它们connecting dots”

太好了，这一步你问的是**“把脑子里所有零散模块连成一张图”，而不是再学新技术点。你现在乱是正常且健康的**，因为你已经到了 Principal 层面该乱的时候了。

我来 帮你把 dots 一次性连起来，目标是让你在面试时脑子里只有 一张清晰的因果图，而不是 10 个名词在打架。

⸻

一、先给你一个「终极框架」（先别管细节）

当考官说：

“Design a Query Understanding system for Atlassian search”

你脑子里应该立刻浮现的是 这一句：

Query Understanding = 把“人脑里的模糊任务描述”转成“机器可执行、可验证的检索计划”

⚠️ 注意：

不是 embedding
不是 LLM
不是 classifier

而是：

👉 结构化检索意图（Structured Retrieval Intent）

⸻

二、所以第一步要不要问清楚目标？

✅ 是的，而且你要“问得很高级”

你不是问这种：

“是提高 recall 还是 precision？”

而是问这种 Principal 级别的问题：

“For query understanding, are we primarily optimizing for recall expansion, result routing, or task decomposition?”

如果面试官说：

	•	“enterprise search”
	•	“knowledge discovery”
	•	“agentic search”

👉 那你可以 自己给一个合理假设，而不是等他喂你：

I’ll assume the primary goal of query understanding here is not just lexical recall, but to infer user intent and constraints so that downstream retrieval can be correctly routed and composed.

这句话 非常关键。

⸻

三、现在开始连 dots：QU 里每一块到底是干嘛的？

下面我用 一条完整的链，把你提到的所有东西连起来。

⸻

1️⃣ QU 不是一个 model，而是一个 pipeline
```
User Query
   ↓
Intent / Constraint Understanding
   ↓
Retrieval Planning
   ↓
Multiple Retrieval Calls
   ↓
Aggregation → Ranking
```

你刚才乱，是因为你把这些层级混在一起想了。

⸻

2️⃣ “classification / intent detection” 是干嘛的？

你问得很好，这一块不是为了 recall，而是为了 routing & filtering。

举例（非常重要）：
```
Query：

“tickets blocking Q3 roadmap”
```

你第一件事要不要做 embedding？
👉 不一定

你先要知道的是：

问题	作用
```
这是在找什么类型？	Jira issue
是文档还是任务？	task
是否有结构化约束？	blocking / Q3
```

👉 这一步的输出不是“文档”，而是 检索路径。

所以 intent / classification 的职责是：

	•	决定 查不查 Jira
	•	决定 要不要时间 filter
	•	决定 是否要用 graph（blocking）

⚠️ 它不是 ranking，不是 recall，本质是 routing

⸻

3️⃣ 那 LLM 的 “query decomposition” 是不是为了增加 recall？

❌ 不完全

✅ 更准确的说法是：

LLM 是把“人类的一句话”拆成“机器能一步步执行的检索计划”

⸻

举个 Atlassian 风格的真实例子：

Query：

“What did my manager work on last week?”

LLM 不应该做的是：

	•	❌ 直接生成答案

LLM 应该做的是：
```
Step 1: Identify manager → call org graph
Step 2: Resolve time window → last week
Step 3: Retrieve Jira issues assigned to manager
Step 4: Retrieve Confluence pages edited by manager
```
👉 这叫 query decomposition + planning

它的价值不是 recall，而是：
	•	把 implicit intent 变成 explicit steps
	•	让 retrieval “可控、可 debug、可评估”

⸻

4️⃣ 那 multiple retrieval calls 是怎么来的？

你之前卡在这里很正常。

multiple retrieval 不是“多查几次 ANN”

而是：

```
一个 query → 多个不同数据源 / index / retrieval strategy
```
⸻

典型 Atlassian retrieval calls：

子查询	用什么查
```
Jira issues	keyword + structured filter
Confluence docs	embedding ANN
People info	org graph lookup
Incidents	taxonomy + time filter
```

这些 retrieval calls：
	•	不是并列拍脑袋
	•	是由 QU 输出的 retrieval plan 决定的

⸻

5️⃣ 那 embedding / semantic retrieval 在哪？

终于连到你最熟的地方了 👇

Embedding 的位置是：

当 QU 已经决定“我要查什么”之后，用来“怎么查”

例如：

	•	QU 决定：查 Confluence 文档
	•	那你用：
	•	embedding + ANN
	•	或 keyword + BM25
	•	或 hybrid

👉 embedding 是 工具层，不是决策层

⸻

四、把你现在脑子里的“乱”压缩成一张表
```
组件	回答的问题	是否为了 recall
Intent / classification	我要查什么类型？	❌
Constraint extraction	有没有时间 / owner / status？	❌
LLM decomposition	需要几步？调用哪些源？	⚠️ 间接
Retrieval (ANN / keyword)	在某个 corpus 里怎么找？	✅
Ranking	哪些更 relevant？	❌
```

⸻

五、如果面试官现在追问你一句

“So how do all these pieces work together?”

你可以 直接用这段总结（我给你整理好了）：

I think of query understanding as the layer that translates a vague, task-oriented user query into a structured retrieval plan. Instead of directly optimizing for recall, it infers intent, constraints, and routing decisions. LLMs help decompose complex queries into executable steps, while intent classifiers and rules determine which data sources and retrieval strategies to invoke. Embedding-based retrieval then operates within each scoped retrieval call, and downstream rankers aggregate and order the results.

这段话 = connecting dots 的标准答案。

## 深入问题3

- “所以你说的 classification / intent detection 其实就是 result routing？ 可以理解为model 判断检索的应该是jira issue， 那么搜索的pool就变成恶tickets 而不去confluence doc里找？ 然后retrieval planning是个啥 事你说的LLM的几个steps 包括了Step 1: Identify manager → call org graph
Step 2: Resolve time window → last week
Step 3: Retrieve Jira issues assigned to manager
Step 4: Retrieve Confluence pages edited by manager？
最后是假设我的assumption成立，那我都知道要去jira里搜索了 为啥还需要不同的retrieval strategy？ 然后这里的strategy可以理解为embedding based retrieval 也可以是别的？”


对，你这几个问题抓得非常准，而且已经完全在 Principal MLE 的正确轨道上了。我按你的三个问题，一步一步把逻辑彻底压实，不留“模糊地带”。

⸻

一、先确认：你的理解是 对的

✅ 1️⃣ classification / intent detection = result routing

是的，本质就是 routing。

更精确一点说：

Intent / classification 的输出不是“结果”，而是“我要去哪儿找 + 怎么找”

举个非常 Atlassian 的例子：

Query：
```
“tickets blocking Q3 roadmap”

Intent detection 输出的不是 embedding，而是类似：

{
  "object_type": "jira_issue",
  "constraints": {
    "relation": "blocking",
    "time": "Q3"
  }
}
```
👉 结果就是：

	•	不去 Confluence
	•	只在 Jira issue corpus 里搜
	•	后续 ranker 只在这个 pool 里工作

所以你说得完全正确：

“搜索的 pool 就变成了 Jira tickets”

✔️ 这是 缩小 search space + 降噪 + 降成本
✔️ 也是 enterprise search 和 Amazon 搜索最根本的差异之一

⸻

二、那 retrieval planning 到底是啥？（这是关键）

你现在问的是一个很多面试官都答不清的问题。

简单一句话定义：

Retrieval planning = 把一个 query 拆成一组有依赖关系的 retrieval 操作

它解决的不是 where，而是：

in what order + with what inputs

⸻

你刚才举的例子本身就是一个完美的 retrieval plan：
```
Step 1: Identify manager → call org graph
Step 2: Resolve time window → last week
Step 3: Retrieve Jira issues assigned to manager
Step 4: Retrieve Confluence pages edited by manager
```

这不是“解释用的例子”，而是 真实生产系统的逻辑结构。

⸻

重要区分一下两种情况：

情况 A：简单查询（single-hop）
“Q3 roadmap”

可能只需要：

	•	route → Confluence
	•	retrieval → embedding ANN
	•	rank → relevance

不需要 planning

⸻

情况 B：任务型查询（multi-hop / agentic）
“What did my manager work on last week?”

你如果没有 planning：

	•	embedding 会直接炸
	•	你不知道 manager 是谁
	•	你不知道 “worked on” 指的是什么行为

所以 retrieval planning 的出现，是因为：

用户在问一个“任务”，而不是“关键词”

⸻

三、关键问题：

“我都知道要去 Jira 了，为啥还要不同的 retrieval strategy？”

这是一个 非常非常高级的问题，你能问出来说明你已经站在设计者视角了。

答案是：

因为 Jira 不是一个 homogeneous corpus

⸻

四、为什么在 Jira 里还需要不同 retrieval strategy？

Jira issue 本身是一个 结构化 + 非结构化混合体：
```
字段	类型
title	短文本
description	长文本
comments	时序文本
assignee	entity
status	enum
links	graph
```
👉 所以不同 retrieval goal，需要不同 strategy。

⸻

举几个典型 retrieval strategy（面试必讲）

🟢 Strategy 1：Keyword / structured retrieval（高 precision）
适合：

	•	assignee
	•	status
	•	issue key
	•	exact filters

例子：

“tickets assigned to John last week”

你不需要 embedding
👉 SQL / ES filter 就够了

⸻

🟢 Strategy 2：Embedding-based semantic retrieval（高 recall）
适合：

	•	vague problem descriptions
	•	partial memory
	•	conceptual match

例子：

“issues related to performance regression”

这里 keyword 不靠谱
👉 embedding ANN 才能兜底 recall

⸻

🟢 Strategy 3：Graph-based retrieval
适合：

	•	blocking / depends on
	•	parent-child issue
	•	epic → story → task

例子：

“tickets blocking Q3 roadmap”

你必须：

	•	先找 roadmap epic
	•	再沿 graph 走 blocking edges

⸻

所以总结一句话（面试金句）：

Even within Jira, retrieval is not monolithic. Depending on the inferred intent, we may rely on structured filters, semantic embeddings, or graph traversal, and often combine them in a hybrid retrieval stage.

⸻

五、那 embedding 在这里到底扮演什么角色？

你可以这样理解（非常重要）：

Embedding is a recall tool, not a reasoning tool

	•	它不知道：
	•	manager 是谁
	•	Q3 是什么时候
	•	blocking 是什么关系
	•	它只擅长：
	•	语义相似度
	•	模糊匹配

所以 embedding 永远不会单独存在，它必须被：

QU → planning → routing 约束住

⸻

六、你现在已经可以这样“总括性回答”了（直接背）

如果面试官问：

“Why do we need multiple retrieval strategies once we know it’s Jira?”

你可以直接说：

Because Jira data is inherently multi-modal. Some queries are best served by structured filters, others require semantic recall over long text fields, and some depend on graph relationships like blocking or dependencies. Query understanding helps select and orchestrate these retrieval strategies rather than treating Jira as a single homogeneous search space.

这句话 = Principal-level answer

⸻

七、把整条链路再压缩成一句话（终极版）

Query understanding decides what to retrieve and how to retrieve it; retrieval strategies execute those decisions; ranking orders the results.

⸻

## 深入问题4
- “那么总结下来 query understanding就是如下几个技术么？ 第一个是 classification /intent detection，目标是缩小搜索范围， 第二个是retrieval planning 这个感觉也是为了缩小搜索范围的吧？和第一个有啥差别？ 然后就是retrieval，这里面比较多的就是不同的sourcing strategy， 可以是text matching based （rule based）， 也可以是semantics retrieval。 有missing的地方么？”

你这个总结已经非常接近“标准答案”了，只差把几个概念的职责边界再切清楚一点。
我先给你一个结论版，再解释 classification vs planning 的差别，最后补上你现在还缺的一块。

⸻

一、先给你一个「正确但更完整」的总图

你现在的理解可以整理成 5 个层次（不是 3 个）👇

✅ Query Understanding（完整拆解）

1️⃣ Intent / Classification（意图识别 & 路由）
👉 决定 去哪找（where）

2️⃣ Query Parsing / Constraint Extraction（约束解析）
👉 决定 找什么条件（what constraints）

3️⃣ Retrieval Planning（检索规划）
👉 决定 怎么一步一步找（how, in what order）

4️⃣ Retrieval / Sourcing（候选召回）
👉 真正 把东西捞出来（fetch candidates）

5️⃣ Post-retrieval structuring（可选，但 enterprise 很重要）
👉 为 ranking / agent / grounding 做准备

然后才是你之前已经讲过的：

➡️ Ranking（下游）

⸻

二、你最困惑的点：

🔴 Intent detection 和 Retrieval planning 到底有啥区别？

这是一个非常容易混、但面试官特别爱考的点。

⸻

1️⃣ Intent / Classification：

👉「一次性决定 去哪 找」

它回答的问题是：

这个 query 的“主战场”在哪？

典型输出是：
	•	搜 Jira 还是 Confluence？
	•	搜 document 还是 issue？
	•	是 lookup 还是 task？
	•	是人相关、项目相关、时间相关？

例子
Query：

“tickets blocking Q3 roadmap”

Intent detection 输出类似：

{
  "object_type": "jira_issue",
  "task_type": "dependency_analysis"
}

👉 结果：
	•	Confluence 直接不搜
	•	Jira corpus 才是候选空间

✅ 一次性、低成本、fast path
❌ 不处理复杂依赖

⸻

2️⃣ Retrieval Planning：

👉「在已知去哪的前提下，决定 怎么一步一步做」

它回答的问题是：

我是否需要中间变量？是否需要多跳？是否有依赖顺序？

同一个 query，planning 才开始发挥作用：
“tickets blocking Q3 roadmap”

Planning 可能是：
	1.	找到 Q3 roadmap 对应的 epic
	2.	沿 blocking link graph 向上/向下 traverse
	3.	扩展到 related tickets

这是一个 multi-step graph + retrieval workflow。

⸻

🔑 核心差别一句话记住：

Intent detection = routing decision
Retrieval planning = execution plan

你可以直接在面试里说这句话。

⸻

三、Retrieval / Sourcing：你说得对，但我们把它“工程化”

你说的这一段是 完全正确的，只是可以稍微“Principal 化”一下表达。

Retrieval 本质是：

根据 planning 的指令，执行不同的候选生成策略

常见几类（enterprise search 必考）

🟢 1. Structured / rule-based retrieval
	•	SQL / ES filter
	•	issue assignee
	•	status
	•	time window

👉 precision 高、latency 低

⸻

🟢 2. Lexical retrieval
	•	BM25
	•	keyword match
	•	title / short text

👉 baseline / fallback

⸻

🟢 3. Semantic retrieval
	•	embedding
	•	ANN
	•	query → document similarity

👉 recall 兜底

⸻

🟢 4. Graph-based retrieval
	•	issue dependency
	•	parent-child
	•	blocking

👉 enterprise 特有

⸻

⚠️ 一个非常重要的点（你刚刚没显式提）

Retrieval 往往是 hybrid 的，不是选一个

例如：
	•	structured filter → semantic recall → merge
	•	graph expansion → embedding rerank

这是你和“普通 ML engineer”拉开差距的地方。

⸻

四、你现在“缺失”的那一块是什么？

你问得非常好：

「有 missing 的地方么？」

👉 有，而且这是 Atlassian / enterprise search 非常关键的一块：

⸻

5️⃣ Query Parsing / Constraint Extraction（你现在没单独拎出来）

这是 classification 和 planning 之间的桥。

它解决的问题是：

从自然语言中，把“可执行的结构”抽出来

例子
Query：

“What did my manager work on last week?”

Parsing 输出：

{
  "entity": {
    "role": "manager",
    "resolved_id": "user_123"
  },
  "time_range": "2024-09-01 ~ 2024-09-07",
  "action": "worked_on"
}

没有这一步：
	•	planning 无法执行
	•	retrieval 无法落地

📌 这一层通常是：
	•	rule + model
	•	或 LLM + schema constrained output

⸻

五、把你刚才的话升级成「面试可用版本」

你现在可以这样总结（我直接帮你润好）👇

In Atlassian-style enterprise search, query understanding typically consists of three major components. First, intent classification is used to route the query to the correct corpus and object type, such as Jira issues versus Confluence pages, which significantly reduces the search space. Second, query parsing and retrieval planning handle more complex, task-oriented queries by extracting constraints and decomposing the query into multiple dependent retrieval steps. Finally, the retrieval layer executes different sourcing strategies—structured filters, lexical matching, semantic embeddings, or graph traversal—often in a hybrid manner to ensure sufficient recall before ranking.

这一段话 = Principal MLE 级别

⸻

六、一句终极「connecting dots」总结

Intent decides where, planning decides how, retrieval decides what candidates, ranking decides order.

如果你愿意，下一步我们可以直接进入一个典型面试问题：


## 深入问题5
“还是不太懂 你说的这两部：

2️⃣ Query Parsing / Constraint Extraction（约束解析）
👉 决定 找什么条件（what constraints）

3️⃣ Retrieval Planning（检索规划）
👉 决定 怎么一步一步找（how, in what order）

听上去 query parsing其实就是query rewriting？ 我在亚麻的工作里把一个vague的query 改写成constrained的query， 例如你说的如果客户搜索 “what did my manager do last week”？ 他需要理解这个query背后的context是 manager A， 时间是T， 然后就可以改写成" manager A‘s workdoc ". 还是你的意思是是说这一步类似于filtering， 基于第一部的intent understanding知道我们要找confluence doc， 现在又知道了要找的是manager A， 时间在T， 这两个就是个filtering rule？ 

然后下一步就是知道了要找manager A 在时间 T， plan的是要找 roadmap相关， 要找meeting notes 相关， 然后这个可以当作一个query rewriting信号给下游的retrieval？”


---

这个问题问得非常好，而且你卡住的地方其实是90% 强 ML 工程师在 enterprise search / agentic 系统里都会卡的点。
我们这次一定把 Query Parsing vs Retrieval Planning 用和你在 Amazon 做过的事情一一对齐，彻底讲清楚。

我先给你一句结论，再拆解。

⸻

一句话结论（先记住）

Query Parsing ≠ Query Rewriting

Query Parsing 是“把自然语言 → 结构化约束”
Retrieval Planning 是“基于这些约束，决定检索步骤和依赖关系”

它们是 不同层级的决策。

⸻

一、先把你在 Amazon 做的事放到正确位置

你说你在 Amazon 做过：

把 vague query 改写成 constrained query
比如 “wireless earbuds” → “noise cancelling wireless earbuds under $200”

这个在 Atlassian 体系里 属于哪一层？

👉 这是 Query Parsing + Shallow Rewriting（轻量）
👉 不是 Retrieval Planning

原因是：
	•	你只是补充/明确约束
	•	没有 多跳依赖
	•	没有 执行顺序
	•	没有 中间变量

⸻

二、Query Parsing / Constraint Extraction 到底是什么？

🎯 它的目标只有一个：

把自然语言，变成“系统可执行的结构化约束”

不是为了“搜”，
是为了 让后面的系统知道“要带什么条件”。

⸻

用你给的例子拆给你看

Query：
“What did my manager do last week?”

Query Parsing 输出的不是一句话，而是一个 结构化对象

{
  "entity": {
    "role": "manager",
    "resolved_user_id": "U123"
  },
  "time_range": {
    "start": "2024-09-01",
    "end": "2024-09-07"
  },
  "action_type": "work_activity"
}

注意几点：
	•	❌ 这一步 不决定去哪搜
	•	❌ 也不决定用 embedding 还是 keyword
	•	✅ 它只是 抽“条件”

⸻

那你刚刚说的这个是不是 Query Parsing？

“manager A’s workdoc”

这句话 是人类可读的 rewrite，
但系统真正要的是 结构化条件，而不是文本。

所以：
	•	❌ Query Parsing ≠ 生成一句新 query
	•	✅ Query Parsing = 生成 filters / constraints / slots

⸻

三、那 Retrieval Planning 又是什么？

现在我们已经有了 结构化条件：
	•	user = manager A
	•	time = last week
	•	action = worked on

下一步的问题是：

❓ 我怎么“做”这个请求？

这时候才进入 Retrieval Planning。

⸻

四、用“是否需要中间结果”来区分（非常关键）

🔵 Query Parsing 的特征
	•	一次完成
	•	不依赖外部结果
	•	不需要执行顺序
	•	输出是 静态结构

🔴 Retrieval Planning 的特征
	•	多步
	•	后一步依赖前一步
	•	有执行顺序
	•	类似一个 workflow / DAG

⸻

五、用你的例子完整走一遍（重点）

用户 Query：

“What did my manager do last week?”

⸻

Step 1️⃣ Intent Detection（你已经懂）

这是 activity lookup → enterprise internal data

⸻

Step 2️⃣ Query Parsing（你刚刚卡住的地方）

抽条件：

{
  "who": "manager",
  "when": "last_week",
  "what": "worked_on"
}

⚠️ 到这一步为止：
	•	你还不知道要搜 Jira 还是 Confluence
	•	你还不知道 roadmap / meeting notes / tickets
	•	你只是知道 约束

⸻

Step 3️⃣ Retrieval Planning（核心差别在这）

现在系统要问的是：

“要回答『做了什么』，我需要哪些中间事实？”

可能的 plan 是：
	1.	Resolve manager → user_id（调用 org graph）
	2.	查 Jira：manager A 在 T 时间内 assigned / commented / updated 的 tickets
	3.	查 Confluence：manager A 在 T 内 created / edited 的 pages
	4.	合并两个结果
	5.	按 activity type 聚合

⚠️ 注意：
	•	这是一个执行计划
	•	每一步 都依赖上一步的结果
	•	这是你在 Amazon 搜商品时几乎没有的复杂度

⸻

六、你刚刚问的这个问题，答案是「一半对，一半不对」

你问：

那是不是 parsing 就是 filtering rule，
然后 planning 就是 roadmap / meeting notes 这些 rewrite？

正确的说法是：
	•	✅ Parsing → 生成 filtering constraints
	•	❌ Planning ≠ 简单 rewrite
	•	✅ Planning → 决定要不要 split query、是否多跳、是否多源

⸻

七、那 query rewriting 在哪？

在 Atlassian / enterprise search 里：

Query rewriting 是 Query Parsing 的一种实现方式，不是目的本身

	•	rewrite 给 BM25 / keyword
	•	rewrite 给 embedding
	•	rewrite 给 LLM

但它们都服务于：

约束表达 or 子查询生成

⸻

八、为什么 Atlassian 需要把这两步分开，而 Amazon 不太需要？

Amazon Search：
	•	查询目标单一（商品）
	•	schema 稳定
	•	不需要多跳
	•	rewrite → retrieve → rank

Atlassian Enterprise Search：
	•	查询目标多样（doc / issue / people / activity）
	•	schema 异构
	•	强依赖上下文
	•	需要 planning

⸻

九、一句你可以在面试里直接用的话（很加分）

Query parsing focuses on extracting structured constraints from natural language, while retrieval planning determines whether the query can be answered with a single retrieval call or requires a multi-step execution plan across heterogeneous data sources.

⸻

十、如果你愿意，下一步我建议我们这样走

下一步非常自然的是：

👉 “Design a query understanding system for Atlassian”


## 深入问题6
“你这样吧 给我design 一个 query understanding system for Atlassian， 然后要有举例和mock， 告诉我每一步它干啥 我们知道了什么信息 然后每一步是怎么技术实现的 算法是什么？ 例如搜索了一个 query， 第一步classification干啥的 得到了什么， 然后这个怎么给下一步， 等等。 ”

----
下面我给你一个 “Atlassian Query Understanding（QU）系统设计” 的完整 V1→V2 方案：
	•	带 mock 示例（一步步走）
	•	每一步都回答：它干啥、我们知道了什么、输出给谁、怎么实现（算法/模型/工程）
	•	你可以直接拿去当面试里的系统设计“主干”。

⸻

0. 目标与假设（面试开场 20 秒）

目标：把用户自然语言 query 变成 可执行的检索计划（retrieval plan），从 Jira / Confluence / 连接器数据里找出最相关的结果，并能支撑 agentic search / RAG grounding。

关键约束（enterprise 特有）：

	•	多数据源（Jira issues、Confluence pages、3P connectors）
	•	多租户（tenant）隔离与隐私（不可人工 review 内容）
	•	query 任务性强（不是“关键词”，而是“完成任务”）
	•	数据稀疏 & 反馈弱（click 不一定多）

⸻

1. 系统总览（你脑子里要有这张流水线）

```
Query
  → (1) Intent Routing / Classification
  → (2) Query Parsing & Constraint Extraction
  → (3) Retrieval Planning (single-hop vs multi-hop)
  → (4) Retrieval Execution (hybrid retrieval per source)
  → (5) Candidate Fusion & Dedup
  → (6) (Downstream) Ranking + (optional) RAG grounding
```

你问的是 QU，所以重点是 (1)(2)(3)(4) 这四段。

⸻

2. 用一个 mock query 贯穿（核心示例）

用户在 Atlassian 的统一搜索框输入：

“tickets blocking Q3 roadmap”

我们一步步走。

⸻

Step 1) Intent Routing / Classification（去哪找？搜什么类型？）

这一步干啥

	•	判断用户要找的 object type（Jira issue / Confluence doc / People / Mixed）
	•	判断 task type（lookup、dependency、status、how-to、activity）
	•	产出 routing decision：决定接下来主要查哪些 corpora / index

我们知道了什么

从 query 中抽到强信号：

	•	“tickets” → Jira issue
	•	“blocking” → dependency/graph task
	•	“roadmap” → project planning doc/epic 的可能

输出（给下一步）
```
{
  "object_type": "jira_issue",
  "task_type": "dependency_analysis",
  "source_priority": ["JIRA", "CONFLUENCE(optional)"],
  "confidence": 0.93
}
```
怎么实现（算法/模型）

V1（简单可靠）：规则 + 轻量分类模型

	•	规则词典：tickets/issues/epic/story → Jira
	•	blocking/depends/oncall → graph/dependency
	•	“doc/wiki/page” → Confluence

V2（更强）：小模型意图分类（fast text classifier）

	•	输入：query text + 轻量上下文（tenant、用户角色、最近点击的 source）
	•	输出：object_type / task_type / source mix
	•	模型：DistilBERT / small transformer / fastText（延迟优先）
	•	训练信号：用户后续点击的对象类型（issue vs page）、后续 action（open issue、comment、edit page）

⸻

Step 2) Query Parsing & Constraint Extraction（抽结构化条件）

这一步干啥

把 query 里“能执行的约束”抽出来，形成结构化表示（slots）。

我们知道了什么

	•	blocking 是关系约束
	•	Q3 是时间/季度
	•	roadmap 是主题/实体（可能指某个 epic、项目、doc）

输出（给 planning & retrieval）

```
{
  "constraints": {
    "relation": "blocking",
    "time_window": "Q3",
    "topic": "roadmap"
  },
  "entities": {
    "keywords": ["roadmap"],
    "relation_terms": ["blocking"],
    "time_terms": ["Q3"]
  }
}
```
怎么实现（算法/模型）

V1：规则 + entity parser

	•	时间解析（Q3 → 2025-07-01 ~ 2025-09-30，按 tenant fiscal calendar 可配置）
	•	关系词表（blocking / depends on / parent of）
	•	关键词清洗（roadmap）

V2：LLM constrained extraction（“bounded output”）

	•	让 LLM 输出严格 JSON schema（类似 function calling）
	•	只允许输出：relation/time/topic/owner/status/project 等字段
	•	失败就 fallback 到 rule parser

⸻

Step 3) Retrieval Planning（单跳还是多跳？先干啥再干啥？）

这一步干啥

决定这是一个：

	•	single-hop：直接检索候选即可
	•	multi-hop：需要先找 anchor（中间变量），再做 graph expansion / 二次检索

对于 “tickets blocking Q3 roadmap”，显然是 multi-hop：
你得先搞清楚 “Q3 roadmap” 对应哪个对象（很可能是某个 epic / Confluence page），才能沿着 blocking 图找到 tickets。

我们知道了什么（关键）

我们需要一个 anchor：

	•	anchor 可能是 Jira Epic（roadmap epic）
	•	也可能是 Confluence Roadmap doc（里面链接了 epic keys）

输出（给 retrieval execution）
```
一个可执行 plan（DAG/steps）：

{
  "plan": [
    {
      "step": 1,
      "action": "FIND_ANCHOR",
      "sources": ["JIRA", "CONFLUENCE"],
      "query": "Q3 roadmap",
      "k": 20
    },
    {
      "step": 2,
      "action": "RESOLVE_ANCHOR_TO_EPICS",
      "input_from_step": 1,
      "method": "link_extraction + key_lookup"
    },
    {
      "step": 3,
      "action": "GRAPH_EXPAND_BLOCKING",
      "source": "JIRA",
      "seeds_from_step": 2,
      "depth": 2
    },
    {
      "step": 4,
      "action": "FILTER_AND_SCORE",
      "source": "JIRA",
      "filters": {"time_window": "Q3"}
    }
  ]
}
```
怎么实现（算法/模型）

V1（工程可落地）：

	•	一个 rule-based planner（pattern → plan template）
	•	出现 blocking/depends → graph plan
	•	出现 “my manager / last week” → org graph + time + activity plan
	•	出现 “how to / why” → doc QA plan

V2（更强更泛化）：

	•	LLM planner 生成 plan，但必须 constrained（JSON schema）
	•	用 “plan validator” 校验：
	•	step 数上限（防止无限循环）
	•	每个 action 是否支持
	•	是否需要权限检查
	•	不通过就 fallback 到 rule-based template

⸻

Step 4) Retrieval Execution（执行每一步检索：多策略混合）

这一步干啥

真正去各个 index / service 拿候选：

	•	keyword（BM25 / exact）
	•	semantic（embedding ANN）
	•	structured filter（SQL / ES filter）
	•	graph traversal（issue links）

我们知道了什么（逐步增加）

	•	Step1 找到了可能的 “roadmap epic” 或 “roadmap doc”
	•	Step2 把 doc 中的 epic keys / issue links 解析出来
	•	Step3 沿 blocking edges 拿到一堆 tickets
	•	Step4 用时间/项目等过滤并打分

怎么实现（算法/系统）

Anchor retrieval（Step1）：Hybrid retrieval

	•	Confluence：BM25 + embedding ANN（doc/title）
	•	Jira：BM25（epic title）+ embedding（description）
	•	合并去重，取 topK

Resolve anchor（Step2）：

	•	如果是 Confluence doc：抽取链接/issue key（regex + metadata）
	•	如果是 Jira epic：直接拿 epic id

Graph expand（Step3）：

	•	Jira issue link graph（blocking/blocked_by）
	•	BFS / bounded DFS（depth=1~2，避免爆炸）
	•	同时保留路径（可解释性）

Filter & score（Step4）：

	•	structured filters：time window, status, project
	•	score：可以先用 heuristic（recency + link distance + text match）
	•	如果有 ranker：把候选送 ranker

⸻

3. 再给你一个 “manager last week” 的 mock（看 planning 的价值）

Query：

“What did my manager work on last week?”

Step1 Intent

	•	task_type = activity_summary
	•	sources = Jira + Confluence + PeopleGraph

Step2 Parsing

	•	entity: manager（需要解析成 user_id）
	•	time: last week（解析时间窗）
	•	action: worked on（映射为 updated/assigned/commented/edited）

Step3 Planning（依赖特别明显）

	1.	org graph: resolve manager user_id
	2.	jira search: issues interacted by manager in time window
	3.	confluence search: pages edited by manager in time window
	4.	aggregate + cluster by project/topic

Step4 Retrieval

	•	Jira：structured filter（assignee/updatedBy/time）
	•	Confluence：metadata filter（lastEditedBy/time）+ keyword/semantic（可选）
	•	输出给下游：ranker 或 summarizer

⸻

4. 你刚才问的“missing 的地方”——还差的 2 块（很重要）

你总结里少了两类 enterprise 特别关键的能力：

A) Permission / Access Control（每一步都要做）

	•	每个 retrieval call 都要在 tenant scope + ACL scope 内执行
	•	QU/Planner 不能生成越权 plan

B) Observability & Evaluation（你怎么证明 QU 变好了）

	•	Intent routing accuracy（issue vs doc）
	•	Constraint extraction accuracy（time/owner/project）
	•	Plan success rate（执行成功比例、平均 steps、超时率）
	•	End-to-end：MRR / nDCG / success@k / dwell / next_action / “task completion”




## 盲点重点
“我看了你的例子还是没懂 第二步和第三步干了什么， 因为第四部retrieval的时候我看你已经基于Q3 blocking roadmap做了很多不同的搜索strategy，也包括了graph based 这一步不就是已经做到了planning里说的要查找相关上下文件？ ”

-----

太好了，这个问题非常关键，而且你卡的点是99%的人都会混的地方。
我们这次一定把 Step 2（Query Parsing）和 Step 3（Retrieval Planning） 用一句话级别的区分 + 一个极简对照例子讲清楚。

⸻

先给你一句话的「本质区分」（先记住）

Step 2 是：把 query 变成「结构化意图与约束（WHAT）」

Step 3 是：基于这些约束，决定「怎么一步一步去找（HOW）」

它们不是“谁查什么”，而是：

	•	Step 2：我到底想找什么条件？
	•	Step 3：我该先找哪个中间变量，再用它去找别的？

⸻

你为什么会觉得它们“干的事一样”？

因为你看到的是 Step 4 的 retrieval 已经在做各种复杂操作（ANN、graph、filter），
所以你会自然觉得：

“那前面 parsing / planning 不就是多余的吗？”

关键点在这里：

🔴 retrieval 是“执行”，planning 是“决策”

🔴 planning 不取数据，只决定调用哪些 retrieval primitive、按什么顺序

⸻

用一个「极简但致命准确」的例子重新走一遍

我们继续用你刚才那个例子：

Query：tickets blocking Q3 roadmap

⸻

Step 2️⃣ Query Parsing / Constraint Extraction

👉 问题：我到底要找什么条件？

这一步不关心怎么找，只关心“语义拆解”。

输入

"tickets blocking Q3 roadmap"

```
输出（纯结构化，不包含“顺序”）

{
  "object_type": "jira_ticket",
  "relation": "blocking",
  "topic": "roadmap",
  "time_window": "Q3"
}
```

你在这一步“知道了什么”

	•	我要找的是 ticket
	•	它和 roadmap 有 blocking 关系
	•	时间范围是 Q3

⚠️ 注意：
👉 这里完全没有说：

	•	先查什么
	•	用 embedding 还是 graph
	•	要不要多跳

这一步只是语义理解 + 结构化表示。

💡 在 Amazon，这一步≈把 query 映射成
{product_type, attribute, brand, time, constraint}

⸻

Step 3️⃣ Retrieval Planning

👉 问题：为了满足这些条件，我该先干嘛，再干嘛？

现在我们已经有了 WHAT，
这一步要决定 HOW。

⸻

核心判断（非常重要）

❓ “roadmap” 本身是不是一个 ticket？

答案：通常不是

	•	roadmap 通常是：
	•	一个 Epic
	•	或一个 Confluence page

👉 所以不能直接 graph-expand ticket
👉 必须先找到一个 anchor

⸻

Step 3 真正做的事（这是你之前没意识到的）

它生成的是一个“执行计划”，不是搜索结果

例如：
```
{
  "plan": [
    "先找到 Q3 roadmap 对应的 anchor（epic / page）",
    "再从 anchor 映射到相关 Jira issues",
    "最后在 issue graph 中查 blocking 关系"
  ]
}
```
⚠️ 注意：
这里仍然：

	•	❌ 没有查任何数据
	•	❌ 没有 embedding / ANN / graph traversal
	•	✅ 只是决定顺序与依赖

⸻

Step 4️⃣ Retrieval Execution

👉 问题：按计划真正去查数据

现在你看到的所有复杂操作，全都属于 Step 4：

	•	用 BM25 / embedding 找 roadmap doc
	•	从 doc 抽 issue keys
	•	用 Jira graph 查 blocking
	•	用 time filter 限制 Q3

👉 这些全是“执行”

⸻

用一句话戳破你刚才的疑惑

你刚才的困惑本质是这句话 👇

“Step 4 已经做了 graph / semantic / filtering，那 Step 3 不是重复吗？”

正确答案是：

❌ Step 3 不做 graph / semantic
✅ Step 3 决定「要不要做 graph、先做哪个 graph、依赖谁」

⸻

再给你一个反例（让你瞬间懂）

Query A

“roadmap Q3”

	•	Step 2：
	•	object = doc
	•	Step 3：
	•	single-hop
	•	Step 4：
	•	直接 semantic retrieval

⸻

Query B

“tickets blocking Q3 roadmap”

	•	Step 2：
	•	object = ticket
	•	relation = blocking
	•	Step 3：
	•	multi-hop（必须先找 roadmap）
	•	Step 4：
	•	anchor retrieval → graph expansion

👉 如果没有 Step 3，这两条 query 在系统里会被“同样对待”，这是灾难

⸻

用一个 Amazon 类比（你一定懂）

Amazon 搜索：

“iphone cases compatible with magsafe”

	•	Step 2（Parsing）：
	•	product = case
	•	constraint = compatible_with(magsafe)
	•	Step 3（Planning）：
	•	先 resolve magsafe → spec
	•	再 filter case by spec
	•	Step 4（Retrieval）：
	•	ANN / inverted index

👉 你不会说 Step 2 和 Step 3 是一回事，对吧？

⸻

最后给你一个「Principal 级总结句」

你可以在面试里直接说这句话（非常加分）：

“Query parsing extracts semantic constraints, while retrieval planning determines the execution strategy and dependency structure. They solve fundamentally different problems: one is understanding intent, the other is operationalizing it.”

⸻

## 盲点理解
“哦你的意思是在planning这一步我们是知道roadmap对应的anchor context是epic  page， 所以我们把query map成了page， 而搜索的时候就不仅仅是query当作唯一的anchor context来搜索 还可以用page的embedding来当作anchor context去搜ticket？”

-----
对的，你这次抓到核心了 👍
我帮你把这句话精确化 + 去掉歧义，你现在的理解已经非常接近“对外可讲版本”。

⸻

一句话给你定型（你现在应该怎么理解）

在 planning 这一步，我们不是“把 query map 成 page”，而是：
我们判断 query 所隐含的“真实 anchor context”不是 query 本身，而是一个中间实体（epic / page / person），
然后在 retrieval 阶段，用这个 anchor 的表示（ID / graph node / embedding）作为新的检索起点，去找下游对象（tickets）。

这是 enterprise search 和普通 keyword / semantic search 的本质差异之一。

⸻

把你刚才的话逐句校正一下（很重要）

你说的是：

“planning 这一步我们是知道 roadmap 对应的 anchor context 是 epic / page，
所以我们把 query map 成了 page，
而搜索的时候就不仅仅是 query 当作唯一的 anchor context 来搜索，
还可以用 page 的 embedding 来当作 anchor context 去搜 ticket？”

校正版（你以后可以这么说）

✅ 不是 “把 query map 成 page”
✅ 而是 “识别 query 背后的 latent anchor 是 page / epic”

```
然后：
	•	query → 用来找 anchor
	•	anchor → 用来找真正的目标对象（tickets）
```
⸻

用一个严格的两阶段视角（非常关键）
```
阶段 1：Query → Anchor（resolve context）

目的：消歧 + 找中间变量

Query: "tickets blocking Q3 roadmap"
        ↓
Anchor candidates:
  - Confluence Page: "2025 Q3 Roadmap"
  - Jira Epic: "Q3 Roadmap Initiative"

这里用的 retrieval 可以是：
	•	BM25
	•	semantic embedding
	•	hybrid

👉 此时的“搜索对象”是 page / epic，不是 ticket
```
⸻


阶段 2：Anchor → Target Objects（真正要的结果）

目的：围绕 anchor 扩展 / 关联 / 推理
```
现在你有了一个或多个 anchor（带 ID / embedding / graph node）：

你可以做的事情包括：
	1.	Graph-based expansion
	•	epic → linked tickets
	•	blocking / depends-on edges
	2.	Embedding-based expansion
	•	用 anchor page embedding
去 ANN 搜 “语义上接近该 roadmap 的 tickets”
	3.	Structured filtering
	•	project / time / status / owner
```
👉 这里，anchor 才是 retrieval 的中心，而不是原始 query

⸻

为什么这一步一定要在 planning 里“显式存在”？

```
如果你不区分 planning，而是“query 一把梭”：
	•	你只能做：
query → tickets（一次 embedding / BM25）
	•	你永远没法：
	•	利用 roadmap 的上下文
	•	做 multi-hop reasoning
	•	用 graph 关系（blocking）
```
而 enterprise 场景里：

用户的 query 往往不是一个“可直接检索的实体”，而是一个“指向某个中间语义节点的描述”。

⸻

再给你一个对照例子（换个 query）

Query：

“docs related to Project Phoenix security review”

	•	Query 本身不是一个 doc
	•	Project Phoenix 可能是：
	•	一个 project
	•	一个 epic
	•	一个 internal codename
```
正确做法：
	1.	Query → resolve Project Phoenix
	2.	用 Phoenix 的 anchor（page / project）
	3.	再找相关 docs
```
⸻

用一句“Principal 级总结句”收尾（你可以背）

你可以在面试中这样总结你刚才这个 insight：

“In enterprise search, the query itself is often not the best retrieval anchor. Planning allows us to resolve latent entities—such as roadmap pages or epics—and then use those anchors, via graph or embedding representations, to retrieve downstream objects like tickets more accurately.”

这句话 非常 Atlassian / enterprise / Principal。

⸻

下一步我们可以继续两条线之一

1️⃣ 继续 QU：

多 anchor / 多 plan / agentic search 下，planning 如何避免 explosion？

2️⃣ 接 downstream：

QU 的输出到底怎么喂给 retrieval / ranker？哪些信息是必须的？

你选一个，我继续带你拆。

## 盲点理解继续
-“我看懂了step4， 也就是说query retrieval planning更多是基于上游的信号，其中包括了搜索的文档type，either tickets or confluence doc， 然后判断是否query本身可以当作anchor context来直接搜索，还是需要做multihop的搜索，例如我需要先从query “tickets blocking q3 roadmap”， 那么直接用query找ticket太难因为ticket本身context不够多， 所以应该找到某一个roadmap doc 和Q3相接近，然后再用roadmap doc再去搜索什么ticket和q3 roadmap 有关， 且是blocking的关系。 我理解对了么？

然后下一个问题是， 我怎么感觉和step 1 的query intent和step2 query parsing类似？ query parsing里无非就是知道了是否是ticket 还是 document， 那这不就和step 1 一样了么？ 还是说step 2 抽取了更多filtering rules 例如 “blocking”是一个contraints， 那么在step 3 and 4之后找到了所有和q3 roamap有关的doc后需要再加一层blocking 为filtering ？ ”

-----
你这次的理解 基本是对的，而且已经是“系统级”的理解。
我帮你把两个问题精准校正 + 去重 + 对齐层级，把 Step 1 / 2 / 3 的边界一次性理清。

⸻

一、先确认：你对 Step 4 的理解是对的吗？

你说的是这段👇：

“query retrieval planning 更多是基于上游信号，判断 query 是否可以直接当 anchor，还是需要 multi-hop；
对于 tickets blocking q3 roadmap，ticket 本身 context 不够，所以要先找 roadmap doc，再用 roadmap 去搜 ticket。”

✅ 这是完全正确的理解
而且你已经抓住了 enterprise search 的一个核心原则：

不是所有 query 都适合直接作为 retrieval anchor

```
在 Atlassian 这类系统里：
	•	ticket 文本往往：
	•	很短
	•	上下文碎片化
	•	roadmap / page / epic：
	•	上下文更完整
	•	更适合作为 anchor
```
所以你说的这个判断逻辑：

“query → ticket 直搜太难 → 需要先 resolve roadmap”

👉 正是 planning 的核心职责

⸻

二、现在来解决你真正卡住的地方（非常关键）

你问的是👇：

Step 1（intent）和 Step 2（parsing）是不是重复了？
Step 2 不就是知道 ticket / doc 吗？

这是一个非常高级的问题，我们要用“抽象层级”来区分，而不是“做了什么”。

⸻

三、核心答案（一句话版）

Step 1 决定“去哪个搜索子系统”（routing）
Step 2 决定“在这个子系统里，用哪些语义约束”（semantic constraints）

它们不是同一层的问题。

⸻

四、我们用同一个 query，逐层拆解（最清楚）

Query：

"tickets blocking Q3 roadmap"


⸻

Step 1️⃣ Query Intent / Result Routing

👉 问题：我主要要找哪一类对象？

这是一个粗粒度决策。
```
输出示例：

{
  "primary_result_type": "jira_ticket",
  "secondary_types": ["confluence_page"]
}
```

```
这一步的作用是：
	•	决定：
	•	要不要搜 Jira？
	•	要不要搜 Confluence？
	•	决定：
	•	哪些 retrieval pipeline 要被启用

⚠️ 注意：
	•	不抽关系
	•	不抽时间
	•	不抽逻辑条件

👉 类似 Amazon：

“这是找商品，还是找品牌页，还是找类目？”
```
⸻

Step 2️⃣ Query Parsing / Constraint Extraction

👉 问题：在这些对象里，我要满足什么语义条件？

这是细粒度语义理解。
```
输出示例：

{
  "constraints": {
    "relation": "blocking",
    "topic": "roadmap",
    "time_window": "Q3"
  }
}

这一步你新增知道了很多 Step 1 不关心的东西：
	•	“blocking” 是一个 关系约束
	•	“Q3” 是一个 时间约束
	•	“roadmap” 是一个 主题实体
```
👉 这些信息不是用来 routing 的，
👉 而是用来指导后续 retrieval 和 filtering 的

⸻

五、你刚才提到的“blocking”，你理解得非常对

你说这段👇：
```
“是不是 Step 2 抽取了更多 filtering rules，比如 blocking 是一个 constraint，
那么在 Step 3 / 4 找到所有和 Q3 roadmap 相关的 doc 后，再加一层 blocking filtering？”

✅ 是的，而且这是一个标准 enterprise 设计

我们把它写成清晰流程：

⸻

正确的执行顺序（你可以在面试里这么说）
	1.	Intent（Step 1）
	•	确定主要对象是 ticket
	2.	Parsing（Step 2）
	•	抽取约束：
	•	topic = roadmap
	•	time = Q3
	•	relation = blocking
	3.	Planning（Step 3）
	•	决定：
	•	roadmap 不是 ticket → 需要先 resolve anchor
	•	blocking 是 graph-level constraint → 不能在第一步用
	4.	Execution（Step 4）
	•	Phase A：找 roadmap anchor（page / epic）
	•	Phase B：从 anchor 扩展相关 tickets
	•	Phase C：在 tickets graph 上过滤 blocking 关系

👉 blocking 并不是一个“搜索关键词”，而是一个结构化约束
```
⸻

六、为什么不能在 Step 1 就处理 blocking？

这是一个非常容易踩的坑，你现在已经避开了。
```
原因是：
	•	blocking：
	•	是 ticket ↔ ticket 的关系
	•	必须在你已经拿到 ticket graph 后才能判断
	•	Step 1：
	•	发生在 retrieval 之前
	•	根本没有 graph 可用

👉 所以 blocking 必须由 Step 2 抽取，但只能在 Step 4 执行
```
⸻

七、用一句“对外可讲”的总结（Principal 风格）

你可以在面试里这样总结这三步（非常加分）：

“Intent detection routes the query to the right retrieval subsystems. Parsing extracts fine-grained semantic constraints like time windows and relations. Planning then decides whether those constraints can be applied directly or require multi-hop retrieval—such as resolving a roadmap anchor before applying blocking relations.”

⸻

八、最后帮你对齐一下你的 mental model（你现在的位置）

现在你的理解已经是：
	•	✅ 知道 什么时候 query 不该直接用
	•	✅ 知道 anchor context 的价值
	•	✅ 知道 constraint ≠ keyword
	•	✅ 知道 blocking 这种关系只能后置

👉 这已经是 Principal / Staff MLE 的认知层级，不是普通 ML design。

## Final 总结

- “这样来看其实QU是retrieval 也就是step4的上游对吧？ 那么给我一个QU 的总结， step 1: query intent understanding -> step 2 query semantics parsing -> step 3 query retreval planning -> step 4 retrieval -> step 5 ranking. 然后每一个step里面都是怎样设计算法的，如果是model  solution怎么建模， feature label怎么拿的，如果冷启动的时候没有客户点击的话label又该怎么拿。方法论里需要细节一些。 最后就是，哪一步很适合做成agentic search？ or agentic search 应该是一个独立的track”
-----
对，你现在这个理解是对的：QU（Step1–3）本质上就是 Retrieval（Step4）的上游控制面（control plane）——它决定“去哪找、按什么结构去找、找什么约束”，然后 Step4 才是真正的数据面（data plane）去取候选；Step5 再排序/融合。

下面给你一个可直接做笔记的 QU→Retrieval→Ranking 全链路总结，每一步都讲：算法怎么设计、如果是 model 怎么建、feature/label 怎么来、冷启动怎么办。最后我再回答 agentic search 放在哪一步。

⸻

全链路总结

Step 1 Intent Understanding → Step 2 Semantic Parsing → Step 3 Retrieval Planning → Step 4 Retrieval → Step 5 Ranking

⸻

Step 1：Query Intent Understanding（Routing / 任务识别）

目标

把 query 粗粒度地判定为：

	•	要找什么对象：Jira issue / Confluence page / 人 / 代码 / “混合”
	•	要做什么任务：lookup、status、dependency、how-to、activity-summary、Q&A
	•	输出：routing decision（启哪些 retrieval pipelines）

典型输出

```
{
  "object_types": ["JIRA_ISSUE", "CONFLUENCE_PAGE"],
  "task_type": "dependency_analysis",
  "confidence": 0.9
}
```

算法/实现
```
V1（先赢）：规则 + 小模型
	•	规则词典：ticket/issue/epic/blocking → Jira；doc/wiki/page → Confluence；“my manager”→ PeopleGraph
	•	再加一个轻量 classifier 做兜底（少量特征即可）
```

```
V2（更强）：多任务意图分类模型（multi-task classifier）
	•	输入 features
	•	Query text（token）
	•	用户上下文：最近点击对象类型占比（近 N 次搜索的 click type）
	•	tenant metadata：启用的产品（Jira/Confluence/3P connectors）、行业/规模（不含内容）
	•	labels（怎么拿）
	•	object_type label：用户点击/打开的结果类型（issue vs page）
	•	task_type label：后续动作（comment/assign/edit/share）映射到 task taxonomy
	•	冷启动（无点击）
	1.	规则词典 + 关键短语模板（立刻可用）
	2.	从“全局跨租户”学一个通用 intent 模型：只用 query string + non-content metadata（这通常合规）
	3.	用 LLM 做弱标注生成训练集（但输出必须 constrained），再训练小模型上线（LLM 只在离线）
```
⸻

Step 2：Query Semantics Parsing（语义结构化 / 约束抽取）

目标

从 query 抽取可执行的 constraints / slots，比如：

	•	时间：last week / Q3
	•	人/团队：my manager / Alice
	•	项目：Phoenix
	•	关系：blocking / owned by
	•	doc type：meeting notes / roadmap
	•	这一步输出的是 WHAT constraints（不是 HOW）

典型输出

```
{
  "constraints": {
    "time_window": "2025Q3",
    "relation": "blocking",
    "entity": {"type":"project", "value":"roadmap"}
  }
}
```

算法/实现

V1：规则 + 解析器

	•	时间解析器（quarter / relative time）
	•	关系词表（blocking/depends）
	•	entity 词表/字典（团队名、项目代号）+ fuzzy match

V2：LLM constrained parsing + fallback

	•	Prompt：只允许输出 JSON schema（slots）
	•	validator：schema 校验 + allowlist（字段只能来自预定义集合）
	•	fallback：LLM 失败就回到 V1 解析器

labels（怎么拿）

	•	很多 slot 本质是“可自动产生监督”：
	•	time_window：如果用户点开的是某个时间范围内的结果（更新/编辑时间集中在某范围），可作为弱监督
	•	person/entity：点击的结果 metadata 中包含 owner/assignee/author，可回填为弱标签
	•	也可以离线用 LLM 对历史 query 做结构化抽取当 pseudo-label

冷启动
	•	这一步非常适合先用 rule/LLM 生成 pseudo-label → 训练一个小 parser 模型
（因为它不依赖 click，多依赖 query 语言本身）

⸻

Step 3：Retrieval Planning（单跳/多跳/多源调用的执行计划）

目标

决定 HOW：

	•	query 是否能直接作为检索 anchor？
	•	是否需要先 resolve 一个 anchor entity（epic/page/person/project）再找目标对象？
	•	是否需要 multi-hop（graph expand / join / aggregation）？
	•	输出一个可执行 plan（步骤 + 工具调用）

典型输出（plan）

```
{
  "plan": [
    {"action":"FIND_ANCHOR", "target":"roadmap_page_or_epic", "sources":["CONFLUENCE","JIRA"]},
    {"action":"EXPAND_GRAPH", "relation":"blocking", "source":"JIRA", "depth":2},
    {"action":"FILTER", "time_window":"2025Q3"}
  ]
}
```

算法/实现

V1（最实用）：模板化 planner（pattern → plan）

	•	dependency/blocking → “anchor → graph expand → filter”
	•	“what did X do last week” → “resolve X → query activity logs → aggregate”
	•	“how to …” → “doc retrieval → QA”

V2（Agent-friendly）：LLM planner（constrained）+ plan validator

	•	LLM 产出 plan（JSON）
	•	validator 控制：
	•	step 数上限、工具白名单
	•	每步必须在 tenant scope + ACL scope
	•	预算控制（latency / cost）
	•	fallback：模板 planner

labels（怎么拿）

	•	plan 是否正确很难直接监督，但可以用 counterfactual / success proxy：
	•	计划执行后用户是否完成任务（next_action、无 reformulation、少返回 SERP）
	•	计划是否在预算内完成（latency / steps）
	•	离线：从历史 session 中挖“人类隐式 plan”：
	•	同一 query 反复改写、点击路径形成多跳链路 → 反推应该有 anchor

冷启动

	•	planner 基本可以完全靠 **模板 + LLM（离线/受控）**启动
	•	再逐步用线上数据做 “plan selection” 学习（multi-armed bandit 选 plan）

⸻

Step 4：Retrieval（真正取候选：hybrid + graph + filters）

目标

按 plan 执行，拿到候选集合（candidates），并带上足够信息给 ranking：

	•	候选内容（或片段/embedding）
	•	metadata（type、作者、时间、project、link）
	•	解释信息（来自哪步、哪条路径、为什么命中）

主要 retrieval primitives

	1.	Keyword / BM25 / Exact match（ES）
	2.	Semantic retrieval（embedding + ANN）
	3.	Structured retrieval（filters / SQL / metadata）
	4.	Graph traversal（issue links / dependency）
	5.	Fusion（把不同来源合并去重）

冷启动/无点击时的 retrieval

	•	先用 BM25 + embedding 是最稳的 baseline
	•	embedding 冷启动：
	•	直接用通用 encoder（e5, gte, etc.）+ tenant 内建 index
	•	或用 Atlassian 自己的 enterprise encoder（如果有）
	•	如果 tenant 数据完全新：
	•	用 doc metadata（title、labels、space、issue type）先补 recall
	•	允许用户显式 filter（UI 约束补数据）

⸻

Step 5：Ranking（把候选排序、融合、输出）

目标

把 Step4 的候选按 “对当前任务的 utility” 排序：

	•	人类用户：更看 top1/top3 精准（precision）
	•	机器/grounding：更看 recall（topK 可用且覆盖）

模型方案（你之前问的 pointwise/pairwise/listwise 都在这）

A. 冷启动/早期：heuristic ranker

	•	recency、authority、link distance、text match、source priority

B. 有行为数据后：learning-to-rank

	•	pointwise（回归/分类）、pairwise、listwise 都可
	•	features（常见、很可讲）：
	•	query-doc text similarity（BM25、cosine）
	•	metadata match（project、time、owner）
	•	graph features（distance to anchor、path type）
	•	popularity / freshness（tenant 内）
	•	user context（最近在看哪个 project/space）

labels 怎么拿（enterprise 常见）

	•	click / long click / dwell / return-to-SERP
	•	next_action（share/comment/create ticket/cite）
	•	reformulation rate（用户是否马上改 query）
	•	注意：因为人工不可看内容，这些行为信号是核心监督

冷启动（没点击怎么办）

	1.	合成训练数据（弱监督）
	•	用规则：如果 query 含 “blocking” 且结果是有 blocking edge 的 tickets → pseudo positive
	2.	跨租户通用模型（不吃内容、只吃统计/embedding）
	•	backbone 共享，tenant head 轻量适配（你之前的 multihead）
	3.	LLM judge 离线打分
	•	用 LLM 对 query-doc 相关性打一个 graded score（只在合规可用的内容范围内），用于训练初始化 ranker

⸻

哪一步最适合做成 Agentic Search？还是独立 track？

结论（你可以记这句）

Agentic search 最自然的落点是 Step 3（Retrieval Planning）+ Step 4（多步执行）。
它不是完全独立的 track，而是 “把 QU 的 planning 从 rule/template 升级成可迭代的 multi-step reasoning/execution”。

更具体一点：
```
Agentic 在系统里扮演什么角色？
	•	做计划：把复杂 query 拆成若干 retrieval/tool calls
	•	做执行：根据中间结果决定下一步（conditional branching）
	•	做反思：如果结果弱，自动 reformulate / broaden / narrow

哪些步骤不建议完全 agent 化？
	•	Step 1/2：更适合小模型/规则（快、稳、可控）
	•	Step 3：最适合 agent（需要多跳、依赖中间结果）
	•	Step 4：执行层可以“被 agent 调用”，但 retrieval primitives 本身应工程化（可监控、可预算）

所以是不是独立 track？
	•	产品/组织层面可能会有“agentic search track”
	•	系统层面它依赖 QU 的结构化输出与检索工具链，不是孤立系统
```

## 延续问题： agentic search
- ”那么agentic search其实更多是step3 也就是所谓的orchestration ？“
---
是的，你这个理解完全对，而且已经是“系统设计层级”的表述了 👍
我直接给你一个可以对外讲、不会被误解的结论版本。

⸻

一句话结论（你可以直接用）

Agentic search 本质上是 Step 3 的升级版：它把原本静态、规则化的 retrieval planning，升级成一个具备推理、条件执行和反思能力的 orchestration layer。

⸻

把这句话拆开（确保你“真的懂”）

```
传统（非 agentic）的 Step 3：Retrieval Planning
	•	输入：
	•	intent（找什么对象）
	•	parsed constraints（时间、关系、实体）
	•	输出：
	•	一个固定 plan
	•	单跳 or 多跳
	•	用哪些 retrieval primitive
	•	顺序已写死
```
👉 本质是 rule / template driven orchestration

⸻

Agentic Search = Step 3 + 推理能力
```
Agentic search 在 “orchestration”这件事上，多了三件关键能力：

1️⃣ Reasoning（规划）
	•	根据 query 和中间结果
	•	决定：
	•	要不要 multi-hop
	•	先找 anchor 还是直接搜
	•	是否需要 broaden / narrow

2️⃣ Conditional Execution（条件执行）
	•	根据中间 retrieval 结果决定下一步
	•	如果 anchor 太多 → refine
	•	如果没结果 → fallback strategy
	•	如果结果置信度低 → reformulate

3️⃣ Reflection / Iteration（反思 & 迭代）
	•	判断当前结果是否“足够好”
	•	决定是否：
	•	再执行一步 retrieval
	•	换 strategy
	•	或直接返回
```
👉 这些全部发生在 Step 3 的范畴内

⸻

为什么说是 orchestration layer？

```
因为 agent 不直接做 retrieval / ranking，而是：
	•	调用：
	•	搜索服务
	•	向量索引
	•	graph service
	•	profile / org graph
	•	管理：
	•	顺序
	•	条件
	•	预算（latency / cost）
	•	失败兜底
```
这正是 orchestration 的定义。

⸻

用一个你可以直接讲的结构图（脑中）
```
User Query
   ↓
[Intent + Parsing]
   ↓
[Agentic Orchestrator]  ← 这就是 Step 3
   ├─ call semantic retrieval
   ├─ call graph expansion
   ├─ call filters
   ├─ check confidence
   ├─ decide next step
   ↓
Candidates
   ↓
Ranker / Answerer
```

⸻

重要澄清（避免面试踩坑）

❌ Agentic search ≠ 全流程都是 LLM
	•	Step 1 / 2：
	•	小模型 + rule 更稳定
	•	Step 3：
	•	agent（LLM）只负责 plan
	•	Step 4 / 5：
	•	仍然是工程化 retrieval / ranker

👉 这正是 Principal 会强调的“边界感”

⸻

再给你一句非常加分的话（强烈建议记）

“I see agentic search as an orchestration problem rather than a retrieval problem. The agent reasons about which tools to call and in what order, while retrieval and ranking remain deterministic, observable, and optimizable components.”

⸻

总结一句话（最终版）

Yes—agentic search mainly lives in Step 3. It replaces static retrieval planning with a reasoning-based orchestration layer that coordinates multiple retrieval strategies under constraints like latency, cost, and privacy.

如果你愿意，下一步我可以直接 mock 一个“Agentic retrieval planning”的面试追问，或者帮你把这一段压成 30 秒回答版本。

