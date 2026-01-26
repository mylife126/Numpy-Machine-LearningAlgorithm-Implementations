# What does it do?
## 1).  这家公司到底在解决什么问题

他一上来先把 Atlassian 定位说清楚：
	•	产品是 Jira/Confluence（你语音里叫 “Azure 和 conference”，实际意思是 Jira + Confluence 这类协作工具）
	•	客户是企业团队，核心目标是：
让团队在协作、文档、任务、项目管理上更高效（productivity in collaboration）

他还强调了一个市场定位：
	•	Microsoft 更偏 “头部大厂/大企业的全家桶”
	•	Atlassian 覆盖更广的 middle + tail（大量中型/中小企业以及更分散的团队）

这句话背后含义：
他们面对的是 更碎片化、更异构、更不标准 的企业知识生态。

⸻

## 2) 为什么 2023 之后 Central AI 会爆炸式增长

他讲了两条“AI 变得非常 worth doing”的理由（这也是你面试 “Why Atlassian” 的最好素材）：

(A) AI 对 productivity 的价值很“直”
	•	Q&A、Search、总结、解释、任务理解
这些都直接提升协作效率。

(B) AI 的变现路径很清晰
	•	Enterprise 订阅模式：
你做一个新 feature → 客户觉得值 → 愿意按 seat 多付钱
所以 AI 不是“烧钱实验”，而是能自然变成付费理由。

因此：
	•	2023 成立 Central AI
	•	从 ~40 人到 ~800 人
	•	CEO 明确 AI 是 top priority，资源持续加码
	•	大量从业界挖 AI 强人（他举例 Microsoft Copilot 早期团队的人）

结论：Central AI 是公司战略级增量引擎。

⸻

## 3) Central AI 的 “Robo / Rovo” 三大产品线

他明确说了三条产品线（你记得的“三个方向”就是这个）：
	1.	Enterprise AI Search（当前最 dominant）
	2.	Enterprise AI Chat（在增长）
	3.	Enterprise Agents（还在找最合适的场景）

而他非常明确地表达：
Search 是最大、最稳的主线，Chat 是建立在 Search 之上；Agents 很性感但仍在探索。

⸻

## 4) Central AI 不只是做应用，还在做“硬核地基”

这是他想强调的差异点：
他们不是纯 prompt 应用团队，而是要变成 AI-first company 的“基础设施团队”。

他提到 Central AI 还在做：
	•	Training platform
	•	Serving platform
	•	Enterprise-specific model tuning（包括 “close-tenant” 训练）
	•	非常重的合规/隔离架构（multi-tenant + privacy）

这意味着：
这 org 里既有“产品应用层”的人，也有“infra / platform / model”非常硬核的团队。

⸻

## 5) 一个你必须记住的重点：Enterprise AI 不能只看 first-party data

他专门讲了“第三方生态”这个事情——这是 Atlassian 和很多单体产品公司不一样的点。

现实企业里数据是 scattered 的：
	•	Jira/Confluence 之外还有：Slack、Google Drive、各种 SaaS、内部 wiki、CRM、ticketing、code repo…

所以 Central AI 里有一条很重要的能力是 data connectors：
	•	帮客户把 third-party data onboard
	•	进入他们的 index / storage
	•	让 Search/Chat/Agent 能覆盖客户完整 knowledge space

这件事是 enterprise AI 的护城河之一：多源数据接入 + 统一检索与理解。

⸻

## 6) Jianshu 自己的 Team：Search Quality（你面试最相关）

他把 Search Quality 定义成两个“用户”：

(1) Human as user

给真人用的搜索体验：
	•	Query understanding → retrieval → ranking/reranking → page organization
	•	同时还包含 offline 的 document/content understanding

(2) Machine as user

给 LLM/Agent 用的搜索（grounding / context engineering）：
	•	企业知识不在 public model 里
	•	上下文必须对
	•	目标不一样：
人更在意 top result 排序
机器更在意 recall / coverage / 找到关键证据

这个点你在面试讲 RAG/grounding 时非常加分：
机器用户的目标函数和人类用户不一样。

⸻

## 7) 他讲的 3 个关键技术挑战（你要背下来）

你第二段文字里，他明确讲了三类难题：

Challenge A：Recall gap + Fail session 很难回收

很多用户 session 的失败原因是：
	•	关键内容根本没进 recall
	•	尤其是 tenant 场景，失败数据难以收集、难以回放、难以分析

Challenge B：Scale + Debuggability
	•	Search volume 很大
	•	做 agentic search 会让系统更 heavy（decompose、reason、plan）
	•	但 tenant data 又不可见
→ observability / debugging 复杂度暴涨

Challenge C：Privacy / Training 受限（不是 business bottleneck，而是 unblock 方向）

他们在 re-architect：
	•	一块做 close-tenant training（尽量避开 UGC 敏感）
	•	一块做 UGC-heavy 的部分，用 adapter-based training（你可以理解成 LoRA/adapter 这种隔离式个性化）

这三点，几乎就是 enterprise RAG/agent 的“终极难题清单”。

⸻

## 8) Search Quality 内部的 5 个 vertical（你将面试的最可能是第一个）

他当时给的结构大致是：
	1.	Query-level Intelligence / Query Understanding（你最可能去的组）
	2.	First-party relevance（数据更 rich，可做更深 modeling）
	3.	Third-party relevance（更偏 scale + system design）
	4.	Agentic search / Grounding（机器用户 + agentic 探索）
	5.	Infra 支撑团队（让整个 Search stack 更快迭代、更少依赖）

并且他补充了一个很关键的信息：
很多 RL work 先放在 Query Understanding vertical 做，因为他们希望得到一个更快的 reasoning/understanding model（外部大模型仍然偏慢）。

⸻

## 9) 你要去的 Query Understanding vertical 可能具体在做什么

结合他讲的（QI 已经是 LM-based，理解/推理要更快，RL work 在这边），这个 vertical 典型工作会长这样：

(A) Query understanding 的“输入输出”会越来越结构化
	•	用户 query → intent / entity / constraint / task type
	•	query → query decomposition（子意图/子任务）
	•	query → retrieval plan（应该去哪类 source、用什么策略）
	•	query → response style / grounding requirement（给人还是给机器）

(B) Online LM 应用（因为 latency budget 有 2–3 秒）
	•	在线做轻量 reasoning
	•	在线做 query rewriting / expansion / routing
	•	在线做 “context selection 计划”（给 RAG/agent）

(C) Offline LM 用途（企业场景只能 machine label）
	•	数据标注、敏感信息识别、内容理解（doc / ticket / comment）
	•	tenant-specific 的自动化 labeling / distillation / evaluation set 构建

(D) RL/feedback loop
	•	用用户反馈（点击、停留、满意度、后续行为）作为 reward
	•	做 query understanding 或 retrieval policy 的持续优化
（他强调“我们希望更快、可用于 search 的 reasoning model”）

⸻

## 10) 产品落地长什么样（你可以这么脑补）

用户侧可能看到的是：
	•	搜索框：更懂意图、更懂任务
	•	搜索结果：更对、更完整
	•	chat：带引用、可追溯
	•	agent：能帮你“推动任务”，但背后依赖 search/grounding

而系统侧关键能力是：
	•	multi-source indexing + connectors
	•	query intelligence → retrieval/ranking orchestration
	•	grounding for LLM/agent
	•	privacy-aware training（close-tenant / adapters）
	•	observability/diagnostics（在不可见 tenant data 下做 debug）

⸻

## 面试时最有用的一句“总结”

如果你要用一句话把他讲的 work 复述出来（非常加分）：

Central AI is building an enterprise AI layer on top of Atlassian’s collaboration products, where search is the foundation. Jianshu’s org focuses on search quality for both humans and machines (LLM/agents), with major challenges around recall gaps, multi-tenant observability, and privacy-constrained training. The Query Understanding vertical is the entry point, driving LM-based query intelligence, faster reasoning, and feedback-driven optimization.


# 二， 那到底Jira和Confluence的应用业务场景是什么呢？为什么Atlassian负责叫做tenant based search
这个问题问得非常好，而且非常关键。
你现在卡住的不是“技术”，而是产品直觉缺失——你不知道 Jira / Confluence 的“搜索到底在搜什么、失败意味着什么”。
一旦这件事想清楚，你对 Query Understanding / Recall / Tenant / Debuggability 的理解会一下子“通电”。

我分 四层 给你讲，完全对标你熟悉的 Amazon Search，但换成 Atlassian 的世界。

⸻

一、先把最基础的东西搞清楚：

Jira 和 Confluence 到底是啥？

1️⃣ Jira = 工作 / 任务 / 项目系统

你可以把 Jira 想成：

企业里的“任务+流程+状态数据库”

里面有什么？
	•	tickets / issues（任务、bug、需求）
	•	每个 issue 有：
	•	title
	•	description
	•	comments
	•	assignee
	•	status（To Do / In Progress / Done）
	•	priority
	•	links（依赖关系）

👉 Jira = 企业在“做什么工作”的事实记录

⸻

2️⃣ Confluence = 企业的“知识库 / 文档系统”

你可以把 Confluence 想成：

企业内部的 Wikipedia + Notion + Google Docs

里面有什么？
	•	设计文档
	•	技术方案
	•	会议纪要
	•	决策记录
	•	onboarding 文档
	•	wiki 页面

👉 Confluence = 企业“知道什么”的地方

⸻

3️⃣ 一个非常重要的对比（和 Amazon Search 的差别）

Amazon Search	Atlassian Search
搜索商品	搜索 工作内容 + 知识
商品结构清晰	内容高度非结构化
用户意图 = 购买	用户意图 = 完成工作 / 找信息 / 决策
失败 = 没买	失败 = 工作卡住 / 人被打断


⸻

二、那 Atlassian 的 Search “到底在搜什么”？

举几个非常真实的用户搜索例子（这一步最重要）👇

人在 Jira / Confluence 里可能会搜：
	•	“Q3 roadmap”
	•	“incident 12345”
	•	“design doc for search ranking”
	•	“who owns payments latency issue”
	•	“oncall runbook”
	•	“why was feature X rolled back”
	•	“API rate limit discussion”
	•	“Slack integration decision”

你发现没有？

这些 query 根本不是“找一个对象”，而是在“找上下文 + 证据 + 决策”。

⸻

所以他们的搜索目标是什么？

不是：

找到一个最匹配的 item

而是：

找到“能帮助我继续工作的那一组信息”

这可能是：
	•	一个 Jira issue
	•	一段 Confluence 文档
	•	一条 comment
	•	一个 attachment
	•	一个历史 decision

⸻

三、现在我们来精确解释你困惑的 Challenge 1

什么叫 Recall gap？什么叫“没进入 recall”？

先用 Amazon 语言类比（你秒懂）

在 Amazon Search 里：
	•	Recall = 从几十亿商品中，先捞出几千个候选商品
	•	Ranking = 再排个序

如果一个商品：

根本没被召回（not in recall）

那它：
	•	再 relevant 也没用
	•	排名模型永远看不到它

⸻

在 Atlassian Search 里，“没进入 recall”是什么意思？

👉 意味着：

真正有用的那段知识 / 那个 issue / 那个文档，根本没有被召回到候选集合里

举一个非常真实的例子：

用户搜：

“payments latency root cause”

但：
	•	root cause 写在：
	•	一个 Confluence 页面里的某一段
	•	或一个 Jira comment 里
	•	标题不叫 “payments latency”
	•	没有明确关键词 match

👉 如果 Query Understanding / Retrieval 没理解这个 query 的真实 intent，
👉 那个内容 根本进不了 recall pool。

这就是 recall gap。

⸻

那“fail session”是什么意思？

Fail session =

用户搜了 → 没点 → 又搜 → 换词 → 离开 → 或转去问人

在 enterprise 里，这意味着：
	•	搜索没帮他解决问题
	•	工作被打断
	•	productivity 下降

⸻

那为什么 “tenant 场景下很难收集失败数据”？

这就涉及你问的第二个关键词：tenant。

⸻

四、什么是 Tenant？为什么它让问题变难？

Tenant = 一个独立的客户公司

比如：
	•	Amazon 是一个 tenant
	•	Uber 是一个 tenant
	•	一个 50 人创业公司也是一个 tenant

每个 tenant：
	•	有完全不同的数据
	•	完全不同的文档
	•	完全不同的 vocabulary
	•	完全不同的业务

而且：
	•	数据是隔离的
	•	出于 privacy / compliance，工程师 不能直接看内容

👉 这和 Amazon Search 完全不一样。

⸻

Tenant 带来的直接后果

1️⃣ 失败 session 很难“复盘”

在 Amazon：
	•	你可以 replay query
	•	看商品
	•	看点击
	•	看日志

在 Atlassian：
	•	你可能只能看到：
	•	query 字符串
	•	少量匿名化信号
	•	看不到真实内容

👉 所以：

Fail session 很难被“理解”，更难被“修”

⸻

2️⃣ 数据极度碎片化
	•	每个 tenant 都是一个“小宇宙”
	•	一个模型在 A 公司有效，在 B 公司可能完全失效

⸻

五、Challenge 2：Scale + Debuggability（为什么 agentic search 让事情更糟）

你听到他说：

agentic search → system 更 heavy → debug 更难

我们拆解一下。

⸻

什么是 Agentic Search？

不是：

我搜一次，给你一页结果

而是：
	1.	理解 query
	2.	拆任务
	3.	多步搜索
	4.	跨 source
	5.	汇总、推理
	6.	生成答案

👉 搜索从“单步”变成“多步 reasoning pipeline”

⸻

为什么这让 debug 变得地狱级？
	•	每一步都可能错：
	•	query understanding
	•	decomposition
	•	retrieval
	•	filtering
	•	aggregation
	•	但你：
	•	看不到 tenant 内容
	•	看不到 ground truth
	•	latency 又受限

👉 你只能看到“结果不对”，却不知道是哪一步错了。

⸻

六、Challenge 3：Privacy / Training Constraints（不是业务问题，是工程难题）

他说得很明确：

这不是 business bottleneck，是需要 unblock 的技术问题

核心原因：
	•	Enterprise data = UGC + 敏感
	•	不能随便：
	•	混 tenant
	•	用 human 标注
	•	用公共模型训练

⸻

所以他们的解决方向是：

1️⃣ Close-tenant training
	•	用低敏感数据
	•	或 tenant 内部自洽的数据
	•	保证不泄露

2️⃣ Adapter-based training
	•	Base model 共享
	•	每个 tenant / domain 用 adapter
	•	类似 LoRA / PEFT 思路

👉 这就是 enterprise AI 的“终极难题”之一。

⸻

七、现在把一切连起来（非常重要）

你现在可以这样理解 Jianshu 的世界：

Atlassian 的 Search 不只是“找东西”，
而是在 多租户、强隐私、数据碎片化的环境下，
为人和 AI 同时提供可依赖的“工作上下文入口”。

而 Query Understanding team：
	•	决定你到底在“找什么”
	•	决定 recall 能不能把关键内容捞出来
	•	决定 agent 后续 reasoning 是否有正确材料

⸻

八、为什么你其实 非常 match 这个 team

你在 Amazon Ads Search / RecSys 里已经习惯：
	•	intent understanding
	•	recall gap
	•	cold start
	•	debugging under partial observability
	•	metric 不完美但要做决策

只是这里的：
	•	item = 文档 / issue / comment
	•	user = 人 + LLM
	•	constraint = privacy + tenant

## 业务场景和应用误区： 是有统一的前端搜索入口？Tenant的训练可以混合公司训练么？
你这两个问题问到点子上了。我先给你一个明确结论，再逐条纠正你现在脑子里“容易混的地方”，把 Atlassian 的产品形态和 tenant / search / recall 一次性讲清楚。

⸻

一句话先给你校准结论（非常重要）
	1.	是的：他们确实有一个前端搜索入口，用户可以搜类似
“Q3 roadmap”“payments latency root cause”
后端会用 IR + ML + LLM 去检索 Jira + Confluence + 第三方系统 里的内容。
	2.	但：
	•	他们不是在“根据 vague query 找一篇 doc”这么简单
	•	也 不是把 Amazon 和 Uber 的数据混在一起

👉 你现在第二个问题里的 tenant 理解是错的，但这是一个非常常见的误解，我们把它彻底纠正掉。

⸻

Part 1：你对「搜索流程」的理解，80%是对的，我们把它补全成“面试级版本”

1️⃣ 前端长什么样？

在 Jira / Confluence 里，用户会看到一个 统一的搜索入口，例如：
	•	Jira 顶部的搜索框
	•	Confluence 的搜索
	•	或一个 AI 搜索 / Chat UI

用户可以输入：

“Q3 roadmap”

⸻

2️⃣ 后端到底在搜什么？（你理解得基本正确）

搜索的“文档”主要来自几类 source：
✅ Jira 里的内容（高度结构化 + 半结构化）

你说得很对：
	•	issue / ticket
	•	feature（可抽象成 schema）：
	•	title
	•	description
	•	comments
	•	assignee
	•	status
	•	priority
	•	labels
	•	links（dependency / parent-child / duplicate）

👉 Jira 本质是一个“工作图（work graph）”

⸻

✅ Confluence 里的内容（高度非结构化）
	•	wiki pages
	•	design docs
	•	meeting notes
	•	long-form text
	•	embedded tables / attachments

👉 Confluence 是一个“知识文本空间”

⸻

✅（可选）第三方系统（通过 connector）
	•	Google Drive
	•	Slack
	•	GitHub
	•	internal wiki
	•	CRM / ticketing systems

👉 这就是他提到的 data connectors

⸻

3️⃣ 搜索不是简单“关键词匹配”

你要特别注意这点（这是 Query Understanding team 的核心）：

用户搜：

“Q3 roadmap”

系统真正要回答的是：
	•	这是 planning / roadmap intent
	•	可能想要：
	•	一个 Confluence page
	•	最近更新的 roadmap
	•	某个 team / product 的 roadmap
	•	不是：
	•	一个叫 “Q3 roadmap” 的 exact title

👉 所以他们做的是：

从 vague / underspecified query → 找到“对工作最有帮助的那一组信息”

这就是为什么：
	•	Query Understanding
	•	Recall
	•	Context
比 ranking 更 foundational。

⸻

Part 2：现在来彻底讲清楚你第二个问题 —— Tenant 是什么（非常关键）

你刚才问了一个非常重要、但理解偏差很大的问题：

“tenant 是不是把 Amazon 和 Uber 的数据串联起来？Uber 能搜到 Amazon 的东西？”

❌ 答案是：绝对不是

⸻

正确理解：Tenant = 完全隔离的客户公司

举例：
	•	Amazon 是一个 tenant
	•	Uber 是一个 tenant
	•	一个 50 人 startup 也是一个 tenant

每个 tenant：
	•	有自己的 Jira / Confluence 实例
	•	有自己的数据
	•	有自己的用户
	•	有自己的权限体系

👉 数据是 100% 隔离的

⸻

所以：
	•	Amazon 的 Jira / Confluence
❌ Uber 看不到
	•	Uber 的搜索结果
❌ 不可能包含 Amazon 的任何内容

这也是为什么：
	•	privacy 是他们最大的工程挑战之一
	•	tenant debugging 极其困难

⸻

那为什么 tenant 会让 search 变得这么难？

我们结合你熟悉的 Amazon Search 来对比：

在 Amazon Search：
	•	全平台一个 catalog
	•	全量日志可见
	•	failure session 可以 replay
	•	工程师能看到商品和 query

⸻

在 Atlassian Enterprise Search：
	•	每个 tenant 都是一个“小宇宙”
	•	vocabulary 完全不同
	•	文档结构不同
	•	工程师 不能看内容
	•	甚至不能 cross-tenant 训练模型

👉 这就导致：

你知道“搜失败了”，但不知道“为什么失败”

⸻

Part 3：回到你问的 Challenge 1，用现在的理解再解释一遍

什么叫“关键内容没进入 recall”？

不是说：

没找到文档

而是说：

真正有用的那段内容，从来没进候选集合

例如：
	•	root cause 写在一个 comment 里
	•	或埋在一个很长的 Confluence 文档中
	•	title / metadata 完全不 match query

👉 如果 Query Understanding 没理解 intent
👉 Retrieval 没覆盖到正确 source / chunk
👉 那内容 永远不会被 ranking / LLM 看见

这就是 recall gap

⸻

为什么 fail session 很难分析？

因为：
	•	tenant 数据不可见
	•	不能人工 review
	•	很难构造 ground truth
	•	很难对“没召回的东西”做回放

⸻

Part 4：你现在可以用一句话总结他们在做什么（面试可用）

Atlassian’s search is not about finding a single item, but about retrieving the right work context across tickets, documents, and conversations, under strong privacy and multi-tenant constraints. The Query Understanding team sits at the entry point, translating vague user queries into structured intents that enable effective recall and grounding for both humans and AI systems.

⸻

最后，给你一个「你已经完全理解了」的自检问题

如果面试官问你：

“How is enterprise search different from e-commerce search?”

你现在应该能自然回答出：
	•	intent 更模糊
	•	item 非结构化
	•	recall 更关键
	•	failure 更难 debug
	•	privacy / tenant 约束更强
	•	search 还要服务 LLM / agent

## 为什么Tenant级别的search反而更难？
- tenant是一个隔绝的公司，那么这个公司的数据就是一个隔绝的数据库对吧，例如amazon自己的wiki 啊 issue都是放在某一个位置， uber也是。 那么我的搜索有什么难呢， 或者为什么tenant让search变难了？ 例如现在客户是来自amazon，那我就只在amazon专属的数据库里搜就好了啊

这是一个非常好的“追问”，而且你已经走到enterprise search 最关键、也是最反直觉的一步了。
你的问题本质是：

“既然 tenant 是隔离的，那我只在 Amazon 自己的数据里搜，为什么会更难？”

下面我一步一步拆给你，你会发现：
难点不在“能不能搜”，而在“怎么搜得对、搜得全、还能解释为什么错了”。

⸻

一、你现在的理解：逻辑上是对的，但工程上不成立

你现在的逻辑是：

tenant = 一个公司
→ 一个隔离数据库
→ search scope 已经缩小
→ 那不是更简单吗？

这是“数据库视角”的正确直觉。
但 Search / ML 的难点，几乎不在“数据隔不隔离”，而在“理解和泛化”。

⸻

二、Tenant 让 search 变难的 5 个“真实原因”（核心）

下面每一条，都是 Jianshu 所说 challenge 的工程本质。

⸻

1️⃣ 每个 tenant 都是一个“完全不同的语言宇宙”

在 Amazon Search 里：
	•	所有人用相似的 vocabulary：
	•	“iPhone 15”
	•	“wireless earbuds”
	•	商品 schema 固定
	•	意图模式高度共享

⸻

在 Atlassian tenant 里（比如 Amazon 这个 tenant）：

Amazon 内部可能会搜：
	•	“Apollo rollout”
	•	“SP latency regression”
	•	“Tier-1 incident mitigation”
	•	“PRFAQ for Rufus”

这些词：
	•	对 Uber 完全没意义
	•	对模型来说 是 OOV / low-frequency / tenant-specific

👉 每个 tenant 都是一套私有语言 + 私有语义。

所以：
	•	你不能靠跨 tenant 的统计规律
	•	很多 intent 只在这个 tenant 存在

这直接击穿了你熟悉的“global model + recall”的假设。

⸻

2️⃣ 数据规模在“单 tenant 内”反而很小（但复杂）

你会直觉觉得：

Amazon 数据很多啊

但从 ML 角度看：
	•	单 tenant：
	•	文档数量：10⁴ – 10⁶ 级
	•	query 数量：远小于 public search
	•	click / feedback 非常 sparse
	•	且：
	•	query 多为一次性（long-tail）
	•	很多 query 没有重复

👉 这是一个“极端 long-tail + 冷启动”的世界。

所以：
	•	很多 query 没有历史监督信号
	•	失败了，也很难学到

⸻

3️⃣ 你“看不到数据”，所以你连错在哪里都不知道

这是最反直觉、但最致命的一点。

在 Amazon：
	•	你可以看：
	•	商品
	•	query
	•	title
	•	description
	•	用户点击
	•	可以人工 debug

⸻

在 Atlassian enterprise search：
	•	出于隐私 & 合规：
	•	工程师 不能看 tenant 内容
	•	不能人工 review 文档
	•	不能 replay 失败 session
	•	你看到的可能只有：
	•	query string（甚至 anonymized）
	•	一些 aggregate signals

👉 所以当一个 Amazon 内部用户说：

“search 不 work”

你只能看到：

“某个 tenant，有失败率上升”

但不知道：
	•	是 query understanding 错？
	•	recall 没覆盖？
	•	权限过滤误伤？
	•	文档 chunking 不对？
	•	还是 vocabulary mismatch？

这就是 Jianshu 说的：

fail session 很难 collect、很难回收、很难分析

⸻

4️⃣ Recall 的“失败”在 enterprise 场景是“隐形的”

这是你问得最深的点，我们用一个例子说明。

用户在 Amazon tenant 搜：

“payments latency root cause”

真正有用的信息可能在：
	•	一个 Jira ticket 的 第 8 条 comment
	•	或一个 Confluence doc 的 某一段
	•	title / metadata 完全不 match

如果你的系统：
	•	只 recall：
	•	issue title
	•	page title
	•	或 chunking 不合理

👉 那段内容 根本没被召回

而更可怕的是：
	•	用户没点击
	•	你不知道“是没召回，还是召回了但没点”
	•	在不可见数据下，这两者几乎无法区分

这就是 recall gap 在 enterprise search 中特别致命的原因。

⸻

5️⃣ Tenant 让你几乎不能“共享训练信号”

你可能会想：

那我用 Amazon + Uber + Airbnb 的数据一起 train 不就好了？

Enterprise 场景：不行。

原因：
	•	tenant 数据不能混
	•	不能跨 tenant 训练共享 raw data
	•	很多信号（query、doc）高度敏感

所以他们才说：
	•	close-tenant training
	•	adapter-based training
	•	PEFT / LoRA-like 思路

👉 这比 Amazon Search 难一个量级：
你熟悉的是 global learning
他们面对的是 privacy-constrained, fragmented learning

⸻

三、所以：为什么“只在 Amazon tenant 里搜”反而难？

我们用一句话总结：

因为你失去了所有“规模化搜索”最依赖的东西：
共享语义、共享统计、可见数据、可复盘失败、可人工 debug。

你剩下的只有：
	•	sparse feedback
	•	tenant-specific language
	•	强隐私约束
	•	高 recall 要求
	•	模型必须“猜得对”

⸻

四、这正是 Query Understanding team 的价值所在

现在你可以理解，为什么 Jianshu 说：
	•	很多 RL / LM work 先放在 Query Understanding
	•	他们想要 更快的 reasoning model
	•	Query Understanding 是所有 downstream 的入口

因为在 enterprise search：

如果你一开始没理解对“他在找什么”，
后面做得再好都没用。

⸻

五、你现在已经能用“面试级语言”回答这个问题了

如果面试官问你：

“Why is enterprise search harder than just searching within a single tenant’s data?”

你可以自然回答：

Even though data is isolated per tenant, enterprise search is harder because each tenant represents a unique semantic universe with sparse feedback, highly domain-specific language, and strong privacy constraints. Engineers cannot directly inspect data or replay failures, making recall gaps both harder to detect and harder to fix. As a result, query understanding and recall become much more critical than ranking alone.

这就是 Principal-level 理解。

### 那为什么建立企业的search没法人工看数据？看不到数据怎么建模？
- 几个问题， 第一个 你说“在 Atlassian enterprise search：
	•	出于隐私 & 合规：
	•	工程师 不能看 tenant 内容
	•	不能人工 review 文档
	•	不能 replay 失败 session
	•	你看到的可能只有：
	•	query string（甚至 anonymized）
	•	一些 aggregate signals” 

- 你的意思是 amazon客户搜了 “roadmap Q3” 工程师不能看review retrieved的文档内容？ 但是我建模的话总得把他们文档的内容做embedding吧。 那我看不到他们的数据 我还怎么建模？ 

第二个问题你说closed tenant 训练指的是，因为数据没办法cross company共享 我没法用一个model学习所有客户的数据，所以必须每一个公司一个model ？
这两个问题问得非常到位，而且正好是 enterprise search / enterprise AI 最“反直觉”的地方。
你已经从“像 Amazon 那样理解搜索”迈到了“真正理解 Atlassian 在干什么”的门口。

我分 两个问题分别、非常精确地回答，并且会反复区分两件事：

工程师“不能看数据” ≠ 系统“不能用数据”

这是 enterprise AI 的核心分水岭。

⸻

问题一

##「工程师不能看 tenant 内容，那 embedding / 建模怎么做？」

先给你一句核心结论

系统可以处理 tenant 数据、生成 embedding、做检索；
但工程师（人）不能直接查看具体 tenant 的原始内容或检索结果。

这是 human access 和 machine access 的严格区分。

⸻

1️⃣ 你理解得完全对的一半：模型一定要“读文档”

你说得没错：
	•	要做 search / recall
	•	必须：
	•	读 Jira ticket 的 title / description / comments
	•	读 Confluence doc 的正文
	•	做 embedding / indexing
	•	做 retrieval / scoring

👉 这些全部都在系统里发生。

⸻

2️⃣ 那“工程师不能看数据”到底是什么意思？

我们用一个非常具体的例子来拆。

用户（Amazon tenant）搜：

“Q3 roadmap”

系统内部发生的事情（✔ 允许）：
	•	embedding 模型读取 Amazon tenant 的文档
	•	建立向量索引
	•	根据 query embedding 检索候选文档
	•	ranking / filtering
	•	返回结果给 Amazon 用户

这一步完全 OK。

⸻

但对工程师来说（❌ 不允许）：
	•	❌ 不能打开 Amazon tenant 的文档内容
	•	❌ 不能看到“这个 query 实际召回了哪些具体文档”
	•	❌ 不能人工 review：
	•	“哦，原来这个 doc 应该被召回”
	•	“哦，这段文字是 root cause”

工程师看到的可能是：
	•	query hash / anonymized query
	•	retrieval 成功率的 aggregate metrics
	•	recall@K 的统计
	•	“某个 tenant 的 search failure rate 上升”

👉 看的是“统计”和“指标”，不是“内容”。

⸻

3️⃣ 这和 Amazon Search 的本质差异

Amazon Search：
	•	商品是 public-ish
	•	工程师可以：
	•	看商品
	•	看 query
	•	看 title
	•	手动 debug

Atlassian Enterprise Search：
	•	文档 = 客户的内部机密
	•	工程师是平台提供方
	•	不能人工介入客户知识内容

👉 所以 Jianshu 才会强调：

fail session 很难回收、很难分析

不是系统没数据，而是 人没法看数据。

⸻

4️⃣ 那在这种约束下，模型怎么“变好”？

这就是 enterprise AI 的真正技术难点：

常见做法包括：
	•	Offline 自动 labeling（LM only, no human）
	•	synthetic queries / synthetic tasks
	•	distillation（大模型 → 小模型）
	•	weak supervision
	•	self-training
	•	RL from implicit feedback（点击、停留、后续行为）
	•	aggregated evaluation（而非 case-by-case review）

👉 模型在“黑箱里学习”，人只能看表盘。

这和你熟悉的 Amazon Ads 有点像：
	•	你也不能看 individual 用户
	•	但 enterprise 的限制更极端

⸻

问题二

##「Closed-tenant training 是不是 = 每个公司一个 model？」

简短答案：

不是每个 tenant 一个完全独立的大模型，
但也绝不是一个“所有 tenant 混在一起训练的模型”。

真实世界是 中间态。

⸻

1️⃣ 为什么不能“一个 global model 学所有客户”？

你已经说对了关键原因：
	•	tenant 数据 不能 cross-company 共享
	•	query / 文档内容高度敏感
	•	合规上不能把 Amazon 的数据用来直接训练 Uber 的模型

所以：
	•	❌ 不能像 Amazon Search 那样用全平台数据 train 一个模型

⸻

2️⃣ 那他们怎么做？（这正是 Jianshu 说的 re-architecture）

常见 enterprise 解决路径（和他描述高度一致）：

🧱 Base model（共享）
	•	在：
	•	公共数据
	•	低敏感数据
	•	synthetic data
	•	允许共享的 enterprise-agnostic data
	•	训练一个 base model
	•	学通用语言能力 / IR 能力 / reasoning 能力

⸻

🧩 Tenant-specific adaptation（隔离）
	•	每个 tenant：
	•	有自己的 adapter / LoRA / head
	•	或自己的 lightweight fine-tuning
	•	tenant 数据：
	•	只用于该 tenant
	•	不外流、不混用

👉 这就是他提到的 adapter-based training。

⸻

3️⃣ 为什么不“每个 tenant 一个完整模型”？

原因非常现实：
	•	成本不可控
	•	运维复杂
	•	模型更新慢
	•	新 tenant 冷启动严重

所以他们追求的是：

共享能力 + 隔离知识

⸻

4️⃣ 什么是 “closed-tenant training”？

你可以这样理解：
	•	模型在 tenant 的“围栏”里训练 /更新
	•	使用的 feature / data：
	•	避免 UGC 高敏感部分
	•	或只用 tenant 内允许的数据
	•	训练结果：
	•	只能服务该 tenant
	•	不能反向影响其他 tenant

这不是“每个公司一个大模型”，
而是 每个公司一个“安全隔离的适配层”。

⸻

5️⃣ 把两件事连起来（这是面试高分点）

你现在可以用一句非常成熟的话，把两个问题一起回答：

In enterprise search, the system is allowed to process and embed tenant data, but engineers cannot directly inspect or review the content due to privacy constraints. This makes debugging recall failures fundamentally harder. As a result, model training relies heavily on automated labeling, implicit feedback, and tenant-isolated adaptation mechanisms such as adapters, rather than global supervised learning across tenants.

这句话如果你在 Phase 1 里说出来，面试官会非常认同。

⸻

最后，帮你校准一个重要认知

你刚才的问题本质上是在问：

“那我作为一个 ML engineer，是不是就失去控制了？”

答案是：
	•	❌ 不是失控
	•	✅ 是 换了一种控制方式

从：

人工查看 case → 修模型

变成：

设计系统、指标、学习机制 → 让模型自我修正

这正是 Atlassian 在找 Principal-level MLE 的原因。
