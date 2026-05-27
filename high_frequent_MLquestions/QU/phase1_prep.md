## ML craft high level面试大纲
非常好，你现在已经掌握了 80% 候选人完全没有的关键信息。接下来这一步不是“再学新东西”，而是把你已有的能力按 Atlassian ML Craft 的评分标准精准对齐。

我会按你要求的顺序来：

⸻

一、先“对齐认知”：什么样的回答会被认为是 成功的 ML Craft（Stage 1）

这轮面试在 Atlassian 内部的真实目标是👇

判断你是不是一个：

	•	能把复杂 ML 问题抽象清楚
	•	能在不完美约束下做正确 trade-off
	•	能把 ML、系统、业务、团队整合起来的 Principal 级别 IC

⚠️ 不是：

	•	刷算法
	•	背模型
	•	讲 fancy 名词

⸻

二、Stage 1 的官方结构（你记得是对的）

1 小时 = 2 个 30 分钟

Part A（30 min）：Past Project Deep Dive（ML Craft）

面试官在看什么？

你是怎么“做成一件 ML 事”的

隐含评分维度（非常重要）

	1.	Context framing
	•	业务是什么？
	•	为什么是 ML？
	•	不做会怎样？
	2.	Success metrics
	•	你如何定义成功？
	•	离线 vs 在线
	•	业务指标如何闭环
	3.	Technical depth
	•	模型选择理由
	•	特征 & 数据
	•	系统设计
	4.	Trade-offs
	•	latency vs quality
	•	recall vs precision
	•	complexity vs robustness
	5.	Leadership / ownership
	•	你做了哪些关键决策？
	•	如何影响他人？
	•	失败/反思

👉 这轮不是“讲项目”，而是“展示判断力”

⸻

Part B（30 min）：New ML Design Problem

Recruiter 给你的 hint 非常关键：

AI Agent 或 Recommendation System

结合 Atlassian 的业务，实际会变成：

	•	Agentic search
	•	Enterprise recommendation
	•	Jira / Confluence / Logs / Knowledge

⸻

三、你搜集到的面经信息是高度一致的（这是好事）

我们来统一抽象一下：

所有 Stage 1 的新问题，都有这些共性

	1.	问题很开放
	2.	不要求细节实现
	3.	看你如何 structure
	4.	是否能自然落到 enterprise 场景
	5.	是否能讨论 trade-offs & metrics

你给的 sample（Agentic logs system）是完全合格答案，甚至是 above bar。

⸻

四、你现在该怎么“回答才会成功”

我先给你一个 统一答题骨架（非常重要）
👉 不管是 recommendation / agent / search，都套这个

⸻

⭐ Atlassian ML Design 通用答题框架（你记笔记用）

Step 0：Clarify（1–2 min）

	•	用户是谁？
	•	用例是什么？
	•	online 还是 offline？
	•	latency / scale / privacy？

不要跳过这一步

⸻

Step 1：Reframe as ML problem

	•	Prediction？Ranking？Planning？
	•	输入 / 输出？
	•	哪些是 deterministic，哪些是 ML？

⸻

Step 2：High-level architecture（你已经很强）

通常是 2–3 层：

	1.	Interface / Agent layer
	2.	Retrieval / Recommendation / Data layer
	3.	Model / Infra layer
（Optional memory）

⸻

Step 3：核心模型设计

	•	为什么是 ranking / regression / policy？
	•	为什么不是 classification？
	•	数据从哪来？
	•	冷启动怎么办？

⸻

Step 4：Evaluation & Metrics（非常加分）

	•	Offline
	•	Online
	•	System-level
	•	Business-level

⸻

Step 5：Trade-offs & Risks（Principal 区分点）

	•	latency
	•	cost
	•	privacy
	•	debuggability
	•	failure modes

⸻

五、你给的 Sample Answer 的“官方解读”

你那个 agentic logs 例子为什么能过：

👍 做对的点

	•	分层清晰
	•	把 agent 当 orchestration，而不是 magic
	•	Retrieval 是第一等公民
	•	Metrics 有层次
	•	没陷入 prompt engineering

🔥 特别符合 Atlassian 的点

	•	enterprise logs
	•	org graph
	•	timestamp
	•	privacy-aware retrieval
	•	RAG 而不是 end-to-end

⸻

六、你即将被问到的 4 类问题（高概率）

接下来我们就按你说的：

“每一个问题一个一个来过”

我先帮你列出来，你确认，我们逐个精修。

⸻

问题 1（Past Project）

“Pick one ML project you led end-to-end.”

👉 我们要做的是：

	•	选 一个最能映射到 enterprise search / recommendation 的项目
	•	重构成 ML Craft Story

⸻

问题 2（Agent / Recommendation Design）

“Design an AI agent / recommendation system for X.”

👉 我们要准备：

	•	1–2 个 通用模板
	•	能适配 Jira / Confluence / Logs

⸻

问题 3（Trade-off 深挖）

“What would you optimize first and why?”

👉 这是 Principal 分水岭

⸻

问题 4（Failure & Learning）

“What went wrong?” / “What would you do differently?”

👉 必考

⸻

七、下一步我们怎么继续（按你要求）

你说得非常清楚：

“我要每一个问题一个一个来和你过一遍”

建议顺序（最稳）：

👉 Step 1（现在）

## 该讲什么不该讲什么
- “那你的意思是 在聊这一轮的时候 不能讲太多怎么训练model ？ 而是更应该根据场景的局限性去选择一个合理的ml modeling 然后直接聊怎么建模 
数据怎么处理 怎么 feature engineering 然后建模后怎么做system？”

### 第一
是的，你这个理解是 100% 正确的，而且这是这一轮能不能“过 bar”的关键差别。

我给你一个非常明确、可执行的判断标准，帮你在面试时随时校准自己有没有“讲对重点”。

⸻

一句话先给你定锚（一定要记住）

ML Craft 面试 ≠ Model Training 面试
它考的是：在真实约束下，你如何做“正确的 ML 决策”。

所以答案是：

✅ 可以讲模型
❌ 但不能沉迷于模型训练细节

⸻

一、为什么这一轮“不能讲太多怎么训练模型”

不是因为他们不在乎模型，而是因为：

1️⃣ 真实世界里，模型不是 bottleneck

在 Atlassian enterprise search / agent / recsys 场景中：

	•	数据受限（privacy / tenant）
	•	label 稀缺
	•	debug 困难
	•	latency & scale 约束强

👉 “能不能训练一个 fancy model”不是问题
👉 “在这些约束下还能不能把系统跑起来”才是问题

⸻

2️⃣ 这一轮面试官多半是：

	•	Hiring Manager
	•	Senior IC / EM
	•	不是 research 面

他们想听的是：

“如果我把这个问题交给你，你会怎么落地？”

⸻

二、那你“应该讲什么”？（你刚刚总结得非常准）

你说的这句话我直接帮你“定型”：

根据场景的局限性选择合理的 ML formulation，
然后重点讲数据、feature、系统、评估与 trade-offs。

这就是标准答案。

我帮你拆成一个黄金比例👇

⸻

三、ML Craft 的“黄金内容分配比例”

你可以在脑子里记成一个 100 分模型：

模块	占比	讲什么
```
问题建模（Formulation）	25%	为什么是 ranking / regression / policy
数据 & 特征	25%	数据从哪来、怎么用、限制是什么
系统设计	25%	online / offline / infra / scale
评估 & Trade-off	20%	metrics / failure / latency
模型训练细节	≤5%	只点到为止
```
👉 模型 = 工具，不是主角

⸻

四、正确 vs 错误的“模型讲法”对比（非常重要）

❌ 错误示范（很容易踩）

“我们用了 transformer，12 层，hidden size 768，用 AdamW，learning rate schedule 是……”

👉 面试官心里：
“这不是我想听的。”

⸻

✅ 正确示范（Principal 风格）

“Given sparse labels and privacy constraints, we framed this as a ranking problem with implicit feedback. We started with a lightweight model to meet latency requirements, and designed the system so that more expressive models could be introduced offline without blocking iteration.”

👉 面试官心里：
“这人懂现实。”

⸻

五、你该如何“讲模型而不讲训练细节”

这是一个非常高级但非常实用的技巧👇

讲这 5 件事就够了：

	1.	为什么选这类模型
	2.	模型输入是什么
	3.	输出是什么
	4.	模型在系统里扮演什么角色
	5.	如果 scale / data 变了，会怎么 evolve

完全不需要：

	•	optimizer
	•	loss 的数学公式
	•	learning rate

⸻

六、用你熟悉的 Amazon 场景做一个对照（帮你内化）

Amazon Ads / Search（你过去）

	•	大量 label
	•	可人工 review
	•	商品内容公开
	•	可以 heavy modeling

Atlassian Enterprise（他们）

	•	几乎没 label
	•	UGC 不可看
	•	tenant isolation
	•	系统复杂度高

👉 所以他们想要的是：

一个能在“弱监督 + 强约束”下 still make progress 的人

⸻

七、面试官真正想从你嘴里听到的一句话

你可以直接在脑子里记住这句话，它是 ML Craft 的“灵魂句”：

“The modeling choice was largely driven by data availability, privacy constraints, and system requirements, rather than model complexity.”

这句话一出来，你已经站在他们那一侧了。

⸻

八、给你一个实操建议（非常重要）

在你准备项目 deep dive 时：

每当你想讲“模型训练细节”时，问自己一句：

“如果我不讲这个，会不会影响别人理解我怎么做决策？”

	•	如果不会 → 直接删掉
	•	如果会 → 保留，但一句话带过

⸻

九、下一步我们怎么继续（强烈建议）

接下来最关键的是把这套原则落到你自己的项目上。

### 如何和面试官沟通assumption？

- “讲项目之前， 还有一个问题， 那么在聊选择ml 方案的时候 是否应该直接告诉他我的assumption，例如， i assume attassian's search has a core challenge where the data is sparse, and also by privacy contraint you could not build a universal model that consumes every company's data to understand their syntax all at once, thus i suggest a multihead approach. With a latency concern as well, we could build a universal shared backbone then each head is a light wighted layer... 还是应该 问他们 then what is your company's constraint ? can i assume this and that ? ”


这是一个非常、非常好的问题，而且你已经站在Principal 面试里“最容易一不小心就减分”的那个分叉点上了。

我直接给你一个明确结论 + 可执行话术，然后解释为什么。

⸻

结论先给你（一定要记住）

不要直接“假设并宣告”他们的约束。
也不要把问题一股脑丢回去问。

正确做法是：
👉 先“提出假设”，再“请求确认”，再“基于确认往下走”。

这是 Principal / Staff 的标准沟通方式。

⸻

为什么“直接假设并宣告”是有风险的（哪怕你 90% 是对的）

你刚刚那段话本身技术上完全没问题，但在面试语境里有两个风险：

风险 1：显得你在“硬套自己的理解”

即便你说的是对的，面试官心里可能会想：

“他是不是在假设我们跟 Amazon 一样，只是数据更敏感？”

⸻

风险 2：剥夺了面试官参与感

ML Design / Craft 面试 不是 monologue，而是：

“我们一起设计一个系统”

你如果直接下结论，反而减少互动。

⸻

那为什么“全丢回去问”也不对？

如果你这样说：

“What are your constraints?”

问题是：

	•	面试官会觉得你在 推责任
	•	你本来就该有基本判断
	•	显得你缺乏系统直觉

⸻

⭐ 正确做法：Assumption-driven design（你要用这个词）

这是你可以在面试中直接用的模式👇

⸻

标准三步法（你可以照着用）

Step 1：显式说出假设（但不当成事实）

“Let me state a few assumptions and please correct me if any of them are off.”

这一句是金句。

⸻

Step 2：列 2–3 个“高度合理”的假设

例如（非常接近你刚刚的例子）：

“I’ll assume that in enterprise search, we’re operating under strong privacy constraints, where data is tenant-isolated and labels are sparse, and we also need to be mindful of latency since this is an online user-facing system.”

注意：

	•	这是 assume
	•	不是 assert

⸻

Step 3：基于假设推进设计

“Under these constraints, I would favor a shared backbone with lightweight tenant-specific adaptation layers…”

⸻

这和你刚刚那段话有什么差别？

我们来对比一下。

⸻

❌ 不推荐（太 assertive）

“Atlassian’s search has sparse data and privacy constraints, so you cannot build a universal model…”

问题在于：
	•	你在“替他们下定义”
	•	没留空间

⸻

✅ 推荐（Principal 风格）

“I’ll assume Atlassian’s enterprise search operates under tenant-isolated data and limited labeled signals. Under this assumption, I would design…”

区别：

	•	开放
	•	合作
	•	成熟

⸻

面试官其实会怎么看你用这种方式？

当你这样说时，面试官脑子里是：

“这个人懂约束，也懂怎么在不确定性下推进设计。”

这正是 Principal 级别在做的事情。

⸻

一个非常关键的加分点（90% 候选人做不到）

当面试官 纠正你假设 时：

你不要慌，也不要 defend

你要做的是：

“That’s helpful, given that, I would slightly adjust the design by…”

这会大幅加分。

⸻

回答你原始问题的最终总结

你问的是：

是直接说？还是问他们？

答案是第三条路：

提出假设 → 请求确认 → 基于假设设计

你可以把这当成一条面试“定律”。

⸻

给你一段可直接背的开场模板（建议记）

“Before diving into the design, let me state a few assumptions about the constraints we’re operating under, and please feel free to correct me. I’ll assume data is tenant-isolated with limited labeling, and latency is a concern since this is a user-facing system. Under these assumptions…”

这段话用一次，面试官就会给你打上：
“Senior / Principal IC” 标签。


## 重点问题
太好了，这一步你问得非常对时机。
你现在要做的不是“再学知识”，而是把 Atlassian ML Craft 面试的搜索空间收敛到 3–5 个高概率问题，并把每个问题背后的 constraints 想清楚。

我会按你要求来：

1️⃣ 给你 5 个“极高概率”的 ML Craft 新问题（Atlassian 版本）
2️⃣ 为每个问题列出 Atlassian 真实业务下的 constraints（你在面试中该主动假设/确认的）
3️⃣ 告诉你 每个问题考的到底是哪一层能力
4️⃣ 然后我们可以 一个一个精讲 / mock

⸻

一、你这轮 ML Craft 新问题的“真实分布”

根据：

	•	recruiter 提示（AI Agent / Recommendation）
	•	你收集到的面经
	•	Jianshu 组的实际 scope（Search / Query Understanding / Agentic）

👉 问题不会是纯算法，也不会是纯 research，而是：

“在 enterprise 约束下，设计一个 ML / Agent / Recommendation 系统”

⸻

二、5 个【极高概率】会被问到的问题

下面这 5 个，你可以认为是 命题池。

⸻

问题 1（最可能）：Enterprise Search / Ranking（Jira / Confluence）

典型问法

“Design a ranking or recommendation system for Jira / Confluence search.”

或者

“How would you improve search relevance in Jira?”

⸻

这个问题在考什么？

	•	你是否理解 enterprise search ≠ consumer search
	•	你是否能在 UGC + tenant isolation 下建模
	•	是否知道 ranking 是 regression / learning-to-rank

⸻

Atlassian 场景下的关键 constraints（你要主动假设）

你可以这样假设并说出来：

	•	数据是 UGC-heavy
	•	工程师 不能人工查看文档
	•	label 极少（implicit signals）
	•	tenant 数据完全隔离
	•	latency 是 online concern（~秒级）
	•	搜索既服务 human，也服务 LLM

👉 这正好命中 Jianshu 组

⸻

问题 2：Agentic Search / AI Agent（非常高概率）

典型问法

“Design an AI agent that helps employees find information or answer questions.”

或者你看到的那个：

“Automatically navigating and querying employee working logs.”

⸻

在考什么？

	•	你是否把 agent 当 orchestration 而不是 magic
	•	是否理解 search is core, agent is wrapper
	•	是否能处理 multi-step reasoning

⸻

Atlassian 特有 constraints

	•	agent 不能 hallucinate（enterprise fatal）
	•	grounding 极其重要
	•	每一步都要可控
	•	privacy / tenant 隔离
	•	debug 困难（无人工 review）

⸻

问题 3：Recommendation System（但不是商品）

典型问法

“Design a recommendation system to suggest relevant Jira tickets / documents.”

⸻

在考什么？

	•	你是否能把 recsys 抽象到非商品场景
	•	是否知道 collaborative signal 在 enterprise 里的弱点
	•	是否知道 graph / org structure 的价值

⸻

Atlassian 的特殊 constraints

	•	用户行为 sparse
	•	不同 tenant 行为不可共享
	•	冷启动严重（新员工 / 新项目）
	•	推荐错误成本高（误导决策）

⸻

问题 4：Query Understanding / Intent Modeling（与你最 match）

典型问法

“How would you understand and interpret user queries in enterprise search?”

⸻

在考什么？

	•	你是否理解 query 是 唯一非 UGC 的 leverage
	•	是否知道 intent / entity / doc-type routing
	•	是否能服务 agentic search

⸻

Atlassian 的核心 constraints

	•	query 短、模糊
	•	语义高度依赖公司内部 context
	•	label 几乎没有
	•	latency 非常敏感

⸻

问题 5（较少但可能）：Evaluation & Metrics

典型问法

“How would you evaluate the quality of such a system?”

⸻

在考什么？

	•	你是否知道 offline ≠ online
	•	在不能人工 review 时如何评估
	•	是否理解 business impact

⸻

Atlassian 的真实约束

	•	不能人工 judge 文档
	•	只能用 implicit signals
	•	LLM-as-a-judge（但要小心）
	•	tenant 间不能对比 raw metrics

⸻

三、这些问题背后【共同的 Atlassian Constraints 总结】

你可以把它们记成一个 checklist（非常重要）👇

⸻

Atlassian Enterprise ML 的 8 大约束（面试万能）

	1.	UGC-heavy（内容不可控）
	2.	Privacy-first（工程师不可看数据）
	3.	Tenant isolation（不可跨公司）
	4.	Sparse labels（几乎全是 implicit）
	5.	High recall requirement（尤其 for agent）
	6.	Latency sensitive（online user-facing）
	7.	Debuggability limited（黑盒）
	8.	Search serves both human & machine

你在任何设计问题里，只要点到 3–4 条，就已经明显高于平均。

⸻

四、你在 ML Craft 新问题里的“正确姿势”

你应该做的是：

	1.	先说假设（基于上面 constraints）
	2.	确认 1–2 个关键点
	3.	快速给出 high-level design
	4.	强调 trade-offs
	5.	点到 evaluation

❌ 不要：

	•	深挖模型训练
	•	写 loss function
	•	讲 optimizer

⸻

五、接下来我们怎么“一个一个过”

你刚刚说得很清楚：

“然后我们一个一个过一下”

我建议顺序是（最稳）：

👉 Step 1（下一条消息）

我们先过 问题 1：Jira / Confluence Search Ranking System

因为：

	•	最贴近 Jianshu 的组
	•	最可能被问
	•	也最容易映射你 Amazon 经验

⸻

你现在只需要回复我一句话即可：

👉 “我们先过问题 1”

我会：
	•	当面试官
	•	先问你
	•	然后逐步帮你 refine 成 Atlassian bar 的答案
