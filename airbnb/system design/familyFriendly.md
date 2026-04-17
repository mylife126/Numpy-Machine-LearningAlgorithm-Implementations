# Design a family friendly recommendation system

Section 1 — Consolidated Summary

问题定义

这是一个 multi-stage ML product design 问题，包含两个子任务：

	1.	Eligibility / classification：判断一个 listing 是否是 family-friendly
	2.	Filtered ranking：当用户使用 family-friendly filter 时，如何在这些 listing 中排序

所以这不是单纯分类，也不是单纯 ranking，而是：

classification + ranking + serving integration

⸻

我对问题的理解

如果 Airbnb 推出一个 family-friendly filter，我会把它设计成：

	•	先建立一个 family-friendly eligibility system
	•	再把这个 signal 接入搜索 / ranking 系统
	•	最后支持在线 serving + periodic refresh

⸻

Objective

任务 1：identify family-friendly listings

目标：

	•	高 precision（不要把不适合家庭的房源错误标为 family-friendly）
	•	也要有 reasonable recall（不能漏太多）

任务 2：filtered ranking

目标：

	•	在 family-friendly pool 中，继续优化 booking utility
	•	同时更强调“family trip fit”

⸻

简化 pipeline（带方法论）

```
Raw listing data + reviews + amenities + images + complaints
        ↓
(1) Family-friendly Eligibility Model
        ↓
Eligibility score / binary flag
        ↓
Index / feature store / search filter integration
        ↓
(2) Family-friendly Ranking
        ↓
Final ranked family-friendly results
```

⸻

每一层怎么做（核心方法论）

(1) Eligibility layer

👉 方法论：multi-source weak supervision + supervised classifier

先用：

	•	explicit rules（crib, kitchen, washer, child-safe, quiet, etc.）
	•	reviews / complaints / support text
	•	images / amenities

建立初版标签，再训练 classifier
目标：给每个 listing 一个 family-friendly score

⸻

(2) Ranking layer

👉 方法论：先过滤，再在 filtered pool 内做 ranking

不是只按 family-friendly score 排，而是：

	•	family-friendly eligibility 作为 gate / strong feature
	•	ranking 继续优化 booking
	•	再加 family intent match feature

⸻

(3) Serving / hosting

👉 方法论：offline heavy computation + online lightweight lookup

	•	family-friendly score 离线刷新
	•	ranking online 调用
	•	支持 search filter 实时响应

⸻

两次迭代（面试够用）

Iteration 1

	•	rules + GBDT / simple classifier for eligibility
	•	ranking 里把 eligibility 当 feature / filter

Iteration 2

	•	multimodal model（text + image + reviews）
	•	better intent-aware ranking for family travelers

⸻

Evaluation（先抓重点）

Eligibility

	•	PR AUC / precision / recall
	•	especially high precision slice

Ranking

	•	NDCG / booking rate in filtered traffic
	•	family-user satisfaction proxy

⸻

Serving / monitoring

	•	score refresh cadence
	•	feature coverage
	•	disagreement between rules and model
	•	family-friendly complaint rate after launch
	•	search / filter latency

⸻

Section 2 — Technical Deep Dive

1. 子任务 1：family-friendly listing identification

1.1 什么叫 family-friendly？

你一上来要先定义，不然模型没法建。

可能维度

	•	适合儿童 / 婴儿
	•	安全性高
	•	噪音低 / neighborhood 安静
	•	家庭设施齐全
	•	空间 / bed layout 适合家庭
	•	host 对家庭 traveler 友好

👉 面试一句话：

I would first define family-friendly operationally, because this label is not directly available and likely includes safety, amenities, layout, and family suitability signals.

⸻

1.2 数据怎么来（这题关键）

A. Explicit structured signals

	•	amenities:
        •	crib
        •	high chair
        •	kitchen
        •	washer / dryer
        •	extra bedrooms
        •	bathtub
        •	parking
        •	occupancy / guest capacity
        •	room layout

B. Review / support text

	•	review 中出现：
        •	great for kids
        •	family trip
        •	stroller-friendly
	•	noisy / unsafe（负信号）
	    •	complaint / support tickets：
        •	safety issue
        •	child-unfriendly issue

C. Image / visual signals

	•	bunk beds
	•	crib
	•	backyard
	•	play area
	•	stairs / hazards（也可能是负信号）

D. Host / neighborhood context

	•	family POI nearby
	•	school / park / family attractions
	•	quietness proxy
	•	neighborhood safety proxy

⸻

1.3 Label 怎么构造

因为 “family-friendly” 通常没有干净人工标签，所以最合理是：

方法 1：weak supervision（推荐）

组合多个 heuristics：

	•	positive labeling functions：
        •	has crib + high chair
        •	reviews mention family with positive sentiment
	•	negative labeling functions：
        •	repeated safety complaints
        •	nightlife-heavy / noisy review topic

然后得到 noisy labels，训练 classifier

方法 2：human review seed set

	•	抽样人工标一批 family-friendly / not family-friendly
	•	用来校准和评估

面试一句话

Since the label is not naturally clean, I would likely start with weak supervision combined with a small human-labeled seed set.

⸻

1.4 模型怎么做

Iteration 1（baseline）

	•	GBDT / logistic regression
	•	输入：
	•	amenities
	•	capacity
	•	review-derived features
	•	neighborhood family-suitability features

Iteration 2（更强）

	•	multimodal model：
	•	text embeddings（listing description + review summary）
	•	image embeddings
	•	structured features

输出
	•	family-friendly probability score

⸻

2. 子任务 2：family-friendly filtered ranking

2.1 核心思路

当用户打开 family-friendly filter：

不是简单按 family-friendly score 排序，而是：

先 filter / gate，再在这个 constrained pool 中做 ranking

⸻

2.2 Ranking 的主目标

	•	booking utility
	•	query relevance
	•	family intent fit

⸻

2.3 Ranking feature 你要重点讲什么

A. Query / request context

	•	destination
	•	dates
	•	guest count
	•	children count（如果有）
	•	trip length
	•	family-friendly filter on/off

B. Listing / host features

	•	booking priors
	•	review score
	•	price
	•	host reliability

C. Family-specific features（最关键）

	•	family-friendly eligibility score ⭐
	•	child amenity count
	•	family review topic score
	•	safety score
	•	quietness score
	•	nearby family POI

D. Cross features

	•	large group fit
	•	price fit for family traveler
	•	family intent × listing suitability

⸻

2.4 Ranking 方法论

Iteration 1
	•	family-friendly score 作为 hard filter + strong rank feature
	•	booking model within filtered pool

Iteration 2
	•	如果有足够数据，做 family-segment ranking
	•	针对 family travelers 学专门 ranking head
直接参考rankingSystem design
⸻

3. Hosting / Serving（你被要求讲）

推荐方法论：offline heavy + online light

Offline

	•	eligibility model 计算 family-friendly score
	•	review text / image embeddings 批量刷新
	•	neighborhood family features 批量聚合

Online

	•	query 进来
        •	search system 先取 candidate
        •	如果 filter = family-friendly：
        •	使用 eligibility score 过滤 / rerank
        •	ranker 在线打分

⸻

4. Evaluation（这题容易被追问）

4.1 Eligibility model

	•	PR AUC（family-friendly positives 不一定平衡）
	•	precision / recall
	•	human review precision
	•	complaint-based false positives

4.2 Ranking

	•	NDCG in filtered traffic
	•	booking rate
	•	family user satisfaction proxy
	•	family-related complaints after booking

4.3 产品级

	•	filter adoption
	•	conversion after filter usage
	•	bounce rate in filtered search

⸻

5. Monitoring

Eligibility model

	•	feature drift（amenities / reviews / image coverage）
	•	score distribution drift
	•	disagreement rate between rules and model

Ranking

	•	booking rate in filtered traffic
	•	long-tail exposure of family-friendly listings
	•	latency impact

Product

	•	complaints from family travelers
	•	override / manual appeals if hosts dispute their label

⸻

Section 3 — Full Mock Answer

Thanks, this is a great question. Let me first clarify the problem, because I see two connected ML tasks here.

The first task is to identify whether a listing is family-friendly. The second task is, once a user applies a family-friendly filter, how to rank the eligible listings. So I would design this as a two-stage system: an eligibility model followed by a filtered ranking system.

For the eligibility part, the first thing I would do is define what family-friendly means operationally. I would expect it to include signals around child-suitable amenities, safety, quietness, family-appropriate layout, and positive family-related traveler experience. Since this label is unlikely to exist cleanly in the raw data, I would build it using multiple sources.

Concretely, I would start with structured signals such as amenities like crib, high chair, kitchen, washer, extra bedrooms, and capacity. I would also use review and support text, for example positive mentions like “great for kids” or negative signals such as repeated safety or noise complaints. In addition, I would consider image and neighborhood context, such as whether the listing appears to have family-suitable spaces or whether the neighborhood has family-oriented points of interest.

For the first iteration, I would likely use weak supervision plus a small human-labeled seed set. For example, strong positive rules could come from child-related amenities and positive family-review language, while strong negative rules could come from repeated safety complaints. Then I would train a classifier, starting with a structured baseline such as GBDT or logistic regression, to produce a family-friendly probability score. In a second iteration, I would move to a multimodal model that combines structured features with text and image embeddings if the business impact justifies it.

Once I have that family-friendly eligibility score, I would integrate it into search. When a user applies the family-friendly filter, I would first use that score as a filter or gating condition to define the candidate pool. Then, within that filtered pool, I would still run a ranking model rather than simply sorting by the family-friendly score.

The reason is that users still care about relevance, quality, and booking utility. So the ranker should combine normal booking-oriented signals such as query relevance, price, reviews, and host reliability with family-specific signals such as family-friendly score, family amenity count, family review sentiment, safety score, and family suitability of the neighborhood.

In terms of serving, I would keep the system mostly offline-heavy and online-light. The family-friendly eligibility score, review-derived features, and image-based signals can be precomputed and refreshed offline. Then at query time, the online search system only needs to fetch those features and apply the family-friendly filter and ranking quickly.

For evaluation, I would evaluate the eligibility model using PR AUC, precision, and human-review precision, since false positives can hurt host trust. For the ranking system, I would look at NDCG and booking rate within family-filtered traffic, and I would also track downstream product metrics such as family-user satisfaction and complaint rates.

Finally, I would monitor feature coverage, score drift, disagreement between rule-based and model-based labeling, and the business impact of the filter itself. Over time, I would iterate from a simple rule-plus-classifier system to a richer multimodal system once I understand which signals actually drive family traveler satisfaction.

## 控场
🚀 控场版答案（推荐直接练）

⸻

🟢 开场（控场 + framing）

Thanks, this is a great question. Let me first frame the problem, because I see two connected ML tasks here.

The first task is to identify whether a listing is family-friendly. The second task is, once a user applies a family-friendly filter, how to rank the eligible listings.

So I would design this as a two-stage system: an eligibility model followed by a filtered ranking system.

I’ll start with how I define and build the eligibility model, then move to ranking and serving. I’ll pause along the way in case you want me to go deeper into any part.

👉 这里已经完成控场

⸻

🟢 Part 1：定义 label（很关键）

For the eligibility part, the first step is defining what “family-friendly” means operationally.

I would expect it to include signals around child-suitable amenities, safety, quietness, family-appropriate layout, and positive family-related traveler experience.

Since this label doesn’t exist cleanly in raw data, I would construct it using multiple sources.

👉 🔵 控场句：

This label definition is important because it directly drives both feature design and model behavior.

👉 🟡 pause点：

I’ll pause here briefly — do you want me to go deeper into label construction or move to feature signals?

⸻

🟢 Part 2：feature + data construction

Concretely, I would start with structured signals such as amenities like crib, high chair, kitchen, washer, extra bedrooms, and capacity.

I would also use review and support text, for example positive mentions like “great for kids” or negative signals such as repeated safety or noise complaints.

In addition, I would consider image and neighborhood context, such as whether the listing appears to have family-suitable spaces or whether the neighborhood has family-oriented points of interest.

👉 🔵 控场句（加分）：

So I’m combining explicit signals, implicit behavioral signals, and contextual signals to approximate family suitability.

⸻

🟢 Part 3：建模（iteration思路很好）

For the first iteration, I would use weak supervision plus a small human-labeled seed set.

For example, strong positive rules could come from child-related amenities and positive family-review language, while strong negative rules could come from repeated safety complaints.

Then I would train a classifier, starting with a structured baseline such as GBDT or logistic regression.

👉 🔵 控场句（关键）：

I’m intentionally starting with a simple model because I want to validate the label quality before increasing model complexity.

👉 非常加分

⸻

继续：

In a second iteration, I would move to a multimodal model that combines structured features with text and image embeddings if the business impact justifies it.

👉 🟡 pause点：

I can go deeper into the weak supervision setup or the multimodal model if that’s interesting.

⸻

🟢 Part 4：ranking（你这里很好）

Once I have that family-friendly score, I would integrate it into search.

When a user applies the family-friendly filter, I would first use that score as a gating condition to define the candidate pool.

Then within that filtered pool, I would still run a ranking model rather than simply sorting by the family-friendly score.

👉 🔥 关键解释（很好，保留）

The reason is that users still care about relevance, price, reviews, and booking utility, so the ranker should combine general booking signals with family-specific signals.

👉 🔵 控场升级：

So conceptually, I treat family-friendly as a constraint plus a feature, not as a standalone ranking objective.

👉 这句话很强

⸻

🟢 Part 5：serving（你这段很好）

In terms of serving, I would keep the system mostly offline-heavy and online-light.

The eligibility score and derived features can be precomputed offline, and at query time the system only needs to apply filtering and ranking.

👉 🔵 控场句：

This helps control latency and makes the system easier to scale.

⸻

🟢 Part 6：evaluation（很好但可以更“结构化”）

For evaluation, I would evaluate both the eligibility model and the downstream ranking impact.

👉 分两层说：

For the eligibility model, I would look at PR AUC, precision, and human-judged precision, since false positives can hurt trust.

For ranking, I would evaluate NDCG and booking rate within family-filtered traffic, as well as downstream satisfaction and complaint metrics.

⸻

🟢 Part 7：monitoring（加一个控场句）

For monitoring, I would track feature coverage, score drift, and disagreement between rule-based and model-based signals.

👉 🔵 控场升级：

One important thing I would watch is whether the filter becomes too restrictive or too permissive over time.

👉 非常 product-aware

⸻

🟢 收尾（一定要有）

Overall, I would start with a simple rule-plus-classifier system, validate the label and business impact, and then evolve to a richer multimodal model once I understand which signals actually matter.

I can go deeper into labeling, ranking integration, or model design depending on where you’d like to focus.

👉 🟢 主动交还控制权（很关键）

⸻

🧠 三、你这题最加分的3句话（建议记住）

1️⃣

I treat family-friendly as a constraint plus a feature, not as a standalone ranking objective.

👉 非常高级

⸻

2️⃣

I’m intentionally starting with a simple model to validate label quality before increasing complexity.

👉 非常 senior

⸻

3️⃣

The problem really depends on how we define the target, because that drives both labeling and system design.

👉 万能句

⸻

🚀 四、你的控场已经是对的，但再优化一点点

你现在已经具备：
	•	有结构
	•	有节奏
	•	有 iteration thinking
	•	有 product sense

你只差：

👉 把“连续讲述”变成“分段推进 + 主动停顿 + 引导选择”

⸻

🧾 五、最后给你一个万能控场模板（可以套所有题）

你明天可以直接用：

Let me first frame the problem in a production setting.
I’ll start with a simple v1 system and walk through architecture, data, modeling, and serving.
There are a couple of natural deep dives here — label construction and modeling tradeoffs — and I’ll pause along the way so we can go deeper where you prefer.