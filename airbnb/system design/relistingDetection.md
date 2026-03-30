# Relisting detection

## Section 1 — Consolidated Summary（快速抓方法论）

问题定义

这是一个 relisting / evasion detection = retrieval + matching + risk decision 问题

👉 核心不是判断“好坏 listing”，而是：

这个新 listing 是否是历史坏 listing 的“换皮重发”

⸻

Pipeline（带方法论的脑图）
```
New Listing
   ↓
(1) Candidate Retrieval  ← 目标：高 recall + 低 latency
   ↓
(2) Pairwise Matching    ← 目标：判断是否 same entity / relisting
   ↓
(3) Risk Aggregation     ← 目标：综合风险 + 决策
   ↓
Action (allow / review / block)
```

⸻

每一层怎么做（核心方法论 ⭐）

(1) Candidate Retrieval（必须讲清楚）

👉 方法论：multi-channel retrieval = exact + semantic

	•	子层1（高 precision）：
        •	exact / fuzzy match：
        •	address / geo proximity
        •	phone / payment / device / IP
	•	子层2（补 recall）：
        •	semantic ANN retrieval：
        •	image embedding
        •	text embedding

👉 一句话总结：

用“强规则 + embedding召回”快速把全量库压到 top-K suspicious candidates

⸻

(2) Pairwise Matching

👉 方法论：pairwise similarity learning

输入：

	•	new listing + candidate listing

输出：

	•	是否 relisting , 0/1 probability

👉 迭代：

	•	Iter1：rules / heuristic
	•	Iter2：GBDT（结构化特征）
	•	Iter3：multimodal matching（text + image）

⸻

(3) Risk Aggregation / Policy

👉 方法论：multi-signal risk scoring + 分层决策

综合：

	•	match score
	•	host risk
	•	graph linkage

输出：

	•	risk tier → 不同 action

⸻

Evaluation（必须脑子里有）

	•	Retrieval：recall@K（有没有抓到）
	•	Matching：PR-AUC / precision@K
	•	Policy：false positive rate / review precision

⸻

⸻

## Section 2 — Technical Deep Dive（细节补全用）

⸻

### 1️⃣ Retrieval（细节你可能会被追问）

为什么 multi-channel？

因为：

	•	adversarial → 改地址 / 改文案 / 改图片
	•	单一方法一定会漏

⸻

推荐你记住这套 retrieval 组合 ⭐

Channel A：exact / strong signals

	•	phone / payment / device / IP
	•	exact / fuzzy address
	•	geo proximity

Channel B：semantic retrieval

	•	image embedding ANN
	•	text embedding ANN

⸻

输出

	•	union top-K candidates（去重）

⸻

### 2️⃣ Pairwise Matching（核心建模）

⸻

feature 四类（必须熟）

A. 强 linkage（最重要）

	•	same phone / payment / device
	•	same image hash
	•	same geo

⸻

B. 内容相似

	•	text similarity
	•	image similarity
	•	amenities / capacity

⸻

C. 行为特征

	•	removal → new listing 时间间隔
	•	posting pattern

⸻

D. graph features（加分）

	•	shared device cluster
	•	shared payment cluster
	•	graph distance to bad nodes

⸻

3️⃣ Label（容易被问）

正样本

	•	confirmed relisting pairs

负样本（重点⭐）

	•	hard negatives：
	•	    same area + similar listing but NOT relisting

⸻

4️⃣ Evaluation（补你容易卡的点）

⸻

Retrieval

Recall@K = 是否真实匹配在 top-K 中

⸻

Matching（为什么 PR-AUC）

因为：

	•	正样本极少
	•	更关心高分 precision

⸻

Policy

	•	precision for auto-removal
	•	false positive rate
	•	review acceptance rate

⸻

5️⃣ Serving（结构）

online（快速）

	•	rules + lightweight model

offline（深度）

	•	graph analysis
	•	heavy model

⸻

6️⃣ Monitoring（重点）

	•	feature missing rate
	•	match score drift
	•	false positive appeal
	•	new evasion pattern

⸻

⸻

Section 3 — Full Mock Answer（带展开但不冗长）

⸻

Thanks, this is a very interesting problem.

I haven’t worked directly on relisting detection before, but I see strong similarities with retrieval and matching problems I’ve worked on, where the goal is to identify highly similar entities using multimodal features. Based on that, I would approach this not as a standalone classifier, but as a retrieval plus matching system.

At a high level, I would design this in three stages: candidate retrieval, pairwise matching, and risk aggregation.

Starting with the retrieval stage, since we need to handle a large corpus of historical bad listings, the goal here is to achieve high recall under low latency. Concretely, I would design this as a multi-channel retrieval stage.

The first sub-layer would focus on strong, high-precision signals, such as fuzzy address matching, geo proximity, and host-level identifiers like phone number, payment information, or device or IP clusters if available. These signals help quickly retrieve highly suspicious candidates.

However, since adversarial hosts can slightly modify their information, I would add a second sub-layer based on semantic retrieval. Specifically, I would use embedding-based ANN retrieval over listing images and text descriptions to capture visually or semantically similar listings. The goal of this stage is simply to shrink the large search space into a manageable top-K candidate set while maintaining high recall.

Then, in the second stage, I would build a pairwise matching model that takes the new listing and each retrieved candidate and predicts the probability that they are relistings of the same underlying entity.

For features, I would organize them into four groups. First, strong linkage features such as shared phone, payment, device fingerprint, or duplicate images. Second, content similarity features such as text and image similarity, amenities, and capacity. Third, temporal and behavioral features such as how quickly the new listing appears after removal. And fourth, graph-based features that capture shared connections through devices, payments, or other identifiers.

For the MVP, I would start with a hybrid approach combining high-precision rules with a structured model like GBDT. In later iterations, I would incorporate multimodal matching models, such as Siamese or two-tower architectures, to better handle adversarial changes in text and images.

In the final stage, I would aggregate the matching scores along with host-level and graph-level risk signals to produce an overall risk score. Based on this score, I would apply a policy layer to decide actions such as allowing the listing, sending it to human review, suppressing it in search, or automatically removing it.

For evaluation, I would separately evaluate each stage. For retrieval, I would measure recall at K to ensure that true relisting matches are captured. For the matching model, I would focus on PR AUC and precision at the review threshold, since positive cases are rare. At the system level, I would track false positive rates and review precision to ensure we are not harming legitimate hosts.

Finally, I would deploy this as a two-speed system, with a lightweight online pipeline for immediate detection at listing creation time, and a deeper offline pipeline that incorporates graph analysis and more expensive models. I would also continuously monitor for drift and new evasion patterns, since adversarial behavior will evolve over time.

⸻

## Question: policy layer how? suppose your second layer says multiple candidates are showing above 0.5 risk confidence?
很好，这个 follow-up 非常典型 + 非常关键（Staff signal）。
本质是在问：

pairwise → entity-level decision 怎么做？

你现在不能再停留在“pairwise score”，而是要进入：

👉 aggregation + decision policy

⸻

✅ Step 1 — 核心问题抽取

当前你有：

	•	一个 new listing
	•	对应 top-K retrieved candidates
	•	每个 pair 有：
$score_i = P(relisting \mid new, candidate_i)$

👉 interviewer问的是：

“现在有多个 0.5 左右的 match，你怎么做最终决策？”

⸻

✅ Step 2 — 你可以直接说的答案（强版本）

⸻

That’s a great question — at this point, we are moving from pairwise predictions to an entity-level decision.

I would not rely on a single pairwise score, but instead aggregate signals across all matched candidates to estimate the overall relisting risk.

Concretely, I would design an aggregation layer that combines multiple match scores, along with additional signals such as host-level risk and graph connectivity.

For example, if a new listing has multiple moderately high similarity matches to previously removed listings, that is often stronger evidence than a single high-confidence match, since it suggests consistency across multiple signals.

So I would construct aggregate features such as the maximum match score, the average of top-K scores, the number of matches above a threshold, and potentially a weighted sum where higher-confidence matches contribute more.

On top of that, I would incorporate host-level signals and graph-based linkage, for example whether this listing is connected to known bad actors through shared devices, payment methods, or contact information.

Finally, I would feed these aggregated features into a lightweight risk model or policy layer, and use calibrated thresholds to decide actions such as allow, send to review, or automatically block.

⸻

🧠 Step 3 — deeper拆解（你要真正理解）

⸻

一、为什么不能直接用 max(score)？

你可以主动说：

❌ 只用 max 的问题

	•	单一 match 可能 noisy
	•	对 adversarial 不鲁棒
	•	容易误伤

⸻

✅ 更合理：multi-evidence aggregation

👉 核心思想：

多个中等证据 > 一个弱证据

⸻

二、aggregation 怎么做（你要能展开）

⸻

推荐你记住这 4 个 feature（非常好用 ⭐）

⸻

1️⃣ Max score

$max_i score_i$

👉 strongest signal

⸻

2️⃣ Top-K average

$\frac{1}{K}\sum_{i=1}^{K} score_i$

👉 稳定性

⸻

3️⃣ Count above threshold ⭐（非常关键）

$\#\{i: score_i > \tau\}$

👉 多个 match = 更可信

⸻

4️⃣ Weighted sum（进阶）

$\sum_i w_i \cdot score_i$

例如：

	•	similarity权重
	•	recency权重

⸻

👉 面试一句话总结：

I would combine max, top-K average, and count-above-threshold features to capture both strong and consistent signals.

⸻

三、你这个case（5个0.5）该怎么判断？

👉 你可以这样解释（非常加分）：

⸻

If I see multiple matches with moderate scores like 0.5, I would actually consider that quite suspicious.

A single 0.5 match may be ambiguous, but multiple consistent matches suggest that the new listing shares multiple dimensions of similarity with known bad listings.

In practice, I would expect the aggregation layer to assign higher risk to this case compared to a single isolated match.

⸻

👉 这句话非常关键：

👉 “consistency across candidates”

⸻

四、最终 decision 怎么做？

⸻
```
方法1：rule-based threshold（MVP）

if max_score > 0.9 → block
elif count(score > 0.5) >= 3 → review
else → allow
```

⸻

方法2：risk model（推荐）

输入：

	•	aggregation features
	•	host risk
	•	graph signals

输出：
$P(\text{relisting risk})$

⸻

方法3：policy layer（最成熟）
```
👉 multi-tier：

Risk	Action
low	allow
medium	review
high	suppress
very high	remove
```

⸻

五、你可以加的一个“Staff加分点” ⭐

👉 graph reasoning

⸻

如果 5 个 match：

	•	都来自同一个 bad host cluster
👉 风险更高

⸻

👉 你可以说：

I would also consider whether these matched candidates are connected in the same graph cluster, since multiple matches within the same bad actor group is a stronger signal than unrelated matches.

⸻

🔥 最终你可以说的完整版（强化版）

⸻

That’s a great question — at this stage, we need to move from pairwise similarity to an entity-level decision.

Instead of relying on a single match, I would aggregate signals across all retrieved candidates. For example, I would compute features such as the maximum match score, the average of top-K scores, and the number of matches above a threshold.

In your example, having multiple matches with moderate scores like 0.5 is actually quite suspicious, because it suggests consistent similarity across multiple dimensions, which is often stronger evidence than a single high-confidence match.

I would also incorporate additional signals such as host-level risk and graph connectivity — for example, whether these matched listings are connected through shared devices or payment methods.

Finally, I would feed these aggregated signals into a risk scoring layer and apply calibrated thresholds to determine actions such as allow, send to review, or automatic removal, depending on the confidence and business tolerance for false positives.

⸻