================================

## ① 技术细节拆解（deep dive primitives）

================================

⸻

### 1. Re-ranking utility（最核心表达）

✅ 最推荐面试表达

 $U(i \mid S) = \alpha \cdot Score_{core}(i) + \beta \cdot ExposureBonus(i) + \gamma \cdot FreshnessBonus(i) - \delta \cdot DivPenalty(i, S)$

----

1.1 Diversity（怎么定义）

👉 推荐讲法（最稳）：

$DivPenalty(i, S) = \max_{j \in S} sim(e_i, e_j)$

	•	sim：cosine similarity（embedding）
	•	S：已经选进 top-K 的集合

👉 面试一句话：

“I would compute diversity penalty based on similarity between the candidate and already selected results, for example using max embedding similarity.”

---

1.2 Exposure（怎么做）

👉 定义：

$Exposure_i = Impressions_i^{(7d)}$

👉 转换为 bonus（关键点）：

$ExposureBonus(i) = \frac{1}{\sqrt{1 + Impressions_i}}$

👉 为什么这样：

	•	bounded（不会爆）
	•	under-exposed 有 boost
	•	over-exposed 不被无限惩罚

👉 面试一句话：

“I would use a smoothed inverse-impression function so under-exposed listings get some exploration without destabilizing ranking.”

---

1.3 Freshness（不要讲太 naive）

👉 推荐定义：

$Freshness(i) = e^{-Age_i / \tau}$

或：

	•	最近 availability 改变
	•	最近 newly eligible

👉 面试一句话：

“Freshness should be a bounded exploration bonus rather than a dominant objective.”

---

### 2. Label & Training（必须说对）

🚨 核心点：booking 太 sparse

👉 正确策略：

	•	north star：booking（CVR）
	•	proxy：
	•	    click
	•	    save
	•	    long dwell
	•	    checkout intent

⸻

✅ 推荐训练方式

方法1：multi-task learning

	•	head1: CTR
	•	head2: CVR

方法2：staged training（你也可以说）

	•	先 CTR pretrain
	•	再 fine-tune CVR

⸻

⚠️ 你之前的风险点（已修正）

你之前说：

downsample negative + CVR直接学

👉 不够 strong

现在正确表达：

“Because booking labels are sparse, I would use auxiliary dense signals such as clicks for pretraining or multi-task learning, while keeping booking as the primary objective.”

---

### 3. Feature Engineering（重点升级版）

⸻

Core ranker feature 四大类（面试高分版）

⸻

A. Request / query context

👉 决定“用户当下要什么”

	•	query text / query embedding
	•	destination / geo location
	•	check-in / check-out dates
	•	trip length
	•	guest count
	•	locale / language
	•	device / surface
	•	request time / seasonality / holiday context

⸻

B. Listing / host features（trust & quality）

Listing features

	•	listing text embedding（title / description）
	•	image embedding（视觉质量）
	•	amenities（wifi, kitchen, etc）
	•	property type（apartment / house）
	•	neighborhood / geo cluster
	•	price（+ normalized / bucketed）
	•	availability pattern
	•	review score
	•	review count
	•	historical booking rate（重要）

⸻

Host features（非常关键）
👉 Airbnb 比 ads 更重要的一类

	•	host response rate
	•	host response time
	•	host cancellation rate ⭐
	•	complaint / dispute rate
	•	host tenure
	•	quality badge（superhost）

👉 面试一句话：

“Host reliability features are critical because they directly affect booking trust and conversion.”

⸻

C. Cross features（最像 ranking）

👉 最重要！！！

	•	query-list semantic similarity ⭐
	•	amenity match（pet-friendly / pool）
	•	price fit（vs market / vs user preference）
	•	geo fit（distance to center / POI）
	•	trip intent fit（family / business / luxury）
	•	date availability match

⸻

D. User / session features（iteration 2）

	•	long-term preference embedding
	•	recent viewed listings
	•	recent clicked/saved/booked sequence
	•	search reformulation
	•	inferred travel style / price sensitivity

⸻

👉 面试一句话总结：

“Cross features and host quality features are often the strongest predictors of booking, while personalization further improves ranking in later iterations.”

----

### 4. Missing feature / robustness

你说的 feature dropout 是对的 👍（这是加分点）

完整版本：

	•	missing indicator（explicit）
	•	feature dropout（训练）
	•	fallback（serving）
	•	feature coverage monitoring（线上）

⸻

👉 面试一句话：

“I’d explicitly encode missingness and use feature dropout during training to improve robustness under real-world feature sparsity.”

---

================================

## ② Consolidated Summary（完整脑图版）

================================

⸻

问题定义

Airbnb search ranking 是一个 two-sided marketplace ranking problem。

核心不是 click，而是：

	•	primary objective: booking likelihood / expected booking utility
	•	guardrails:
	•	    guest trust / quality（cancellation, reliability）
	•	    marketplace health（diversity / exposure / freshness）
	•	    latency / scalability

⸻

系统边界

假设：

	•	retrieval 已存在（ANN / semantic retrieval）
	•	当前 focus：ranking + re-ranking

⸻

Pipeline

Query → Retrieval → Core Ranker → Re-ranker → Final Ranking

⸻

Core ranker

Iteration 1（baseline）

	•	non-personalized
	•	model: GBDT / shallow LTR
	•	features: request + listing + cross
	•	objective: booking-oriented score

⸻

Iteration 2（personalization）

	•	user long-term embedding
	•	session behavior
	•	pooled embedding → sequential model

⸻

Advanced（PEPNet思想）

	•	separate universal + personalized
	•	gating instead of direct fusion
	•	improve generalization & sparsity handling

⸻

Label strategy

	•	north star: booking
	•	proxy: click / save / dwell
	•	training:
	•	multi-task OR staged

⸻

Re-ranker / policy layer

$U(i \mid S) = \alpha \cdot Score_{core}+ \beta \cdot Exposure + \gamma \cdot Freshness - \delta \cdot Diversity$

⸻

Robustness

	•	missing features
	•	feature dropout
	•	fallback
	•	monitoring

⸻

Evaluation

Offline

	•	PR- AUC / logloss
	•	NDCG
	•	calibration
	•	slice analysis

Online

	•	booking rate ⭐
	•	revenue / GMV
	•	cancellation / trust
	•	exposure distribution
	•	Feature drip via PSI
	•	new listing success

⸻

Serving / latency

	•	cache listing embeddings
	•	precompute static features
	•	lightweight real-time features
	•	fallback if personalization missing

⸻

Monitoring / debugging

	•	feature drift
	•	latency
	•	label drift
	•	exposure concentration
	•	cold-start performance

⸻

================================

## ③ Full Mock Answer（最终面试话术）

================================

⸻

1️⃣ Problem / Objective

Thanks for the question. Let me first clarify the scope and objective.
I’ll focus on designing the ranking stage for Airbnb search, assuming candidate retrieval is already in place after filters like destination, dates, and availability.

Since Airbnb is a two-sided marketplace, I would define the primary objective as maximizing booking likelihood rather than just engagement.

In addition, I would introduce two sets of guardrails. First, guest trust signals such as cancellation risk, listing quality, and host reliability. Second, marketplace health signals such as avoiding exposure concentration, supporting new listings, and maintaining diversity.

Finally, the system needs to satisfy latency and scalability constraints.

⸻

2️⃣ System Design

At a high level, I would structure the system into two layers after retrieval.

The first layer is a core ranker that scores each listing independently based on booking utility.
The second layer is a lightweight re-ranking or policy layer that adjusts the top results to incorporate marketplace constraints such as diversity, freshness, and exposure balance.

The overall pipeline is: query → retrieval → core ranking → re-ranking → final results.

I prefer this separation because the model focuses on learning relevance and booking probability, while the re-ranking layer provides transparent and controllable optimization for marketplace objectives.

⸻

## 3️⃣ Model Design

### Feature part to start with:

For the first MVP, I’d start with a non-personalized ranker.
I’d organize the features into four groups.

First, request context features: query text or query embedding, destination, trip dates, trip length, guest count, locale, and similar request-level signals.

Second, listing and host features: listing text/image embeddings, amenities, property type, price, geographic features, review score, review count, as well as host reliability features such as response rate, response time, historical cancellation rate, and complaint-related signals.

Third, cross features that capture how well the listing matches the request: query-list semantic similarity, amenity match, price fit relative to that market, and location fit.

### Iteration 1:
For the first iteration, I’d start with a relatively simple but strong baseline such as a gradient-boosted tree model or a shallow learning-to-rank model, because it handles heterogeneous features well, is easier to debug, and lets us move quickly before committing to a heavier personalized architecture.

### Iteration 2:
In the second iteration, I’d add personalization features, especially long-term guest preferences and short-term session behavior, such as recently viewed, clicked, saved, or booked listings. For the short-term session representation, I’d first use a pooled embedding over recent listing interactions as a lightweight MVP, 
and only move to a sequential encoder via attention mechanism if offline and online gains justify the additional serving complexity.

### Advanced（PEPNet）

One refinement I’d consider is separating general relevance signals from personalization signals using a gated architecture.
Instead of directly mixing them, user-specific signals can modulate shared embeddings, which helps with data sparsity and improves generalization.

⸻

### Training

For training, one key challenge is that booking labels are sparse and delayed, so I would not rely only on raw booking labels in the first version. I’d likely use denser proxy labels such as click or save as auxiliary supervision, while keeping booking as the north-star business metric. A practical setup could be multi-task learning or staged supervision, depending on the amount of booking signal available.

Another challenge is missing or unstable online features. I’d handle that by explicitly encoding missingness, using models that tolerate partial feature availability, applying feature dropout during training to improve robustness, and monitoring feature coverage in production.

Then, after the core ranker produces a booking-oriented score, I’d use a lightweight re-ranking layer with a utility like:
core score plus a bounded bonus for under-exposed listings and fresh listings, minus a diversity penalty based on similarity to already selected results.
This gives transparent control over marketplace-health tradeoffs without forcing every objective into a single model.
⸻

### 4️⃣ Re-ranking

After the core ranker produces a booking-oriented score, I would apply a greedy re-ranking strategy.

At each step, I select the next listing by maximizing a utility function that combines the core score with a bounded bonus for under-exposed and fresh listings, and a penalty for similarity with already selected items.

This allows us to balance short-term booking performance with long-term marketplace health.

⸻

### 5️⃣ Evaluation

Offline, I would evaluate using ranking metrics such as NDCG and AUC, along with calibration and slice analysis.

Online, I would run A/B tests measuring booking rate as the primary metric, along with guardrails such as cancellation rate, exposure distribution, and new listing success rate.

⸻

### 6️⃣ Serving / Latency

For serving, I would cache heavy features such as listing embeddings and precompute static features.

Real-time features would be lightweight, such as session signals.

I would also design fallback strategies so that if personalization signals are unavailable, the system can still produce a reasonable ranking.

⸻

### 7️⃣ Monitoring / Debugging

In production, I would monitor feature coverage, latency, and model drift.

I would also track marketplace-level metrics such as exposure concentration and cold-start listing performance, and use these signals to iteratively refine both the model and re-ranking policies.

⸻

## Follow up Questions
```
1. PSI 怎么使用来检测飘逸， embedding的feature怎么检测飘逸， 为什么PR AUC， NDCG怎么验证？
```

核心问题

你这个 follow-up 本质上在问三件事：

	1.	feature drift 怎么监控，尤其是 embedding drift 怎么监控
	2.	eCVR / booking ranker 的 offline evaluation 该选什么 metric，为什么
	3.	NDCG 到底怎么落到 query-listing ranking 上，binary label 和 graded label 分别怎么做

⸻

你可以直接说的回答话术

Great questions. I’d separate this into three parts.

First, for feature drift, PSI works well for scalar tabular features like price or review count, but for embeddings I would monitor drift differently. For embeddings, I’d track summary statistics such as embedding norm, per-dimension mean and variance, cosine similarity to the historical centroid, and distributional distance between training-time and serving-time embeddings. I’d also monitor downstream proxies like nearest-neighbor stability or retrieval overlap, because sometimes the embedding distribution may look stable while its semantic behavior changes.

Second, for evaluating a booking-oriented ranker, I would not rely on ROC AUC alone because booking labels are usually very sparse. PR AUC is often more informative in that setting since it emphasizes performance on the positive class. But for ranking systems, I would also use ranking metrics such as NDCG, because the model is not just classifying independently — it is ordering listings within each query context.

Third, for NDCG, I would compute it within each search request. The model outputs a score for each candidate listing under that request, we sort by score, then compare the ranked list against observed relevance labels. If labels are binary, DCG is computed using 1 for booked and 0 for not booked. If labels are graded, for example click < save < booking, then we use those graded relevance values inside DCG. Then we average NDCG across queries.

So in practice, I would use a combination of calibration / AUC-type metrics for score quality, and NDCG-type metrics for ranking quality.

⸻


### 一、Feature Drift：PSI怎么做，embedding drift怎么做

1) 连续特征：price 这类怎么做

你前面那个 PSI 例子可以继续沿用，面试里这样讲最稳：

步骤

	1.	选一个 reference distribution
        •	训练集
        •	或最近稳定窗口（比如过去30天）
	2.	把 feature 分 bucket 例如 price 分 10 个分位数桶
	3.	计算训练分布 p_i 和线上分布 q_i

$PSI = \sum_i (p_i - q_i)\log\frac{p_i}{q_i}$

阈值（面试里可以说）

	•	< 0.1：稳定
	•	0.1–0.25：有 drift
	•	0.25：严重 drift / alert

Ready-to-say

For scalar features like price, I would bucket the feature, compare the training distribution with the serving distribution, and compute PSI. It is easy to operationalize and works well for continuous tabular signals.

⸻

2) Embedding drift：不能直接只用 PSI

因为 embedding 是高维向量，不能简单当成单个 scalar feature。

我推荐你面试里说 4 层监控

⸻

A. 向量级 summary stats

最简单也最实用：

	•	embedding norm 的分布
	•	每一维的 mean / variance
	•	PCA 后前几主成分的 mean / variance
	•	cosine similarity to historical centroid

例子
设训练期 embedding centroid 为：

$c_{train} = \frac{1}{N}\sum e_i$

线上窗口 centroid：

$c_{serving} = \frac{1}{M}\sum e_i$

监控：

$\cos(c_{train}, c_{serving})$

如果突然下降，说明整体语义空间漂了。

面试一句话

For embeddings, I would first monitor coarse statistics such as norm distribution, centroid drift, and low-dimensional projections, since these are cheap and stable to track online.

⸻

B. 分布距离（更强一点）

可以对 embedding 做 PCA / random projection 降维，再做：

	•	PSI
	•	KS test
	•	Wasserstein distance
	•	KL divergence（如果离散化）

面试一句话

Because embeddings are high-dimensional, I’d usually project them into a lower-dimensional monitoring space and then compute distance metrics there.

⸻

C. 邻居稳定性 / 语义稳定性（非常加分）

如果上游 embedding model 变了，最关键不是数值漂没漂，而是语义关系变没变。

可以监控：

	•	top-k nearest neighbors overlap
	•	retrieval overlap rate
	•	average cosine similarity to historical nearest neighbors

例子
对固定一批 probe listings / queries：

	•	旧 embedding top-20 neighbors
	•	新 embedding top-20 neighbors
	•	算 overlap ratio

如果 overlap 暴跌，说明 embedding 语义变了。

面试一句话

I would also monitor behavioral drift of embeddings, for example nearest-neighbor stability or retrieval overlap on a fixed probe set, because semantic regressions may not be visible from simple moment statistics alone.

⸻

D. Downstream impact（最重要）

Embedding changed 最终会体现在：

	•	retrieval recall drop
	•	ranker score distribution shift
	•	booking / CTR / save rate slice drop

面试一句话

Ultimately, embedding drift should also be validated through downstream signals, because the embedding can change numerically without necessarily hurting the product, and vice versa.

⸻

3) 发现 embedding drift 后怎么处理

情况 1：上游模型版本变了

	•	versioned rollout
	•	shadow compare old vs new embeddings
	•	AB / canary before full launch

情况 2：distribution drift but no bug

	•	retrain downstream ranker with new embeddings
	•	re-normalize / recalibrate
	•	keep compatibility bridge for old/new embedding versions

情况 3：严重故障

	•	rollback embedding model
	•	fallback to previous cached embeddings

Ready-to-say

If the embedding model changes, I would never silently swap it in production. I’d do versioned rollout, compare old and new embedding behavior on a probe set, and retrain or recalibrate downstream models if the semantic space changes materially.

⸻

### 二、eCVR / booking ranker：ROC AUC, PR AUC, 该选什么？

1) 先给结论

对稀疏正样本（booking / CVR）

PR AUC 通常比 ROC AUC 更有信息量。

因为：

	•	booking 是极不平衡问题
	•	ROC AUC 对大量 TN 不敏感，容易“看起来很好”
	•	PR AUC 更关注 positive retrieval quality

⸻

2) 为什么 ROC AUC 可能误导

假设：

	•	1% booking
	•	99% no-booking

一个模型把大部分负样本分开就能拿到不错 ROC AUC，
但它可能对真正要找的 positive（会book的 listing）帮助一般。

面试一句话

ROC AUC can look artificially strong in highly imbalanced problems because true negatives dominate, while PR AUC is more sensitive to performance on the rare positive class.

⸻

3) 为什么 PR AUC 更适合 eCVR

因为 eCVR / booking 本质是：

	•	rare positive
	•	我更关心 high-score 区域是不是抓住了真正会转化的 listing

PR AUC 直接反映：

	•	precision among top predicted positives
	•	recall of rare positives

面试一句话

For eCVR or booking models, PR AUC is often more aligned with what we care about operationally: among the items we score highly, how many are truly likely to convert?

⸻

4) 那 ROC AUC 就没用吗？

不是。
更成熟的回答是：

	•	ROC AUC：可以保留，做整体 separability 参考
	•	PR AUC：对稀疏 positive 更关键
	•	Calibration：也要看
	•	NDCG：ranking 场景必须看

Ready-to-say

I would not choose only one metric. I’d typically look at PR AUC for sparse conversion labels, calibration for score quality, and NDCG for actual ranking quality within each search context.

⸻

### 三、NDCG 到底怎么算？你的 ranker 输出的是每个 query impression 下 listing 的 conversion likelihood，对吧？

对。
这正适合 NDCG，因为 NDCG 本来就是 grouped ranking metric。

⸻

1) NDCG 的正确应用场景

对每一个 query / search request：

	•	你有一组候选 listing
	•	ranker 给每个 listing 一个 score（比如 eCVR / booking likelihood）
	•	用这个 score 排序
	•	再跟观察到的 relevance label 比较

关键点

NDCG 是在每个 query group 内算，然后再 across queries 取平均。

⸻

2) Binary label 场景怎么做 NDCG

例子

某次 query 下有 5 个 listing，模型排出来：
```
rank	listing	predicted score	observed booking label
1	A	0.40	0
2	B	0.35	1
3	C	0.20	0
4	D	0.10	1
5	E	0.05	0
```

这里 observed relevance rel_i 就用 binary booking label：

	•	booked = 1
	•	not booked = 0

DCG

$DCG = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$

所以：
$DCG = \frac{0}{\log_2 2} + \frac{1}{\log_2 3} + \frac{0}{\log_2 4} + \frac{1}{\log_2 5}$

IDCG

理想排序应该把两个 booked item 放最前面：
```
ideal rank	rel
1	         1
2	         1
3	         0
4	         0
5	         0
```
$IDCG = \frac{1}{\log_2 2} + \frac{1}{\log_2 3}$

NDCG

$NDCG = \frac{DCG}{IDCG}$

⸻

面试怎么解释

In the binary case, I can treat booking as a relevance label of 1 and non-booking as 0. For each search request, I sort the candidates by model score, compute DCG based on those binary labels, then normalize by the ideal ordering to get NDCG.

⸻

3) Graded observed label 场景怎么做 NDCG

这通常更实用，因为 booking 太 sparse。
你可以定义 graded relevance，例如：

	•	no interaction = 0
	•	click = 1
	•	save / long dwell = 2
	•	booking = 3

或者更业务化：

	•	bounce = 0
	•	click = 1
	•	checkout intent = 2
	•	booking = 4

DCG 的常见公式

有两种写法：

简单版
$DCG = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$

更常见的 gain 版
$DCG = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$

这个更能放大高相关样本的重要性。

例子

某 query 下模型排序后 observed graded labels 是：
```
rank	rel
1	    1
2	    3
3	    0
4	    2
5	    0
```
用 gain 版：

$DCG = \frac{2^1 -1}{\log_2 2} + \frac{2^3 -1}{\log_2 3} + \frac{2^0 -1}{\log_2 4} + \frac{2^2 -1}{\log_2 5}$

再算 ideal ordering 的 IDCG，最后归一化。

⸻

面试怎么解释

In the graded case, instead of using binary booking labels only, I can assign higher relevance weights to stronger user outcomes, for example click < save < booking. Then NDCG measures whether the model ranks more valuable outcomes earlier in the list, which is often more stable than relying on sparse booking-only labels.

⸻

4) Binary vs graded，什么时候用哪个？

Binary NDCG

适合：

	•	north star 明确是 booking
	•	label 足够多
	•	想直接对齐 business outcome

Graded NDCG

适合：

	•	booking 太 sparse
	•	需要更稳定 offline metric
	•	想利用 click/save/dwell 等 richer signal

最成熟的回答

In practice, I’d likely use both. Binary booking-based NDCG is more directly aligned to the final business objective, while graded NDCG is often more stable offline because it leverages denser intermediate signals.

⸻

你可以直接说的总结版

For offline evaluation, I would not rely on just one metric.
For sparse conversion labels such as booking, PR AUC is often more informative than ROC AUC because it focuses on the rare positive class.
Since this is a ranking problem, I would also compute NDCG within each search request. In the simplest case, I can use binary booking labels as relevance values. In a more stable setup, I can define graded relevance, for example click < save < booking, and compute graded NDCG.
For feature drift, scalar features can be monitored using PSI, while embeddings need additional monitoring such as centroid drift, norm distribution, nearest-neighbor stability, and downstream retrieval overlap.
If drift is detected, I would distinguish between upstream data bugs and real distribution shifts, then either fix the pipeline, retrain the downstream model, recalibrate, or roll back the embedding version if needed.

⸻
