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
## 容易被问到的问题
可以。我先把你这轮面试里被 challenge 的核心问题抽出来，再给你每个问题一版更好的回答，最后给你一组明天还能直接用的话术建议。

⸻

### 一、这轮面试里你真正被 challenge 的点

我看下来，主要有 4 个。

1. Non-personalized ranker 的数据怎么收集、怎么切 train/test

你当时说的是 split by customer。
这不是完全错，但不够贴题。因为 ranker 的基本预测单位不是 customer，而是一次 search request 下的一组 candidates。

面试官纠结这个问题，说明他想确认你有没有真正理解：

	•	ranker 的 training instance 是什么
	•	label 是怎么来的
	•	为什么 test set 要模拟真实线上分布
	•	为什么不能随便 random split

⸻

2. Sparse / high-cardinality feature（例如 amenities）怎么处理

你给了三个方向：

	•	从模型复杂度解决
	•	heuristic 按 locale 选 feature
	•	embedding self-denoise

这里的问题不是你没思路，而是表述还不够硬。
面试官更想听的是：

	•	稀疏问题本质是什么
	•	你用什么具体方法建模
	•	怎么让低频 feature 不学坏
	•	heuristic 是主方案还是补充方案

⸻

3. 为什么 host / listing / query 放进 universal embedding，而 personalized 单独拿出去

你回答了：

	•	这几个 non-personalized feature 有 inherent correlation
	•	personalized 单独拆出去可以减少 latency
	•	后面如果有问题再引入 gating / 更多 head

方向是对的，但还可以更清晰。
面试官这里其实在看你是否能讲明白：

	•	建模角色为什么不同
	•	系统设计上为什么拆开更合理
	•	为什么不是一开始全耦合在一起

⸻

4. 你整体节奏不错，但有些地方术语偏“感觉正确”，不够“生产级”

比如：

	•	complexity
	•	self-denoise
	•	heuristic
	•	domain seesaw

这些词不是不能说，而是一旦 interviewer 往下追问，你要能立刻翻译成具体机制。
否则容易给人感觉“方向是对的，但做得不够深”。

⸻

### 二、每个问题的更好回答

下面我按“问题 -> 好回答 -> 技术讲解”的方式整理。

⸻

1）Ranker 的数据收集与 train/test split

这个问题你更好的回答

如果这个 ranker 是 non-personalized，我不会把 customer 当作主要 sample unit。
我会把 一次 search request 当作基本训练单元。

因为 ranker 的任务是：

	•	给定一个 search query / request context
	•	对这一组 candidate listings 排序

所以训练数据应该按 search request group 来组织，而不是按 customer 来组织。

数据收集上，我会从历史 search logs 里收集：

	•	query / context：locale, check-in/out, guests, filters, device
	•	candidate/listing 特征：price, rating, review, amenities, availability 等
	•	exposure 信息：哪些 listing 被展示了，展示位置是什么
	•	label：click, long click, save, booking

训练时我主要使用 exposed items，因为没有曝光的 item 不能直接当负样本。

在 split 上，我的主方案会是：

	•	按 search request group 切分
	•	并且最好 forward in time holdout

也就是：

	•	用较早时间的数据训练
	•	用之后一段时间做 validation
	•	再用更后面的时间做 test

而不是把同一个时期的数据随机打散。

⸻

### 技术讲解（简洁版）

什么叫 split by search request group

一次搜索，比如：

	•	Paris
	•	2 guests
	•	Apr 10–Apr 14
	•	price < 300

系统会生成一组 candidate listings，并展示 top K。
这整个搜索请求就是一个 request group。

一个 group 里面有很多 item。
切分时要保证：

	•	同一个 request_id 下的所有 items 只能出现在 train 或 test 的其中一边
	•	不能一部分在 train，一部分在 test

因为 ranking 是 group task，不是独立样本分类任务。

⸻

什么叫 forward in time holdout

就是 按时间往前切：

	•	Train：过去一段时间
	•	Valid：更近的一段时间
	•	Test：最新的一段时间

例如：

	•	Train: Jan-Feb
	•	Valid: early Mar
	•	Test: late Mar

为什么这么做？

因为线上部署时，你永远是：

	•	用过去训练模型
	•	在未来流量上预测

所以 test set 最好也模拟这个场景。
这就比 random split 更接近真实线上表现。

⸻

为什么不是 split by customer

因为 non-personalized ranker 不依赖 customer history，
所以它要 generalize 的重点不是“新用户”，而是“新的 search requests / future traffic”。

customer split 只能作为补充 sanity check，
不能作为主 split。

⸻

你明天可以直接说的话术

For a either personalized or non-personalized ranker, I’d define the training unit at the search-request level rather than the customer level. I’d train on exposed impression groups with engagement labels, and my main offline evaluation would be a forward-in-time holdout by request group, because that best matches future serving traffic.

⸻

2）Sparse / high-cardinality feature，例如 amenities，怎么处理

这个问题你更好的回答

Amenities 这类 feature 本质上是：

	•	sparse
	•	multi-hot
	•	high-cardinality
	•	long-tail

我不会只用 raw one-hot 去建模，因为这样低频 feature 很难学稳，也没法利用 feature 之间的相似性。

我的主方案会是：

	1.	给 amenity 做 embedding
	2.	对一个 listing 的多个 amenities 做 pooling 比如 sum / mean / attention pooling
	3.	对低频 amenity 做 threshold 或合并到 other bucket
	4.	加 regularization，比如 embedding dropout / L2

如果我有产品 taxonomy，我还会做 hierarchical grouping，
例如把 amenities 归到：

	•	kitchen
	•	family-friendly
	•	workspace
	•	luxury
	•	accessibility

这样模型能在相似 amenities 之间共享统计强度。

至于 locale-specific heuristic，我会把它当作补充 guardrail，不会当主方案。

⸻

技术讲解（简洁版）

为什么 one-hot 不够

因为 amenity 很多，且很多都很稀有。
直接 one-hot 的问题是：

	•	低频 amenity 没有足够样本学权重
	•	“wifi”和“fast wifi”这种相关性学不出来
	•	feature 空间太稀疏

⸻

为什么 embedding 更好

embedding 的好处是：

	•	把离散稀疏特征映射到 dense vector
	•	相似 amenity 可以在 embedding space 里共享信息
	•	比每个 one-hot 单独学一个权重更稳

⸻

heuristic 按 locale 筛特征，好不好

可以提，但要降级成次要方案。
更好的说法是：

	•	冷启动市场可作为先验
	•	数据少时可做 guardrail
	•	长期主方案还是 learned representation

⸻

你明天可以直接说的话术

For sparse multi-hot features like amenities, I would not rely on raw one-hot alone. I’d learn amenity embeddings, pool them into a dense representation, bucket very rare values, and regularize the embedding table. In this way, similar amentities would share similar statistical strength also we can smooth out the semantic similarity such as "wifi" and "high speed wifi". Locale-aware heuristics can help as a fallback, but I would not make them the primary long-term solution.

⸻

3）为什么 universal embedding 放 host/listing/query，而 personalized 单独拆出去 PEPNet

这个问题你更好的回答

我会把 host、listing、query 放到一个 shared non-personalized tower 里，
把 personalized signals 放到单独的 head 里，原因有三层。

第一，建模角色不同

	•	query / listing / host 学的是通用 relevance 和供需匹配
	•	personalized signals 学的是 user-specific residual preference

也就是说：

	•	shared tower 负责“大家都觉得相关不相关”
	•	personalized head 负责“这个用户额外偏不偏好”

第二，系统角色不同

non-personalized features 更稳定，更容易缓存或预计算；
personalized features 更动态、更依赖在线用户状态，也更敏感于 latency。

所以拆出来更容易：

	•	serving
	•	fallback
	•	feature availability 管理
	•	latency 控制

第三，实验和 rollout 更清晰

先上 base universal tower，再加 personalized residual head，
可以清楚看到 personalization 带来的 incremental gain。

如果后面发现 shared representation 出现负迁移，
再引入 gating / MoE / domain-specific residual head。

⸻

技术讲解（简洁版）

shared tower 在学什么

主要是：

	•	semantic match
	•	quality prior
	•	supply-demand relevance

personalized head 在学什么

主要是：

	•	用户短期偏好
	•	用户长期口味
	•	个体 residual

所以两者拆开是合理的，不是“为了拆而拆”。

⸻

你明天可以直接说的话术

I separate the shared query-listing-host tower from personalization because they play different roles: the shared tower learns generic relevance, while the personalization head learns user-specific residual preference. It is also cleaner from a systems perspective because non-personalized features are easier to cache, while personalized features are more dynamic and latency-sensitive.

⸻

4）你被 challenge 的不是方向，而是“术语落地度”

这个问题你更好的修正方式

以后尽量少说这些单独悬空的词：

	•	complexity
	•	self-denoise
	•	heuristic
	•	domain seesaw

说了也没问题，但要立刻接具体机制。

例如：

不要只说

“embedding 可以 self-denoise”

要说

“embedding lets similar sparse categories share statistical strength, and I’d stabilize it with pooling, rare-bucket thresholding, and regularization”

⸻

不要只说

“复杂模型可以解决 sparsity”

要说

“I’d move from raw sparse IDs to embedding-based representation learning, possibly with feature interactions only after I have stable dense representations”

⸻

不要只说

“domain seesaw”

要说

“If the shared tower shows negative transfer or popular-domain dominance, I’d add gating or a domain-specific residual head”

⸻

三、你这轮面试表现里做得好的地方

这个也要保留，因为你明天还要继续用。

1. 你有主线，不是散点输出

你先讲 architecture，再收窄到 ranker，再聊 feature / model，这是对的。

2. 你没有一开始把所有复杂度堆满

这也对。
先讲 v1，再根据 challenge 升级到 v2/v3，这很像真正做系统的人。

3. 你能让 interviewer 引导你往深处走

这说明你不是在背稿，而是在共建答案。这个状态很好。

⸻

四、你明天 ML system design 最值得注意的建议

我给你最实用的，不讲空话。

建议 1：每个设计都先给一个“v1 简洁版”

不要一开始就把所有 fancy 机制全放出来。

开场可以这样说：

I’ll start with a simple production-friendly v1, then I can layer in personalization, debiasing, or more advanced modeling if you want to go deeper.

这句话很好用。

⸻

建议 2：一旦 interviewer 追问，就切到“问题-原因-方案-代价”

例如他说：

“为什么不直接把 personalized 信号也放进 shared tower？”

你就按这个模板答：

	•	问题：可以这么做
	•	原因：但会带来在线依赖和耦合
	•	方案：所以我先 modularize
	•	代价：可能少掉部分 cross-feature gain，后续再看是否值得升级

这个结构非常稳。

⸻

建议 3：少说抽象词，多说机制词

把下面这些词尽量替换：

	•	complexity → representation learning / regularization / modular serving
	•	self-denoise → share statistical strength / reduce sparse overfitting
	•	heuristic → guardrail / prior / fallback
	•	domain seesaw → negative transfer / representation interference

⸻

建议 4：主动给 interviewer 两个下钻方向

这样既显得你有全局，也避免你说太满。

比如：

There are two natural deep dives here: one is data and label construction, and the other is modeling and serving tradeoff. I’m happy to go either direction.

这句话非常适合 system design。

⸻

建议 5：每说一个设计，都补一句“为什么先不上更复杂方案”

这个很 senior。

例如：

I’m not starting with a tightly coupled joint architecture because I want clean ablation, easier debugging, and safer latency control in the first production version.

这类句子会让你显得不是“不会高级方案”，而是“有意不先上”。

⸻

五、你明天可以直接复用的几个高频话术

1. 先讲 v1

I’ll start with a simple v1 that is easy to train, evaluate, and deploy, and then I can extend it if we identify clear failure modes.

2. interviewer challenge 后承接

That’s a good challenge. My initial design is optimized for simplicity and rollout safety, but if that becomes the bottleneck, here is how I would evolve it.

3. 解释为什么拆模块

I’m separating these components because they have different statistical roles and different serving constraints.

4. 解释为什么不用更复杂方案

I’m intentionally not starting with the most coupled architecture, because I want clean ablation, lower latency risk, and better debuggability.

5. 解释数据切分

My offline split should preserve the ranking group structure and ideally simulate future traffic, so I’d split by request group and evaluate with a forward-in-time holdout.

⸻

六、你今晚最该记住的精简版

就记这 4 句。

1  Ranker 的 sample unit 是 search request group，不是 customer。

2  主评估最好是 forward-in-time holdout，而不是纯 random split。

3  Sparse high-cardinality feature 的主方案是 embedding + pooling + rare bucket + regularization。

4  Shared tower 学通用 relevance，personalized head 学 user-specific residual，所以先拆开更合理。

⸻

七、最后给你的一个临场建议

你现在不是“不会答”，而是已经有对的 intuition，只需要再把表述压得更硬、更短、更像 production 决策。

明天你可以刻意做一件事：

每答完一个点，自己补一句 “why this first, why not something more complex yet.”

只要你做到这一点，你整场会更稳。

如果你愿意，我下一条直接给你做一版 “明天 ML system design 面试前 10 分钟速记稿”，只保留最该背的内容。