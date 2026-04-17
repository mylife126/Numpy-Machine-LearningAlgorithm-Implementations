

## 1️⃣ 中文梗概（先快速建立框架）

👉 核心问题：

企业里不同数据源经常冲突：

	•	doc 说 A
	•	ticket 说 B
	•	meeting note 说 C

⸻

👉 普通 RAG 的问题：

❌ 把这些拼在一起 → LLM自己选
❌ 或直接“平均” → hallucination
❌ 不解释来源 → 不可信

⸻

👉 Gaia 要做的是：

把“冲突”当作一等公民来处理

⸻

✅ 正确解决方案 = 三步

1️⃣ Claim extraction（拆成结构化事实）

	•	每个 source → claim
	•	带 metadata（time / owner / source type）

⸻

2️⃣ Evidence-based resolution（不是LLM拍脑袋）

用三个核心信号：

	•	recency（新旧）
	•	authority（权威性）
	•	corroboration（一致性）

⸻

3️⃣ 决策机制（最关键）

	•	confidence高 → resolve并记录 rationale
	•	confidence低 →
	•	abstain（不回答）
	•	或 ask clarification
	•	或 escalate

⸻

👉 一句话总结：

Don’t collapse conflicts — reason over them

⸻

2️⃣ 🎯 你直接说（英文，可复述 + example）

⸻

I think knowledge conflict resolution is a core requirement for this system, because in real organizations, different sources frequently disagree.

Instead of treating it as a standard retrieval problem, I would explicitly model it as an evidence reasoning problem.

My approach would have three steps.

First, I would extract structured claims from each source.
For example, a document might say the blocker is infrastructure, a recent ticket might say it is already fixed, and a meeting note might indicate the real issue is legal approval.
Each of these is treated as a separate claim with metadata such as timestamp, source type, and ownership.

Second, I would perform evidence-based resolution.
Instead of letting the model implicitly decide, I would evaluate each claim based on recency, authority, and corroboration.
For example, a recent ticket from the system of record might be more reliable than an outdated document, while multiple sources agreeing on a claim increases confidence.

Third, I would introduce a decision policy based on confidence.
If the system has high confidence, it resolves the conflict and records the rationale in the Case File.
If confidence is low, the system should not guess — it can either ask a targeted clarification question or escalate to a human.

Let me give a concrete example.

In the first iteration, the system concludes that the blocker is infrastructure based on documents and dashboards.
This result is stored in the Case File.

In a later iteration, a new ticket indicates the issue has been fixed, and a meeting note suggests legal approval is now the main blocker.

The system extracts these as new claims, evaluates their recency and authority, and determines that the previous assumption is outdated.
It then updates the Case File, invalidates the blocker node, and re-runs downstream nodes such as recovery planning.

If the evidence is still conflicting, the system would explicitly surface that uncertainty, for example by asking the user: “There are conflicting signals between recent tickets and meeting notes — can you confirm whether legal approval is the current blocker?”

So overall, instead of collapsing conflicting knowledge into a single answer, the system reasons over evidence, maintains uncertainty when needed, and ensures downstream workflows do not rely on incorrect assumptions.

⸻

3️⃣ 可能追问的问题

⸻

❓ Why not just let LLM decide?

👉

Because LLMs tend to collapse conflicting information into a single answer without exposing uncertainty.
In enterprise workflows, this can propagate incorrect assumptions into execution, which is much more costly.

⸻

❓ How do you compute confidence?

👉

I would combine multiple signals such as recency, source authority, agreement across sources, and validation signals.
In practice, this can be implemented as a scoring function or a lightweight model.

⸻

❓ What happens if you choose the wrong claim?

👉

That’s why we store the decision rationale and maintain traceability.
If new evidence arrives, we can invalidate the affected nodes and re-execute downstream steps.

⸻

❓ How is this different from ranking?

👉

Ranking selects the most relevant documents, while conflict resolution operates at the claim level and reasons about correctness and consistency across sources.

⸻

4️⃣ 技术讲解（帮你真正掌握）

⸻

1️⃣ 核心不是 retrieval，是 reasoning

你可以记：

RAG → find info
Conflict resolution → decide what is true (or uncertain)


⸻

2️⃣ claim-level abstraction（很重要）

不要直接处理 doc，而是：

Doc → claims

例如：

Doc A: blocker = infra
Ticket B: blocker = fixed
Meeting C: blocker = legal


⸻

3️⃣ 三个核心 scoring signal
```
信号	含义

recency	    最新信息更可信
authority	system of record / owner
corroboration	多源一致性
```

⸻

4️⃣ 决策策略（非常关键）

不是 binary：

resolve / abstain / ask / escalate


⸻

5️⃣ 和 graph execution 的连接

👉 这是关键连接点（面试加分）

	•	conflict → invalidate node
	•	update Case File
	•	re-run affected subgraph

⸻

6️⃣ 和你背景的连接（强烈建议说）

你可以说：

This is similar to how we handle conflicting signals in ranking systems, where we don’t rely on a single feature, but combine multiple signals with learned or rule-based weighting.

