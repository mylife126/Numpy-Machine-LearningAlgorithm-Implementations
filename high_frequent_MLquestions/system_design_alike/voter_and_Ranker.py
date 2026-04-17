"""
	•	你要设计一个 in-memory Voting data structure
	•	有操作：录入一张投票（投票里有第 1/2/3 名）
	•	计分规则：
	•	第 1 名 +3 分
	•	第 2 名 +2 分
	•	第 3 名 +1 分
	•	需要一个函数能 输出当前排名（按总分从高到低）
-（常见 tie-break：分数相同按名字字典序；我会默认这么做，面试可一句话说明）

我们只需要维护两件事：
	1.	每个候选人的总分

	•	用 name -> score 的 dict
	•	每次来一票：对 top1/top2/top3 分别加 3/2/1

	2.	输出排名

	•	把所有候选人按（分数降序，名字升序）排序
	•	输出排序后的列表即可

为什么不需要更复杂的数据结构？
	•	因为 update 很简单：O(1) 改一个人的分
	•	排名查询通常不会在每次投票后都要极致快（面试 V1 用排序即可）
	•	如果面试官追问“getRank 频繁”，再升级成 heap / balanced tree
"""
from collections import defaultdict
class VoterAndRanker:
    def __init__(self):
        self.name2score = defaultdict(int)

        # set the scoring policy
        self.rank2points = {
            0:3,
            1:2,
            2:1,
        }

    def addVote(self, top3_names):
        """
        we only consider the top3 name, because we only have 3 score conditions
        """
        if not top3_names:
            return

        limit = min(3, len(top3_names)) # make sure we only record the 3 names

        for i in range(limit):
            name = top3_names[i]
            points = self.rank2points[i]

            self.name2score[name] += points

    def getRanking(self):
        items = []
        for name in self.name2score:
            items.append((name, self.name2score[name]))

        items.sort(key=lambda x: (-x[1], x[0]))

        ranking = []
        for name, score in items:
            ranking.append((name, score))

        return ranking

    def getScore(self):
        # just for debugging
        return dict(self.name2score)


## Request, if we are going to frequently add the votes, how to use the heapq to dynamically sort but with minimal insertion
"""
逻辑是 一直在heapq里添加最新的 name score， 这里面肯定有stale的但是没关系 我们不做查找和删除， 在最后我们while heap， 不断peek最高分数的人
然后对比字典里的name to score是否一致，如果一致则说明这个是最高分，不然则说明这个是stale的分数
"""

from collections import defaultdict
import heapq
class VoterAndRankerV2:
    def __init__(self):
        self.name2score = defaultdict(int)

        # set the scoring policy
        self.rank2points = {
            0:3,
            1:2,
            2:1,
        }

        self.max_heap = []

    def _push_current(self, name):
        latest_score = self.name2score[name]
        heapq.heappush(self.max_heap, (-latest_score, name))

    def addVote(self, top3_names):
        """
        we only consider the top3 name, because we only have 3 score conditions
        """
        if not top3_names:
            return

        limit = min(3, len(top3_names)) # make sure we only record the 3 names

        for i in range(limit):
            name = top3_names[i]
            points = self.rank2points[i]
            self.name2score[name] += points
            self._push_current(name)

    def getRanking(self):
        if len(self.name2score) == 0:
            return

        while self.max_heap:
            neg_score, name = self.max_heap[0] # peek
            score = -neg_score

            current_score = self.name2score[name]
            if current_score == score:
                return (name, score)

            # otherwise this is a stale record
            heapq.heappop(self.max_heap)

    def getScore(self):
        # just for debugging
        return dict(self.name2score)



if __name__ == "__main__":
    voter = VoterAndRanker()
    voter.addVote(["A", "B", "C", "D"])
    voter.addVote(["A", "B", "C"])
    voter.addVote(["B", "C", "D"])
    print(voter.getRanking())



    voter = VoterAndRankerV2()
    voter.addVote(["A", "B", "C", "D"])
    voter.addVote(["A", "B", "C"])
    voter.addVote(["B", "C", "D"])
    print(voter.getRanking())