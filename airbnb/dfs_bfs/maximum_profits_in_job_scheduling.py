"""
1235. Maximum Profit in Job Scheduling

We have n jobs, where every job is scheduled to be done from startTime[i] to endTime[i],
obtaining a profit of profit[i].

You're given the startTime, endTime and profit arrays,
return the maximum profit you can take such that there are no two jobs in the subset with overlapping time range.

If you choose a job that ends at time X you will be able to start another job that starts at time X.
"""
"""
Solution 1: 用DFS和Memo来剪枝，对每一个job做以下操作

对于每一个job来说，它都有两个选择
1. 拿到当前的profit 然后找到它自己之后的第一个valid的job这里的valid 要判断为 next job的start time 大于等于当前的end time

2. 不做当前的job， skip， 去看它下一个job可能带来的最大收益

那么一定有一个问题，为什么不是找到第一个valid 最大收益，而是greedy的找到第一个valid job呢？ 因为我们还要加一个判断
max = max(take+dfs(next_valid), dfs(current_index + 1))

那么其实dfs会不断遍历下去找到最大解答。 例如
jobs:
(1,3,50)   ← 当前
(2,4,10)   ❌ overlap
(3,5,40)   ✅
(3,6,70)   ✅
(5,8,100)  ✅


binary search 返回： next_index = 2
我们不会直接选择 index=2的profit而是
takeprofit + dfs(2) 
而dfs（2）会考虑：
    - 选 job2
    - 不选 job2 → 看 job3
    - 不选 job3 → 看 job4


当前 job = (1,3,50)
next_index = 2
dfs(2):
    ├── take (3,5,40) → 40
    ├── skip → (3,6,70) → 70
    ├── skip → (5,8,100) → 100
    ...
    → 返回最大
dfs(2) = max(所有合法组合)

所以每一个dfs里，都做以上两个事情。

那么假设：
jobs = [
(1,3,50),
(2,4,10),
(3,5,40),
(3,6,70)
]

job1 = dfs（index=0）
         --------> option 1 take: 50 + dfs(first_next_valid)
         --------> option 2 skip: dfs(current + 1)

return max（option1， option2） 
最后能得到每一个 state， dfs(1) dfs(2) dfs(3) 等等的最大profits
"""
"""
Solution 1: 用DFS和Memo来剪枝，对每一个job做以下操作

对于每一个job来说，它都有两个选择
1. 拿到当前的profit 然后找到它自己之后的第一个valid的job这里的valid 要判断为 next job的start time 大于等于当前的end time

2. 不做当前的job， skip， 去看它下一个job可能带来的最大收益

那么一定有一个问题，为什么不是找到第一个valid 最大收益，而是greedy的找到第一个valid job呢？ 因为我们还要加一个判断
max = max(take+dfs(next_valid), dfs(current_index + 1))

那么其实dfs会不断遍历下去找到最大解答。 例如
jobs:
(1,3,50)   ← 当前
(2,4,10)   ❌ overlap
(3,5,40)   ✅
(3,6,70)   ✅
(5,8,100)  ✅


binary search 返回： next_index = 2
我们不会直接选择 index=2的profit而是
takeprofit + dfs(2) 
而dfs（2）会考虑：
    - 选 job2
    - 不选 job2 → 看 job3
    - 不选 job3 → 看 job4


当前 job = (1,3,50)
next_index = 2
dfs(2):
    ├── take (3,5,40) → 40
    ├── skip → (3,6,70) → 70
    ├── skip → (5,8,100) → 100
    ...
    → 返回最大
dfs(2) = max(所有合法组合)

所以每一个dfs里，都做以上两个事情。

那么假设：
jobs = [
(1,3,50),
(2,4,10),
(3,5,40),
(3,6,70)
]

job1 = dfs（index=0）
         --------> option 1 take: 50 + dfs(first_next_valid)
         --------> option 2 skip: dfs(current + 1)

return max（option1， option2） 
最后能得到每一个 state， dfs(1) dfs(2) dfs(3) 等等的最大profits
"""

import bisect
class Solution(object):
    def jobScheduling(self, startTime, endTime, profit):
        """
        Return the maximum profit from non-overlapping jobs
        """

        # ------------------------------------------------
        # Step1: combine jobs and sort by start time
        # ------------------------------------------------
        jobs = list(zip(startTime, endTime, profit))
        jobs.sort(key=lambda x:x[0])

        sorted_start_times = sorted(job[0] for job in jobs)

        # set a memo for dfs(index)
        memo = {}

        # create the main DFS function to operate 2 tasks
        def dfs(job_index):
            """
            to find each state's max profit
            """

            # first, create the return condition
            # 当job index >= len(job) 则说明没有下一个job了
            # 我们的binary search， 当没有下一个的时候，return的就是len（job)
            if job_index >= len(jobs):
                return 0

            if job_index in memo:
                return memo[job_index]

            # Task 1, skip the current job to rely on next job's max profit
            skip_profit = dfs(job_index + 1)

            # Task 2, take the current profit, and look out for the next first valid job's profit
            current_start, current_end, current_profit = jobs[job_index]

            # use binary search to find the next valid job, whose start time is not overlapping with the current
            # 如果没有下一个了，则return了len（sorted start times）
            next_index = bisect.bisect_left(sorted_start_times, current_end)

            take_profit = current_profit + dfs(next_index)

            this_state_max_profit = max(skip_profit, take_profit)

            memo[job_index] = this_state_max_profit

            return this_state_max_profit

        return dfs(0)

import bisect


class Solution(object):
    def jobScheduling(self, startTime, endTime, profit):
        """
        Bottom-up DP solution for weighted interval scheduling
        """

        # ------------------------------------------------
        # Step1: combine and sort jobs by start time
        # ------------------------------------------------
        jobs = list(zip(startTime, endTime, profit))
        jobs.sort(key=lambda x: x[0])

        total_jobs = len(jobs)

        # extract sorted start times for binary search
        sorted_starts = [job[0] for job in jobs]

        # ------------------------------------------------
        # Step2: precompute next_index for each job
        # next_index[i] = first job index with start >= end[i]
        # ------------------------------------------------
        next_index = [0] * total_jobs

        for i in range(total_jobs):
            current_end = jobs[i][1]

            # binary search to find next valid job
            next_index[i] = bisect.bisect_left(sorted_starts, current_end)

        # ------------------------------------------------
        # Step3: dp array
        # dp[i] = max profit starting from job i
        # we use size n+1 to handle base case dp[n] = 0 也就是说假设从i开始话，最大profit是多少，那么bottom up的思想就能成立，因为对于最后一个job 它没有下家，它最大profit就是自己，然后往前推 i -1 就存在两个选择，一个是选择自己的profit 或者 skip 自己拿下家的
        #  ------------------------------------------------
        dp = [0] * (total_jobs + 1)
        print(dp)
        # ------------------------------------------------
        # Step4: fill dp from right to left
        # ------------------------------------------------
        for i in range(total_jobs - 1, -1, -1):

            current_start, current_end, current_profit = jobs[i]

            # option1: skip current job
            skip_profit = dp[i + 1]

            # option2: take current job
            take_profit = current_profit + dp[next_index[i]]

            print(skip_profit, take_profit)
            # choose the better one
            dp[i] = max(skip_profit, take_profit)

        # answer is dp[0]
        return dp[0]