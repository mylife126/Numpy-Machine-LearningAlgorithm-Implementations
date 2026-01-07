"""
358. Rearrange String k Distance Apart
Solved
Hard
Topics
conpanies icon
Companies
Given a string s and an integer k, rearrange s such that the same characters are at least distance k from each other. If it is not possible to rearrange the string, return an empty string ""."
"""

"""
Rearrange String k Distance Apart

Idea:
- Use a max heap to always pick the most frequent available character.
- After we used a character, it needs to wait k steps (cooldown) before we can use it again.
- We keep a queue of (count, ch, next_available_time) for cooling characters.
- IMPORTANT: At each step, first release all characters whose cooldown is done (time <= step),
  then pick from the heap. If heap is empty and no one can be released -> impossible.
"""
from collections import Counter, deque
import heapq

class Solution(object):
    def rearrangeString(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        if k == 0:
            return s  # no distance requirement

        # 1) count frequency
        freq = Counter(s)

        # 2) build max heap (store as negative count for Python min-heap)
        heap = [(-count, ch) for ch, count in freq.items()]
        heapq.heapify(heap)

        # 3) cooldown queue: (count, ch, next_available_time)
        cooldown = deque()

        result = []
        step = 0

        # keep going while we still have chars in heap or in cooldown
        while heap or cooldown:

            # (A) first, release all that are ready at this step
            while cooldown and cooldown[0][2] <= step:
                ready_count, ready_ch, _ = cooldown.popleft()
                heapq.heappush(heap, (ready_count, ready_ch))

            # (B) pick the best available char
            if heap:
                count, ch = heapq.heappop(heap)
                result.append(ch)
                # we used one instance -> increase negative count by +1
                count += 1
                # if still remaining, put it into cooldown with next available time = step + k
                if count < 0:
                    cooldown.append((count, ch, step + k))
            else:
                # heap empty but still have items cooling (can't place idle slots) -> impossible
                return ""

            step += 1

        return "".join(result)

'''
a @ 3, step = 1
b @ 3, step = 2
c @ c, step = 3
heap is empty

Thus, we must release the eligibile item from cooldown heap to the max heap! before check the maxheap for the next char
'''