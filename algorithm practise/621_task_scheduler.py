"""
621. Task Scheduler
Solved
Medium
Topics
conpanies icon
Companies
Hint
You are given an array of CPU tasks, each labeled with a letter from A to Z, and a number n. Each CPU interval can be idle or allow the completion of one task. Tasks can be completed in any order, but there's a constraint: there has to be a gap of at least n intervals between two tasks with the same label.

Return the minimum number of CPU intervals required to complete all tasks.
"""

"""
思路是，类似之前那个358 rearrage string such that each letter is k distance away from each

1. prioritize the most frequent jobs, so set a max heap, it saves [-count, task]
2. then maintain a cool down deque, for the finished job, put it into the deque, track this task's next available time = n + time; 
only when the the top left cooling task's time has reached the new available time, put it back to the priority queue 
the time into total to finish off the two queues are the final needed intervals
"""

from collections import Counter, deque
import heapq

class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """

        # Step 1: count frequency of tasks
        frequency = Counter(tasks)

        # Step 2: build max heap (store as negative)
        heap = [(-count, task) for task, count in frequency.items()]
        heapq.heapify(heap)

        # Step 3: initialize time and cooldown
        time_now = 0
        cooldown = deque()  # each item: (available_time, count, task)

        # Step 4: run scheduling
        while heap or cooldown:
            time_now += 1  # each loop = one unit time

            # Step 4.1: execute top task if available
            if heap:
                count, task = heapq.heappop(heap)
                count += 1  # because count is negative
                if count != 0:
                    # put into cooldown queue
                    its_next_available = time_now + n
                    cooldown.append((its_next_available, count, task))

            # Step 4.2: release all tasks whose cooldown is done
            while cooldown and cooldown[0][0] == time_now:
                available_time, count, task = cooldown.popleft()
                heapq.heappush(heap, (count, task))

        return time_now
