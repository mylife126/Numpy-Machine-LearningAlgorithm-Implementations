from collections import defaultdict
import heapq


class Solution(object):
    def findCheapestPrice(self, n, flights, src, dst, k):
        """
        :type n: int
        :type flights: List[List[int]]
        :type src: int
        :type dst: int
        :type k: int
        :rtype: int
        """
        # set up the adjencency graph first
        adjacency_list = defaultdict(list)
        for flight in flights:
            start, stop, price = flight[0], flight[1], flight[2]
            adjacency_list[start].append((stop, price))

        # then we will use the minheap to maintain the status
        # it tracks the current total cost, current city, and the total stops
        min_heap = [(0, src, 0)]
        heapq.heapify(min_heap)

        # then we can track the visited city and its stops
        visited_stops = dict()

        while min_heap:
            current_cost, current_city, stop_used = heapq.heappop(min_heap)
            # the min heap always ensures the cost at the current city is the minimal and it will be wihthin the minimal k stop using the statemnt below later
            if current_city == dst:
                return current_cost

            if stop_used > k:
                continue

                # 我之前已经用更少（或者相同）航班数到过这个城市了
            # 如果以前已经更优，那么当前状态才应该被剪掉。
            if current_city in visited_stops and visited_stops[current_city] <= stop_used:
                continue

            visited_stops[current_city] = stop_used

            for item in adjacency_list[current_city]:
                neighbor_city, ticket_price = item[0], item[1]
                new_cost = current_cost + ticket_price
                heapq.heappush(min_heap, (new_cost, neighbor_city, stop_used + 1))

        return -1

# start 0, 0, 0
# 0 -> 1 queue(100, 1, 1), then 0->2 queue(500, 2, 1)
# then min heap pops (100, 1, 1)
# 1 stop <= K then track the city 1 's travel stop to be 1
# -> go to next city from 1, which is 1, 2, 100
# queueu(200, 2, 2)
# pop, meets the condition, return