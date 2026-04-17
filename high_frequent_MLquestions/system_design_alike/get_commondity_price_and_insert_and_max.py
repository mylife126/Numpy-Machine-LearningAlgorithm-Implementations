"""
VO1: Backend Coding

build an in-memory solution that supports two operations on a stream of (timestamp, commodityPrice) pairs:
1. upsertCommodityPrice(timestamp, price)
if the timestamp already exists, update its price.
Otherwise, insert the new (timestamp, price) pair.
2. getMaxCommodityPrice()
Return the maximum commodity price across all current timestamps.
"""
import heapq


class CommodityPriceStore(object):
    def __init__(self):
        self.time2price = {}
        self.max_price_heap = []
        heapq.heapify(self.max_price_heap)

    def upsertCommodityPrice(self, timestamp, price):
        self.time2price[timestamp] = price

        heapq.heappush(self.max_price_heap, (-price, timestamp))

    def getMaxCommodityPrice(self):
        # edge case where there is no price entered yet
        if len(self.time2price) == 0:
            return None

        while self.max_price_heap:
            # Should not do the below to pop, because the heapq is in memory, thus, once you pop it you lost the result.
            # If you want to check the max again, you lose the data!!
            # price, timestamp = heapq.heappop(self.max_price_heap)

            price, timestamp = self.max_price_heap[0]  # peek only
            price = -price

            if timestamp in self.time2price:
                its_latest_price = self.time2price[timestamp]
                if its_latest_price == price:
                    return price

                # otherwise the price is already being updated
                # do nothing -> change from do nothing to pop the stale value
                heapq.heappop(self.max_price_heap)
        return None


"""
follow-up:

1.upsertCommodityPrice(timestamp, price)
- Assigns a new checkpoint ID, returned as int.
- Internally records (timestamp, price) at that checkpoint.

2. getCommodityPrice(timestamp, checkpoint)
- Returns the price that was current as of that checkpoint.

"""

import heapq
from collections import defaultdict


class CommodityPriceStoreV2(object):
    def __init__(self):
        self.time2price = {}
        self.max_price_heap = []
        self.checkpoint = 0
        self.timestamp2checkpoint2price = defaultdict(list)
        heapq.heapify(self.max_price_heap)

    def upsertCommodityPrice(self, timestamp, price):
        self.time2price[timestamp] = price
        checkpoint = self.checkpoint
        self.timestamp2checkpoint2price[timestamp].append([checkpoint, price])
        self.checkpoint += 1
        heapq.heappush(self.max_price_heap, (-price, timestamp))

    def getCommodityPrice(self, timestamp, checkpoint):
        # if there is no history for this timestamp's history
        if timestamp not in self.timestamp2checkpoint2price:
            return None

        checkpoint_price_list = self.timestamp2checkpoint2price[timestamp]

        # operate the binary search
        left_point = 0
        right_point = len(checkpoint_price_list) - 1
        answer = None
        while left_point <= right_point:
            mid_point = (left_point + right_point) // 2
            the_midpoint_checkpoint, the_midpoint_price = checkpoint_price_list[mid_point][0], \
                checkpoint_price_list[mid_point][1]

            if the_midpoint_checkpoint <= checkpoint:
                answer = the_midpoint_price
                left_point = mid_point + 1

            else:
                right_point = mid_point - 1
        return answer

    def getMaxCommodityPrice(self):
        # edge case where there is no price entered yet
        if len(self.time2price) == 0:
            return None

        while self.max_price_heap:
            # Should not do the below to pop, because the heapq is in memory, thus, once you pop it you lost the result.
            # If you want to check the max again, you lose the data!!
            # price, timestamp = heapq.heappop(self.max_price_heap)

            price, timestamp = self.max_price_heap[0]  # peek only
            price = -price

            if timestamp in self.time2price:
                its_latest_price = self.time2price[timestamp]
                if its_latest_price == price:
                    return price

                # otherwise the price is already being updated
                # do nothing -> change from do nothing to pop the stale value
                heapq.heappop(self.max_price_heap)
        return None


if __name__ == '__main__':
    myStore = CommodityPriceStoreV2()
    print(myStore.getMaxCommodityPrice())

    myStore.upsertCommodityPrice(123, 456)
    myStore.upsertCommodityPrice(123, 45)
    print(myStore.getMaxCommodityPrice())

    myStore.upsertCommodityPrice(125, 45999)
    print(myStore.getMaxCommodityPrice())

    print(myStore.getCommodityPrice(123, 0))
