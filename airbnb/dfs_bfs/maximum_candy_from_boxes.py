from collections import deque


class Solution(object):
    def maxCandies(self, status, candies, keys, containedBoxes, initialBoxes):
        """
        :type status: List[int]
        :type candies: List[int]
        :type keys: List[List[int]]
        :type containedBoxes: List[List[int]]
        :type initialBoxes: List[int]
        :rtype: int

        核心问题是， 我们得track几个status：
        1. 拥有的盒子 represented by list of indicies， box gained
        2. 拥有的key represented by list of indicies， keys gained
        3. 已经打开的盒子， 防止循环， represented by indicies

        所以这里的trick就是:
        1. 先看key， 如果key in boxed_gained ， 因为都是index representation， meaning key = 1 意味着box 1 有钥匙，那么_boxed_gained 有1 就意味着我们也有box 1.
           -》 如果key 在gained box里 且 key不在opened box里
                加入该key == box to queue

        2. 在看新获得的box
           -》 把 new box加入box gained
           在看新box在不在key set里， 如果在且没有被打开则把这个new box 放入queue。  or 这个new box本身就是开的，且没有被打开，则放入

        那么在bfs里 每次pop出来的都是可以打开的盒子，就可以找到其对应的candies， candies ++
        """
        # track the indicies meaning the box we have
        box_gained = set()

        # track the keys we gained
        keys_gained = set()

        # track the opened boxes to prevent the loop or double count
        opened = set()

        # simulate the process
        queue = deque()

        for box in initialBoxes:
            if status[box] == 1:
                queue.append(box)
            box_gained.add(box)

        total_candies = 0

        while queue:
            # get the latest opened/openable boxes
            current_box = queue.popleft()

            if current_box in opened:
                continue

            opened.add(current_box)

            its_candies = candies[current_box]
            total_candies += its_candies

            # -------------
            # get new keys
            # -------------
            for key in keys[current_box]:
                keys_gained.add(key)

                # now check if the key has the box paied/ gained already, if the requirements met, then this is a openable box
                if key in box_gained and key not in opened:
                    queue.append(key)

            # --------
            # Get box
            # --------
            for new_box in containedBoxes[current_box]:
                box_gained.add(new_box)

                # now check the status of the new_box
                if (status[new_box] == 1 or new_box in keys_gained) and new_box not in opened:
                    queue.append(new_box)

        return total_candies