"""
You have n boxes labeled from 0 to n - 1. You are given four arrays: status, candies, keys, and containedBoxes where:

status[i] is 1 if the ith box is open and 0 if the ith box is closed,
candies[i] is the number of candies in the ith box,
keys[i] is a list of the labels of the boxes you can open after opening the ith box.
containedBoxes[i] is a list of the boxes you found inside the ith box.
You are given an integer array initialBoxes that contains the labels of the boxes you initially have.
You can take all the candies in any open box and you can use the keys in it to open new boxes and you also can use the boxes you find in it.

Return the maximum number of candies you can get following the rules above.



Example 1:

Input: status = [1,0,1,0], candies = [7,5,4,100], keys = [[],[],[1],[]], containedBoxes = [[1,2],[3],[],[]], initialBoxes = [0]
Output: 16
Explanation: You will be initially given box 0. You will find 7 candies in it and boxes 1 and 2.
Box 1 is closed and you do not have a key for it so you will open box 2. You will find 4 candies and a key to box 1 in box 2.
In box 1, you will find 5 candies and box 3 but you will not find a key to box 3 so box 3 will remain closed.
Total number of candies collected = 7 + 4 + 5 = 16 candy.
Example 2:

Input: status = [1,0,0,0,0,0], candies = [1,1,1,1,1,1], keys = [[1,2,3,4,5],[],[],[],[],[]], containedBoxes = [[1,2,3,4,5],[],[],[],[],[]], initialBoxes = [0]
Output: 6
Explanation: You have initially box 0. Opening it you can find boxes 1,2,3,4 and 5 and their keys.
The total number of candies will be 6.

这里的问题核心是 你只有先拿到箱子， 然后再看有没有配对的钥匙才能打开这个箱子。

所以逻辑还是bfs， bfs 存入的是可以打开的像一个箱子index。
而展开访问的边则是：
1. 先收集里面的箱子， 以及收集里面的keys
2. 再看目前收集的箱子里几个箱子是有配对的钥匙的话 或者 该新拿到的box就是可以打开的了，那么加入queue 在下一次bfs展开 但是这个箱子没有被上一层bfs打开过

"""
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
        """
        # maintain all collected box for later key check
        collected_boxes = set()

        # maintain all keys collected along the way
        collected_keys = set()

        # to prevent revisit the opened box to reduce the computation cost
        visited_boxes = set()

        queue = deque()
        for box in initialBoxes:
            if status[box]  == 1:
                queue.append(box)

            # 注意啊 这里必须init 把最开始的initial箱子加进去 不然会漏， 有一个情况是 一开始给你的箱子有几个打不开，但后续也许可以打开
            collected_boxes.add(box)

        num_candies = 0
        while queue:
            current_box = queue.popleft()
            if current_box in visited_boxes:
                continue

            its_candies = candies[current_box]
            num_candies += its_candies

            # step 1 get its keys, this is because keys are the prerequisite of the box to be opened or not
            all_new_keys = keys[current_box]

            # step 2 维护key
            for key in all_new_keys:
                # 收集key先
                collected_keys.add(key)

                # 同时， 如果此刻我们collected box里已经有这个key对应的box了， 且还没有打开 那它可以被打开 放在下一层bfs
                if key in collected_boxes and key not in visited_boxes:
                    queue.append(key)

            # step 3 维护新拿到的box
            all_its_new_boxes = containedBoxes[current_box]
            for new_box in all_its_new_boxes:
                collected_boxes.add(new_box)
                if new_box in visited_boxes:
                    continue

                if status[new_box] == 1 or new_box in collected_keys:
                    queue.append(new_box)

            visited_boxes.add(current_box)

        return num_candies