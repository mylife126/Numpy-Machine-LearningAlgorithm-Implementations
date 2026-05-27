"""
queue 里面放入 openable rooms
然后对于这个room里面的key 放入queue中，因为这个key对应了一个room
同时记录 访问过的room

那么bfs的展开则是按照每进去一个屋子后拿到了多少钥匙来展开

升级的题目则是 max candy from boxes 它需要维护的状态更多
"""

from collections import deque


class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        if not rooms or len(rooms) == 0:
            return False

        # 用来记录访问过的屋子
        visited_rooms = set()

        # obtained keys
        obtained_keys = set()

        queue = deque()

        # 永远能进入第一个屋子
        queue.append(0)

        while (queue):
            current_room = queue.popleft()
            if current_room in visited_rooms:
                continue

            # otherwise we visit that room
            for key in rooms[current_room]:
                if key in obtained_keys:
                    continue

                queue.append(key)
                obtained_keys.add(key)

            visited_rooms.add(current_room)

        if len(visited_rooms) == len(rooms):
            return True

        return False