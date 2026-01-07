"""
286. Walls and Gates
Solved
Medium
Topics
conpanies icon
Companies
You are given an m x n grid rooms initialized with these three possible values.

-1 A wall or an obstacle.
0 A gate.
INF Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

Approach 1:
以gate为 starting的BFS

首先init queue， 而que里添加的是每一个 gate的ij， step init为0

然后开始从gate 做上下左右的扩展，
那么只要能走通，就是一个room， 所以那个room 当作一个新的节点加入queue中 （也就是说 那个room在被pop的时候被当作一个“gate”），且此刻的room当前的step+=1 且就是它所需要的step了

        [
        [room,-1,【0】, room],
        [room,room,room,-1],
        [room,-1,room,-1],
        [0,-1,room,room]
        ]

假设以【0】为例子
先往右走-》 找到room ，step +=1 = 0 + 1 = 1 ， rooms[0][3] = step 然后放进queue -> node'
然后再往下走 找到 room step +=1 = 0 + 1 = 1， rooms[1][2] = step,然后放进queue -> node''
此时对于节点为room的都走完了，

pop node' -> i = 0, j= 3, 没路可走 结束
pop node'' -> i = 1, j = 2, 可以往左走 -> step+=1 = 1+ 1 = 2, 加入queue， 
也可以往下走， i=2, j=2 step+=1 = 1+1 =2 , 加入queue
以此循环往复
"""

from collections import deque
class Solution(object):
    def wallsAndGates(self, rooms):
        mrows, ncols = len(rooms), len(rooms[0])
        gate_queue = deque()
        for i in range(mrows):
            for j in range(ncols):
                # find the gate
                if rooms[i][j] == 0:
                    gate_queue.append((i, j, 0))

        directions = [[0,1],
                      [0,-1],
                      [1,0],
                      [-1,0]]
        seen = set()
        while gate_queue:
            current = gate_queue.popleft()
            i, j, step = current[0], current[1], current[2]

            for d in directions:
                newi = i + d[0]
                newj = j + d[1]

                if mrows>newi>=0 and ncols>newj>=0 and rooms[newi][newj] == 2147483647 and (newi, newj) not in seen:
                    #found a new room
                    its_room_step = step + 1
                    rooms[newi][newj] = its_room_step
                    seen.add((newi, newj))
                    gate_queue.append((newi, newj, its_room_step))

        return rooms