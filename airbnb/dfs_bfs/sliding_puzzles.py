from collections import deque


class Solution(object):
    def slidingPuzzle(self, board):
        """
        Solve 2x3 sliding puzzle using BFS.
        """

        # ------------------------------------------------
        # convert 2D board into string representation
        # this helps us store states in visited set
        # ------------------------------------------------
        start_state = "".join(str(cell) for row in board for cell in row)

        target_state = "123450"

        # ------------------------------------------------
        # adjacency mapping for each index in string
        # this represents where '0' can move
        # ------------------------------------------------
        neighbors = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4],
            4: [1, 3, 5],
            5: [2, 4]
        }

        # ------------------------------------------------
        # BFS queue: (state, steps)
        # ------------------------------------------------
        queue = deque([(start_state, 0)])

        # visited set to avoid revisiting states
        visited = set([start_state])


        while queue:

            current_state, steps = queue.popleft()

            # ------------------------------------------------
            # if we reach target, return steps
            # BFS guarantees this is minimum
            # ------------------------------------------------
            if current_state == target_state:
                return steps


            # ------------------------------------------------
            # find index of '0' (empty slot)
            # ------------------------------------------------
            zero_index = current_state.index('0')


            # ------------------------------------------------
            # try all possible swaps (moves)
            # ------------------------------------------------
            for neighbor_index in neighbors[zero_index]:

                # convert string to list for swapping
                state_list = list(current_state)

                # swap '0' with neighbor
                state_list[zero_index], state_list[neighbor_index] = \
                    state_list[neighbor_index], state_list[zero_index]

                new_state = "".join(state_list)

                # ------------------------------------------------
                # if not visited, push into queue
                # ------------------------------------------------
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, steps + 1))


        # if BFS finishes without reaching target
        return -1