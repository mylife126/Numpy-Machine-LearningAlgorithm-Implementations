"""
1. deque record the position of tail and head
for example, tail at (0, 0), head at (0,3)

2. 2D grid for the position tracking

3. up, down, left, right -> (0, 1), (0, -1), (-1, 0), (1, 0) to update the head

4. but to update the head there is constrains :
0. check if the head hits the wall already
1. if the head eats the food, points +1, then head update, tail delete, and check if head hits the tail
2. if the head eats nothing, points + 0, then head update, tail delete, and check if head hits the tail

5. use deque to maintain the snake, append to update the head, and popleft() to delete the tail from left to right as
the first in first out

tail --------> head, think how the snake evolves, it evolves from tail to head

use the set to track the tail position

6. another constraint is that the food is a 2D index, but it is shown by order, only pops 1 at a time, and
if you eat it, you cannot eat it again
"""
from collections import deque


class SnakeGame:
    def __init__(self, screen_width, screen_height, food):
        self.width = screen_width
        self.height = screen_height
        self.food = food

        # tail ------> head
        self.snake = deque()

        # init the snake
        self.snake.append((0, 0))
        self.snake_set = set()
        self.snake_set.add((0, 0))

        # init the direction map
        self.dir_map = {
            "U": (-1, 0),
            "D": (1, 0),
            "L": (0, -1),
            "R": (0, 1),
        }

        # init the eat food index
        self.food_index = 0

        # init the score
        self.score = 0

    def move(self, direction):
        """
        direction can be U, D, L, R
        """
        dx, dy = self.dir_map[direction][0], self.dir_map[direction][1]
        head_x, head_y = self.snake[-1]
        new_x = head_x + dx
        new_y = head_y + dy
        new_head = (new_x, new_y)

        # 1. check if the new_x and new_y hit the wall
        if (new_x < 0 or new_x >= self.height or new_y < 0 or new_y >= self.width):
            return -1

        # 2. now check if the head is at the food
        eat_food = False
        if self.food_index < len(self.food):
            where_is_food = self.food[self.food_index]
            if new_x == where_is_food[0] and new_y == where_is_food[1]:
                eat_food = True

        # 3. now if not eating the food
        if not eat_food:
            # remove the old tail
            tail = self.snake.popleft()
            # remove the last tail status
            self.snake_set.remove(tail)

        # 4. then check if the head hits the tail at any stage or no
        if new_head in self.snake_set:
            return -1

        # 5. now update the head
        self.snake.append(new_head)
        self.snake_set.add(new_head)

        # 6. track the score
        if eat_food:
            self.food_index += 1
            self.score += 1

        print("the current score is: ", self.score)
        return self.score


## Limiter snake game
from collections import deque
class SlidingWindowRateLimiter(object):
    def __init__(self, max_calls, window_seconds, now_fn):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.now_fn = now_fn

        # key -> deque of timestamps
        self.key2timestamps = {}

    def allow(self, key):
        now = self.now_fn()

        if key not in self.key2timestamps:
            self.key2timestamps[key] = deque()

        q = self.key2timestamps[key]
        window_start = now - self.window_seconds

        # 1) remove old timestamps
        while q and q[0] <= window_start:
            q.popleft()

        # 2) check capacity
        if len(q) >= self.max_calls:
            return False

        # 3) record this request
        q.append(now)
        return True


class SnakeGame(object):
    def __init__(self, width, height, food):
        self.width = width
        self.height = height
        self.food = food
        self.food_index = 0
        self.score = 0

        # snake: tail -> head
        self.snake = deque()
        self.snake.append((0, 0))

        self.snake_set = set()
        self.snake_set.add((0, 0))

        self.dir_map = {
            "U": (-1, 0),
            "D": (1, 0),
            "L": (0, -1),
            "R": (0, 1),
        }

    def move(self, direction):
        head_r, head_c = self.snake[-1]
        dr, dc = self.dir_map[direction]
        new_head = (head_r + dr, head_c + dc)

        # wall
        if (new_head[0] < 0 or new_head[0] >= self.height or
            new_head[1] < 0 or new_head[1] >= self.width):
            return -1

        # food
        eats_food = False
        if self.food_index < len(self.food):
            if new_head == tuple(self.food[self.food_index]):
                eats_food = True

        # remove tail first if not eating
        if not eats_food:
            tail = self.snake.popleft()
            self.snake_set.remove(tail)

        # self collision
        if new_head in self.snake_set:
            return -1

        # add head
        self.snake.append(new_head)
        self.snake_set.add(new_head)

        if eats_food:
            self.score += 1
            self.food_index += 1

        return self.score


class RateLimitedSnakeGame(object):
    def __init__(self, game_id, width, height, food, rate_limiter):
        self.game_id = game_id
        self.game = SnakeGame(width, height, food)
        self.rate_limiter = rate_limiter

    def move(self, direction):
        # rate limit first; if rejected, do NOT update game state
        if not self.rate_limiter.allow(self.game_id):
            return -2  # rate limited

        return self.game.move(direction)

if __name__ == "__main__":
    # a 3cols 2rows grid
    """
    0 1 0
    0 0 1
    """
    game = SnakeGame(3, 2, [[1,2], [0,1]])

    print(game.move("R"))  # 0
    print(game.move("D"))  # 0
    print(game.move("R"))  # 1
    print(game.move("U"))  # 1
    print(game.move("L"))  # 2
    print(game.move("U"))  # -1


# Rate limiter version
if __name__ == "__main__":
    # Fake clock so we can control time
    class FakeClock(object):
        def __init__(self):
            self.t = 0
        def now(self):
            return self.t
        def tick(self, seconds):
            self.t += seconds

    clock = FakeClock()

    # Allow at most 2 moves per 5 seconds
    limiter = SlidingWindowRateLimiter(max_calls=2, window_seconds=5, now_fn=clock.now)

    game = RateLimitedSnakeGame(
        game_id="game_1",
        width=3,
        height=2,
        food=[[1, 2], [0, 1]],
        rate_limiter=limiter
    )

    print(game.move("R"))  # t=0 allowed
    print(game.move("D"))  # t=0 allowed
    print(game.move("R"))  # t=0 rejected -> -2 (rate limited)

    clock.tick(5)          # move time forward so window clears

    print(game.move("R"))  # t=5 allowed again