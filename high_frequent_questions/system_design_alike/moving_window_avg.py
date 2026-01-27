from collections import deque

class MovingWindowAvg(object):
    def __init__(self, size):
        """
        size: int, the sliding window size
        """

        self.size= size
        self.queue = deque()
        self.running_sum = 0

    def add(self, val):
        """
        this function adds the new value to the queue only when the size is not k yet, otherwise
        add it after deleting the stale value
        """

        self.queue.append(val)
        self.running_sum += val

        if len(self.queue) > self.size:
            removed = self.queue.popleft()
            self.running_sum -= removed

        return self.running_sum / len(self.queue)

if __name__ == '__main__':
    tracker = MovingWindowAvg(3) # set the k to be 3
    print(tracker.add(1)) # 1
    print(tracker.add(10)) # 11/ 2 = 5.5
    print(tracker.add(3))  # 11 + 3 / 3 = 4.6
    print(tracker.add(5))  # 10 + 3 + 5 = 18, 18 / 3 = 6

