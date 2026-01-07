"""
Number of Islands Problem using BFS (Breadth-First Search)

Given a 2D grid map of '1's (land) and '0's (water), count the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands
horizontally or vertically. You may assume all four edges of the grid are all
surrounded by water.

Example:
    grid = [
        ['1', '1', '0', '0', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '1', '0', '0'],
        ['0', '0', '0', '1', '1']
    ]
    Result: 3 islands
"""

from collections import deque
from typing import List


class IslandCounter:
    """
    A class to count islands in a 2D grid using BFS algorithm.
    
    BFS Approach:
    1. Traverse the grid cell by cell
    2. When we find a '1' (land), we've found a new island
    3. Use BFS to mark all connected land cells as visited
    4. Continue until all cells are processed
    """
    
    def __init__(self, grid: List[List[str]]):
        """
        Initialize the IslandCounter with a grid.
        
        Args:
            grid: 2D list of characters where '1' is land and '0' is water
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.visited = set()  # Track visited cells to avoid revisiting
    
    def _is_valid_cell(self, row: int, col: int) -> bool:
        """
        Check if a cell is valid (within bounds, is land, and not visited).
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if the cell is valid for BFS exploration
        """
        # Check if row is within bounds
        if row < 0 or row >= self.rows:
            return False
        
        # Check if column is within bounds
        if col < 0 or col >= self.cols:
            return False
        
        # Check if cell is land (not water)
        if self.grid[row][col] != '1':
            return False
        
        # Check if cell has already been visited
        if (row, col) in self.visited:
            return False
        
        return True
    
    def _bfs_explore_island(self, start_row: int, start_col: int) -> None:
        """
        Use BFS to explore and mark all connected land cells as visited.
        
        BFS works by:
        1. Starting from a land cell
        2. Adding it to a queue
        3. While queue is not empty:
           - Remove a cell from the queue
           - Mark it as visited
           - Add all valid neighboring land cells to the queue
        4. This continues until all connected land is explored
        
        Args:
            start_row: Starting row index of the island
            start_col: Starting column index of the island
        """
        # Directions: up, down, left, right (4-connected)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Initialize queue with starting position
        queue = deque()
        queue.append((start_row, start_col))
        
        # Mark starting cell as visited immediately
        self.visited.add((start_row, start_col))
        
        # BFS: process cells level by level
        while queue:
            # Remove the front cell from the queue
            current_row, current_col = queue.popleft()
            
            # Explore all 4 neighbors (up, down, left, right)
            for direction in directions:
                # Calculate neighbor coordinates
                delta_row, delta_col = direction
                neighbor_row = current_row + delta_row
                neighbor_col = current_col + delta_col
                
                # Check if this neighbor is valid (land, in bounds, not visited)
                if self._is_valid_cell(neighbor_row, neighbor_col):
                    # Mark as visited and add to queue for further exploration
                    self.visited.add((neighbor_row, neighbor_col))
                    queue.append((neighbor_row, neighbor_col))
    
    def count_islands(self) -> int:
        """
        Count the number of islands in the grid using BFS.
        
        Algorithm:
        1. Iterate through every cell in the grid
        2. If we find a land cell ('1') that hasn't been visited:
           - We found a new island
           - Use BFS to explore and mark all connected land
           - Increment island count
        3. Return total island count
        
        Returns:
            Number of islands in the grid
        """
        # Edge case: empty grid
        if self.rows == 0 or self.cols == 0:
            return 0
        
        island_count = 0
        
        # Traverse every cell in the grid
        for row in range(self.rows):
            for col in range(self.cols):
                # Check if this cell is land and not yet visited
                if self.grid[row][col] == '1' and (row, col) not in self.visited:
                    # Found a new island!
                    island_count += 1
                    # Use BFS to explore all connected land cells
                    self._bfs_explore_island(row, col)
        
        return island_count


def num_islands(grid: List[List[str]]) -> int:
    """
    Convenience function to count islands in a grid.
    
    Args:
        grid: 2D list of characters where '1' is land and '0' is water
        
    Returns:
        Number of islands
    """
    counter = IslandCounter(grid)
    return counter.count_islands()


# Example usage and test cases
if __name__ == "__main__":
    # Test Case 1: Example from problem description
    print("Test Case 1:")
    grid1 = [
        ['1', '1', '0', '0', '0'],
        ['1', '1', '0', '0', '0'],
        ['0', '0', '1', '0', '0'],
        ['0', '0', '0', '1', '1']
    ]
    result1 = num_islands(grid1)
    print(f"Grid: {grid1}")
    print(f"Number of islands: {result1}")
    print(f"Expected: 3\n")
    
    # Test Case 2: Single island
    print("Test Case 2:")
    grid2 = [
        ['1', '1', '1'],
        ['1', '1', '1'],
        ['1', '1', '1']
    ]
    result2 = num_islands(grid2)
    print(f"Number of islands: {result2}")
    print(f"Expected: 1\n")
    
    # Test Case 3: All water
    print("Test Case 3:")
    grid3 = [
        ['0', '0', '0'],
        ['0', '0', '0'],
        ['0', '0', '0']
    ]
    result3 = num_islands(grid3)
    print(f"Number of islands: {result3}")
    print(f"Expected: 0\n")
    
    # Test Case 4: Each cell is a separate island
    print("Test Case 4:")
    grid4 = [
        ['1', '0', '1'],
        ['0', '1', '0'],
        ['1', '0', '1']
    ]
    result4 = num_islands(grid4)
    print(f"Number of islands: {result4}")
    print(f"Expected: 5\n")
    
    # Test Case 5: Empty grid (edge case)
    print("Test Case 5:")
    grid5 = []
    result5 = num_islands(grid5)
    print(f"Number of islands: {result5}")
    print(f"Expected: 0\n")
    
    # Test Case 6: Single row
    print("Test Case 6:")
    grid6 = [['1', '0', '1', '1', '0', '1']]
    result6 = num_islands(grid6)
    print(f"Number of islands: {result6}")
    print(f"Expected: 3\n")

