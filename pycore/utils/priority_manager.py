import heapq

class PriorityManager:
    """Manages prioritized items using a min-heap."""
    def __init__(self):
        # Min-heap stores tuples: (priority, item)
        self._heap = []
        self._counter = 0 # To ensure stable ordering for items with the same priority

    def add_item(self, item, priority):
        """Adds an item with a given priority to the manager."""
        # Priority is typically lower for higher importance in a min-heap
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def get_next_item(self):
        """Retrieves and removes the item with the highest priority (lowest priority value)."""
        if not self._heap:
            return None
        # Return the item, discarding the priority and counter
        return heapq.heappop(self._heap)[2]

    def is_empty(self):
        """Checks if the priority manager is empty."""
        return not self._heap

    def size(self):
        """Returns the number of items in the priority manager."""
        return len(self._heap)

    def peek_next_item(self):
        """Retrieves the item with the highest priority without removing it."""
        if not self._heap:
            return None
        return self._heap[0][2]