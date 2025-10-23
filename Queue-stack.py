#queue.py
# import threading
# from queue import Queue, SimpleQueue
# from concurrent.futures import ThreadPoolExecutor

# # 1. Unbounded Queue with Lock
# class UnboundedQueue:
#     def __init__(self):
#         self.queue = Queue()  # Thread-safe by default

#     def enqueue(self, item):
#         self.queue.put(item)

#     def dequeue(self):
#         if not self.queue.empty():
#             return self.queue.get()
#         return None

#     def traverse(self):
#         # Safely get a snapshot of the queue
#         with self.queue.mutex:  # internal lock of Queue
#             temp = list(self.queue.queue)
#         print("Queue:", temp)

# # 2. Lock-Free Unbounded Queue
# class LockFreeQueue:
#     def __init__(self):
#         self.queue = SimpleQueue()  # Thread-safe, lock-free

#     def enqueue(self, item):
#         self.queue.put(item)

#     def dequeue(self):
#         try:
#             return self.queue.get_nowait()
#         except:
#             return None

#     def traverse(self):
#         # Drain elements safely to display without losing them
#         temp = []
#         while True:
#             try:
#                 temp.append(self.queue.get_nowait())
#             except:
#                 break
#         print("Lock-Free Queue:", temp)
#         # Re-enqueue items to restore original state
#         for item in temp:
#             self.queue.put(item)

# # Example usage
# def add_items(q, items):
#     for item in items:
#         q.enqueue(item)

# def remove_items(q, count):
#     for _ in range(count):
#         q.dequeue()

# if __name__ == "__main__":
#     uq = UnboundedQueue()
#     lfq = LockFreeQueue()

#     # Concurrent operations
#     with ThreadPoolExecutor(max_workers=3) as executor:
#         executor.submit(add_items, uq, [1, 2, 3, 4])
#         executor.submit(remove_items, uq, 2)
#         executor.submit(add_items, uq, [5, 6])

#         executor.submit(add_items, lfq, [10, 20, 30])
#         executor.submit(remove_items, lfq, 1)
#         executor.submit(add_items, lfq, [40, 50])

#     # Results
#     uq.traverse()
#     lfq.traverse()
#--------------------------------------------------------------------------------
#stack.py
# import threading
# import time
# import random
# from concurrent.futures import ThreadPoolExecutor

# # Node class
# class Node:
#     def __init__(self, value):
#         self.value = value
#         self.next = None

# # Lock-Free Stack with Back-off (simulated using a global lock)
# class LockFreeStack:
#     def __init__(self):
#         self.head = None
#         self.lock = threading.Lock()  # simulate atomic CAS

#     # Push operation with back-off
#     def push(self, value):
#         new_node = Node(value)
#         while True:
#             with self.lock:  # simulate atomic CAS
#                 new_node.next = self.head
#                 self.head = new_node
#                 break
#             time.sleep(random.uniform(0.001, 0.01))  # back-off

#     # Pop operation with back-off
#     def pop(self):
#         while True:
#             with self.lock:
#                 if self.head is None:
#                     return None
#                 popped_value = self.head.value
#                 self.head = self.head.next
#                 return popped_value
#             time.sleep(random.uniform(0.001, 0.01))  # back-off

#     # Traverse stack
#     def traverse(self):
#         current = self.head
#         values = []
#         while current:
#             values.append(current.value)
#             current = current.next
#         print("Stack:", values)

# # Example usage
# def push_items(stack, items):
#     for item in items:
#         stack.push(item)

# def pop_items(stack, count):
#     for _ in range(count):
#         stack.pop()

# if __name__ == "__main__":
#     stack = LockFreeStack()

#     # Concurrent operations
#     with ThreadPoolExecutor(max_workers=3) as executor:
#         executor.submit(push_items, stack, [1, 2, 3])
#         executor.submit(push_items, stack, [4, 5, 6])
#         executor.submit(pop_items, stack, 2)

#     # Final stack
#     stack.traverse()
