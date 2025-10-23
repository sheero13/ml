# SLL-Mutex.py
# import threading

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None

# class SinglyLinkedList:
#     def __init__(self):
#         self.head = None
#         self.lock = threading.Lock()

#     def insert(self, data):
#         with self.lock:
#             new_node = Node(data)
#             new_node.next = self.head
#             self.head = new_node
#             print(f"Inserted: {data}")

#     def delete(self, data):
#         with self.lock:
#             temp = self.head
#             prev = None
#             while temp:
#                 if temp.data == data:
#                     if prev:
#                         prev.next = temp.next
#                     else:
#                         self.head = temp.next
#                     print(f"Deleted: {data}")
#                     return
#                 prev = temp
#                 temp = temp.next
#             print(f"{data} not found")

#     def display(self):
#         with self.lock:
#             temp = self.head
#             while temp:
#                 print(temp.data, end=" -> ")
#                 temp = temp.next
#             print("None")

# # --- Test ---
# if __name__ == "__main__":
#     sll = SinglyLinkedList()
#     t1 = threading.Thread(target=sll.insert, args=(10,))
#     t2 = threading.Thread(target=sll.insert, args=(20,))
#     t3 = threading.Thread(target=sll.delete, args=(10,))

#     t1.start(); t2.start(); t3.start()
#     t1.join(); t2.join(); t3.join()

#     sll.display()

#------------------------------------------------------------------------------

#sll-fine.py

# import threading

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None
#         self.lock = threading.Lock()

# class FineGrainedLinkedList:
#     def __init__(self):
#         self.head = None
#         self.head_lock = threading.Lock()

#     def insert(self, data):
#         new_node = Node(data)
#         with self.head_lock:
#             new_node.next = self.head
#             self.head = new_node
#             print(f"Inserted: {data}")

#     def delete(self, data):
#         prev = None
#         curr = self.head
#         while curr:
#             curr.lock.acquire()
#             if prev:
#                 prev.lock.release()
#             if curr.data == data:
#                 if prev:
#                     prev.next = curr.next
#                 else:
#                     with self.head_lock:
#                         self.head = curr.next
#                 print(f"Deleted: {data}")
#                 curr.lock.release()
#                 return
#             prev = curr
#             curr = curr.next
#         if prev:
#             prev.lock.release()
#         print(f"{data} not found")

#     def display(self):
#         curr = self.head
#         while curr:
#             print(curr.data, end=" -> ")
#             curr = curr.next
#         print("None")

# # --- Test ---
# if __name__ == "__main__":
#     sll = FineGrainedLinkedList()
#     threads = [
#         threading.Thread(target=sll.insert, args=(10,)),
#         threading.Thread(target=sll.insert, args=(20,)),
#         threading.Thread(target=sll.delete, args=(10,))
#     ]

#     for t in threads: t.start()
#     for t in threads: t.join()
#     sll.display()

#------------------------------------------------------------------------------------
#sll-lockfree.py
# from concurrent.futures import ThreadPoolExecutor

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None

# class LockFreeLinkedList:
#     def __init__(self):
#         self.head = None

#     def insert(self, data):
#         new_node = Node(data)
#         new_node.next = self.head
#         self.head = new_node
#         print(f"Inserted: {data}")

#     def delete(self, data):
#         prev = None
#         curr = self.head
#         while curr:
#             if curr.data == data:
#                 if prev:
#                     prev.next = curr.next
#                 else:
#                     self.head = curr.next
#                 print(f"Deleted: {data}")
#                 return
#             prev = curr
#             curr = curr.next
#         print(f"{data} not found")

#     def display(self):
#         curr = self.head
#         while curr:
#             print(curr.data, end=" -> ")
#             curr = curr.next
#         print("None")

# # --- Test ---
# if __name__ == "__main__":
#     sll = LockFreeLinkedList()
#     with ThreadPoolExecutor(max_workers=3) as executor:
#         executor.submit(sll.insert, 10)
#         executor.submit(sll.insert, 20)
#         executor.submit(sll.delete, 10)

#     sll.display()
