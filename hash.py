#hash.py

# class HashTable:
#     def __init__(self, size):
#         self.size = size
#         self.table = [[] for _ in range(size)]  

#     def hash_function(self, key):
#         return key % self.size

#     def insert(self, key):
#         index = self.hash_function(key)
#         if key not in self.table[index]:
#             self.table[index].append(key)

#     def search(self, key):
#         index = self.hash_function(key)
#         if key in self.table[index]:
#             return True
#         return False
    
#     def delete(self, key):
#         index = self.hash_function(key)
#         if key in self.table[index]:
#             self.table[index].remove(key)
#     def display(self):
#         for i, bucket in enumerate(self.table):
#             print(f"Bucket {i}: {bucket}")


# if __name__ == "__main__":
#     ht = HashTable(5)  # Hash table of size 5

#     # Insert keys
#     for key in [10, 20, 15, 7, 25]:
#         ht.insert(key)

#     print("Hash Table after insertion:")
#     ht.display()

#     # Search keys
#     print("\nSearching key 15:", ht.search(15))
#     print("Searching key 100:", ht.search(100))

#     # Delete a key
#     ht.delete(20)
#     print("\nHash Table after deleting key 20:")
#     ht.display()

#---------------------------------------------------------------------------

#concurrent_hashing.py

# import threading
# from concurrent.futures import ThreadPoolExecutor

# class ConcurrentHashTable:
#     def __init__(self, size):
#         self.size = size
#         self.table = [[] for _ in range(size)]  
#         self.locks = [threading.Lock() for _ in range(size)]  

#     def hash_function(self, key):
#         return key % self.size

#     def insert(self, key):
#         index = self.hash_function(key)
#         with self.locks[index]:
#             if key not in self.table[index]:
#                 self.table[index].append(key)

#     def search(self, key):
#         index = self.hash_function(key)
#         with self.locks[index]:
#             return key in self.table[index]

#     def delete(self, key):
#         index = self.hash_function(key)
#         with self.locks[index]:
#             if key in self.table[index]:
#                 self.table[index].remove(key)

#     def display(self):
#         for i, bucket in enumerate(self.table):
#             print(f"Bucket {i}: {bucket}")

# def add_keys(ht, keys):
#     for key in keys:
#         ht.insert(key)

# def remove_keys(ht, keys):
#     for key in keys:
#         ht.delete(key)

# if __name__ == "__main__":
#     cht = ConcurrentHashTable(5)

#     # Perform concurrent operations
#     with ThreadPoolExecutor(max_workers=3) as executor:
#         executor.submit(add_keys, cht, [10, 20, 15, 7, 25])
#         executor.submit(remove_keys, cht, [20, 7])
#         executor.submit(add_keys, cht, [30, 35])

#     # Display final table
#     print("Concurrent Hash Table after operations:")
#     cht.display()

#     # Search keys
#     print("\nSearch results:")
#     print("Search 15:", cht.search(15))
#     print("Search 7:", cht.search(7))

#--------------------------------------------------------------------------------------

#cuckoo.py

# class CuckooHashing:
#     def __init__(self, size=11, max_kicks=10):
#         self.size = size
#         self.table1 = [None] * size
#         self.table2 = [None] * size
#         self.max_kicks = max_kicks


#     def h1(self, key):
#         return key % self.size

#     def h2(self, key):
#         return (key // self.size) % self.size


#     def insert(self, key):
#         for _ in range(self.max_kicks):
#             # Try table1
#             if self.table1[self.h1(key)] is None:
#                 self.table1[self.h1(key)] = key
#                 return True
#             # Kick out existing key from table1
#             key, self.table1[self.h1(key)] = self.table1[self.h1(key)], key

#             # Try table2
#             if self.table2[self.h2(key)] is None:
#                 self.table2[self.h2(key)] = key
#                 return True
#             # Kick out existing key from table2
#             key, self.table2[self.h2(key)] = self.table2[self.h2(key)], key

#         print(f"Rehash needed for key {key}")
#         return False

#     def search(self, key):
#         return key == self.table1[self.h1(key)] or key == self.table2[self.h2(key)]

#     def delete(self, key):
#         if self.table1[self.h1(key)] == key:
#             self.table1[self.h1(key)] = None
#             return True
#         if self.table2[self.h2(key)] == key:
#             self.table2[self.h2(key)] = None
#             return True
#         return False

#     def display(self):
#         print("Table1:", self.table1)
#         print("Table2:", self.table2)

# if __name__ == "__main__":
#     ch = CuckooHashing(size=7)  
#     keys = [10, 20, 15, 7, 25]

#     # Insert keys
#     for key in keys:
#         ch.insert(key)

#     print("Cuckoo Hash Tables after insertion:")
#     ch.display()

#     # Search keys
#     print("\nSearch results:")
#     print("Search 15:", ch.search(15))
#     print("Search 100:", ch.search(100))

#     # Delete a key
#     ch.delete(20)
#     print("\nCuckoo Hash Tables after deleting key 20:")
#     ch.display()
