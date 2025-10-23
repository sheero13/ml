#HUFFMAN.py

# import heapq

# class Node:
#     def __init__(self, char, freq):
#         self.char = char
#         self.freq = freq
#         self.left = None
#         self.right = None

#     def __lt__(self, other):
#         return self.freq < other.freq

# def build_huffman_tree(text):
#     freq = {}
#     for ch in text:
#         freq[ch] = freq.get(ch, 0) + 1

#     heap = [Node(ch, fr) for ch, fr in freq.items()]
#     heapq.heapify(heap)

#     while len(heap) > 1:
#         left = heapq.heappop(heap)
#         right = heapq.heappop(heap)
#         merged = Node(None, left.freq + right.freq)
#         merged.left = left
#         merged.right = right
#         heapq.heappush(heap, merged)

#     return heap[0]

# def build_codes(root, current_code, codes):
#     if root is None:
#         return
#     if root.char is not None:
#         codes[root.char] = current_code
#         return
#     build_codes(root.left, current_code + "0", codes)
#     build_codes(root.right, current_code + "1", codes)

# def huffman_encoding(text):
#     root = build_huffman_tree(text)
#     codes = {}
#     build_codes(root, "", codes)
#     encoded_text = "".join(codes[ch] for ch in text)
#     return encoded_text, codes

# def huffman_decoding(encoded_text, codes):
#     reverse_codes = {v: k for k, v in codes.items()}
#     current_code = ""
#     decoded_text = ""
#     for bit in encoded_text:
#         current_code += bit
#         if current_code in reverse_codes:
#             decoded_text += reverse_codes[current_code]
#             current_code = ""
#     return decoded_text

# # Example usage
# text = "hello huffman"
# encoded_text, codes = huffman_encoding(text)
# decoded_text = huffman_decoding(encoded_text, codes)

# print("Original Text:", text)
# print("Encoded Text:", encoded_text)
# print("Codes:", codes)
# print("Decoded Text:", decoded_text)

#----------------------------------------------------------------------------
#LCS.py
# def lcs(X, Y):
#     m, n = len(X), len(Y)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]

#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             if X[i - 1] == Y[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1] + 1
#             else:
#                 dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

#     print("\nDP Table:")
#     for row in dp:
#         print(row)

#     i, j = m, n
#     lcs_str = ""
#     while i > 0 and j > 0:
#         if X[i - 1] == Y[j - 1]:
#             lcs_str = X[i - 1] + lcs_str
#             i -= 1
#             j -= 1
#         elif dp[i - 1][j] > dp[i][j - 1]:
#             i -= 1
#         else:
#             j -= 1

#     return dp[m][n], lcs_str

# X = input("Enter first string: ")
# Y = input("Enter second string: ")

# length, sequence = lcs(X, Y)

# print("\nLength of LCS:", length)
# print("LCS:", sequence)
