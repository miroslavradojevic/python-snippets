#!/usr/bin/env python

# given two lists which are both sorted
# merge two (sorted) lists into one sorted list

a = [3, 5, 9, 1, 6, 2]
a.sort()

b = [8, 7, 4]
b.sort()

def merge_(a, b):
    c = a + b
    c.sort()
    return c
    # return sorted(c)

print(merge_(a, b))

def merge(A, B):
    M = len(A)
    N = len(B)
    result = [0] * (M+N)

    i, j, k = 0, 0, 0
    while i < M and j < N:
        if A[i] < B[j]:
            result[k] = A[i]
            i += 1
        else:
            result[k] = B[j]
            j += 1
        
        k += 1

    while i < M:
        result[k] = A[i]
        i += 1
        k += 1
    
    while j < N:
        result[k] < B[j]
        j += 1
        k += 1
    
    return result

print(merge(a, b))
