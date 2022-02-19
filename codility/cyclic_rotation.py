#!/usr/bin/env python3 

def isInt(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

def rotateRight(A):
    N = len(A)
    temp_val = A[N-1]
    for i in range(N-1, 0, -1):
        A[i] = A[i-1]
    A[0] = temp_val

def cyclicRotation(A, K):
    # Given an array A consisting of N integers [−1000..1000] and an integer K
    # returns the array A rotated K times
    # The goal is to rotate array K times
    # Each element of A will be shifted to the right K times
    # N and K are integers within the range [0..100]

    print(f"\nA={A}\nK={K}")

    # check input
    K_valid = isInt(K) and 0 <= K <= 100
    
    if not K_valid:
        print(f"Invalid rotation count: {K}")
        return

    A_valid = all(isinstance(a_, int) for a_ in A) and min(A) >= -1000 and max(A) <= 1000
    
    if not A_valid:
        print("Invalid array values")
        return

    for i in range(K):
        rotateRight(A)
        print(f"{A}")

if __name__=='__main__':
    
    A = [3, 8, 9, 7, 6]
    K = 3
    cyclicRotation(A, K)

    A = [0, 0, 0]
    K = 1
    cyclicRotation(A, K)

    A = [1, 2, 3, 4]
    K = 4
    cyclicRotation(A, K)