#!/usr/bin/env python3

from cmath import nan

def isInt(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

def readInt(input):
    try:
        val = int(input)
        return True, val
    except ValueError:
        return False, nan

def binaryGap(input_int):
    # Find longest sequence of zeros in binary representation of an integer
    # A binary gap within a positive integer N is any maximal sequence of consecutive zeros that is surrounded by ones at both ends in the binary representation of N

    # convert integer into binary representation
    input_val_bin = bin(input_int).replace('0b', '')

    count_curr = 0
    count_max = 0
    for i in input_val_bin:
        if i=='0':
            count_curr+=1
        else: # '1'
            if count_curr>count_max:
                count_max=count_curr
            count_curr=0
        
        print(f"{i:5}: {count_curr:5} {count_max:5}")

    # in case gap does not need to be surrounded by both ends
    # if count_curr>count_max:
    #     count_max=count_curr
    # print(f"{'end':5}: {count_curr:5} {count_max:5}")

    return count_max

INT_LIMIT = pow(2,31)-1

if __name__=='__main__':
    while True:
        num = input("\nEnter integer number (press 'q' to quit): ")
        
        is_int, val_int = readInt(num)
        
        # check if it is integer
        if is_int:
            if 1 <= val_int <= INT_LIMIT:
                longest_binary_gap = binaryGap(val_int)
                print(f"Length of longest binary gap: {longest_binary_gap}")
            else:
                print(f"Input integer must be in range [{1},{INT_LIMIT}]")
        elif num.upper()=='Q':
            print("Exiting")
            break
        else:
            print("Entry was not integer, try again")

