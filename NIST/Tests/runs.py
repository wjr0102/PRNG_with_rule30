#!/usr/local/bin
# -*- coding: utf-8 -*-
# @Author: Jingrou Wu
# @Date:   2018-12-27 13:41:12
# @Last Modified by:   Jingrou Wu
# @Last Modified time: 2018-12-27 13:56:13

import math
import sys
import os
sys.path.append(os.path.abspath('../..'))
print(sys.path)
import utils.NIST_utils as NU


def runs(sequence):
    '''To determine whether the number of runs of ones and zeros of various lengths 
    is as expected for a random sequence.

    Args:
        sequence: the sequence of the numbers to test

    Returns:
        p: the p-value
        result: if p > 0.01, pass the test, otherwise, fail
    '''

    n = len(sequence)   # the length of the bit string
    t = 2 / math.sqrt(n)    # the frequency standard
    # Step 1. Calculate the ratio of 1#
    one_num = sum(sequence) * 1.0 / len(sequence)
    # Step 2. Check if the frenquency test pass
    if abs(one_num - 0.5) >= t:
        print("Fail frequency test!")
        return 0.0, False
    # Step 3. Calculate the test statistics
    obs = 1
    for i in range(len(sequence) - 1):
        if sequence[i] != sequence[i + 1]:
            obs += 1
    print(obs)
    # Step 4. Calculate the p-value
    p = math.erfc((obs - 2 * len(sequence) * one_num * (1 - one_num)) /
                  (2 * math.sqrt(2 * len(sequence)) * one_num * (1 - one_num)))
    print(p)

    # Retrun the p-value and the result
    return p, p > 0.01


if __name__ == "__main__":
    # The example from the manual.
    '''Processing and Result
    total number of runs obs = 7
    p-value = 0.147232
    '''
    example = NU.string2list('1001101011')
    print(runs(example))
    # Another example from the manul.
    '''Processing and Result
    obs = 52
    p-value = 0.500798
    '''
    example = NU.string2list(
        '1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000')
    print(runs(example))
