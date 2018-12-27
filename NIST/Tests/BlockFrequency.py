#!/usr/local/bin
# -*- coding: utf-8 -*-
# @Author: Jingrou Wu
# @Date:   2018-12-27 13:11:19
# @Last Modified by:   Jingrou Wu
# @Last Modified time: 2018-12-27 13:26:11

import scipy.special as special
import sys
import os
sys.path.append(os.path.abspath('../..'))
print(sys.path)
import utils.NIST_utils as NU


def blockFrequency(M, sequence):
    '''Do Frequency test within a Block.

    Args:
        M:  the length of each block.
        sequence:  the sequence to test.

    Returns:
        p: the p-value
        result: if p > 0.01, pass the test, otherwise, fail
    '''
    n = len(sequence)   # the length of the bit string
    N = n // M          # The number of blocks
    ratios = []         # The list contains every 1's ration in every block

    for i in range(N):
        # Step 1. Divide the sequence
        seq = sequence[i * M:(i + 1) * M]
        # Step 2. Calculate the ratio of 1
        ratio = sum(seq) * 1.0 / len(seq)
        ratios.append(ratio)    # Add the ration into the list

    # Step 3. Calculate the statistics
    obs = 4 * M * sum((ratio - 0.5)**2 for ratio in ratios)

    # Step 4. Calculate the p-value
    p = 1 - special.gammainc(N * 1.0 / 2, obs / 2)

    # Retrun the p-value and the result
    return p, p > 0.01


if __name__ == "__main__":
    # The example from the manual.
    '''Processing and Result
    chi squre of test statistic obs = 1
    p-value = 0.801252
    '''
    example = NU.string2list('0110011010')
    M = 3
    print(blockFrequency(M, example))
    # Another example from the manul.
    '''Processing and Result
    obs = 7.2
    p-value = 0.706438
    '''
    example = NU.string2list(
        '1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000')
    M = 10
    print(blockFrequency(M, example))
