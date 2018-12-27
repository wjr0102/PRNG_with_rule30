#!/usr/local/bin
# -*- coding: utf-8 -*-
# @Author: Jingrou Wu
# @Date:   2018-12-27 13:58:19
# @Last Modified by:   Jingrou Wu
# @Last Modified time: 2018-12-27 14:54:01

import scipy.special as special
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('../..'))
print(sys.path)
import utils.NIST_utils as NU

# Static Datas
# Choose M according to n
n_M = {'Minimum n': [128, 6272, 750000], 'M': [8, 128, 10000]}
nM_TABLE = pd.DataFrame(n_M)
# Tabulate the frequencies vi
LRO_v = {8: [1, 2, 3, 4, None, None, None], 128: [4, 5, 6, 7, 8, 9, None],
         10000: [10, 11, 12, 13, 14, 15, 16]}
LRO_v_TABLE = pd.DataFrame(LRO_v, index=['v%d' % i for i in range(7)])

# directory recording pie
LRO_pie = {8: [0.2148, 0.3672, 0.2305, 0.1875], 128: [0.1174, 0.2430, 0.2493, 0.1752,
                                                      0.1027, 0.1124], 10000: [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]}
# directory recoding K and N
LRO_kn = {8: (3, 16), 128: (5, 49), 10000: (6, 75)}


def longestRunOfOnes(sequence):
    n = len(sequence)   # The length of the sequence
    if n < 128:     # If the sequence length is shorter than 128, we could not do the test
        print("The sequence is too short! (Minimum bits: 128)")
        return 0.0, False

    # Step 0. Accoding to the length of the sequence, choose M
    for i in range(len(nM_TABLE)):
        if n >= nM_TABLE.loc[i, 'Minimum n']:
            M = nM_TABLE.loc[i, 'M']
    K = LRO_kn[M][0]

    N = n // M  # The number of blocks
    # Initialize the frequencies counter
    dic = {}
    for i in range(len(LRO_v_TABLE[M].dropna())):
        dic[i] = 0

    for i in range(N):
        # Step 1. Divide the sequence into M-bit blocks
        seq = sequence[i * M:(i + 1) * M]
        # Step 2. Calculate the longest runs of 1 in the block
        max_adj = 0     # Record the longest runs of 1
        ones = 0        # Current runs length
        for j in range(len(seq)):   # Count the runs length
            if seq[j] == 1:
                ones += 1
            else:
                ones = 0
            if ones > max_adj:  # Get the longest runs of 1
                max_adj = ones

        # According to the table, change the max_adj
        if max_adj <= LRO_v_TABLE[M][0]:
            max_adj = LRO_v_TABLE[M][0]
        elif max_adj >= LRO_v_TABLE[M].dropna()[-1]:
            max_adj = LRO_v_TABLE[M].dropna()[-1]

        dic[(LRO_v[M].index(max_adj))] += 1  # Count this length

    N = LRO_kn[M][1]
    # Step 4. Compute chi square of obs
    obs = sum([(dic[i] - N * LRO_pie[M][i])**2 / (N * LRO_pie[M][i])
               for i in range(K + 1)])
    print(obs)
    # Step 5. Calculte p-value
    p = 1 - special.gammainc(K * 1.0 / 2, obs / 2)
    print(p)

    # Retrun the p-value and the result
    return p, p > 0.01


if __name__ == "__main__":
    # The example from the manual.
    '''Processing and Result
    vs: v0 =4;v1 =9;v2 =3;v3=0;
    chi square obs = 4.882457
    p-value = 0.180609
    '''
    example = NU.string2list(
        '11001100000101010110110001001100111000000000001001001101010100010001001111010110100000001101011111001100111001101101100010110010')
    print(longestRunOfOnes(example))
