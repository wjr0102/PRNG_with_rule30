#!/usr/local/bin
# -*- coding: utf-8 -*-
# @Author: Jingrou Wu
# @Date:   2018-12-24 19:15:04
# @Last Modified by:   Jingrou Wu
# @Last Modified time: 2018-12-25 19:21:15

import math
import scipy.special as special
import pandas as pd
import numpy as np
from itertools import permutations


def frequency(sequence):
    seq = sequence[:]

    # step 1. Change 0 to -1
    for i in range(len(seq)):
        if seq[i] == 0:
            seq[i] = -1
    # Get the sum of the sequence
    s = sum(seq)
    # Step 2. Calculate the statistics
    obs = abs(s) / (math.sqrt(len(seq)))
    # Step 3. Calculate the p-value
    p = math.erfc(obs / (math.sqrt(2)))

    if p > 0.01:
        return p,True
    else:
        return p,False


def blockFrequency(M, n, sequence):
    N = n // M
    ratios = []
    for i in range(N):
        # Step 1. Divide the sequence
        seq = sequence[i * M:(i + 1) * M]
        # Step 2. Calculate the ratio of 1
        count = 0
        for j in range(len(seq)):
            if seq[j] == 1:
                count += 1
        ratio = count * 1.0 / len(seq)
        ratios.append(ratio)

    # Step 3. Calculate the statistics
    for i in range(len(ratios)):
        ratios[i] = (ratios[i] - 0.5)**2
    obs = 4 * M * sum(ratios)

    # Step 4. Calculate the p-value
    p = 1 - special.gammainc(1.5, 0.5)
    # print(p)
    if p > 0.01:
        return p,True
    else:
        return p,False


def runs(sequence):
    t = 2 / math.sqrt(len(sequence))
    # Step 1. Calculate the ratio of 1#
    one_num = sum(sequence) * 1.0 / len(sequence)
    # Step 2. Check if the frenquency test pass
    if abs(one_num - 0.5) >= t:
        return False
    # Step 3. Calculate the statistics
    obs = 1
    for i in range(len(sequence) - 1):
        if sequence[i] != sequence[i + 1]:
            obs += 1
    print(obs)
    # Step 4. Calculate the p-value
    p = math.erfc((obs - 2 * len(sequence) * one_num * (1 - one_num)) /
                  (2 * math.sqrt(2 * len(sequence)) * one_num * (1 - one_num)))
    print(p)

    if p > 0.01:
        return p,True
    else:
        return p,False


def longestRunOfOnes(sequence):
    n = len(sequence)
    # Step 0. Accoding to the length of the sequence, choose M
    if n < 128:
        print("Too few bits!")
        return False
    elif n < 6272:
        M = 8
    elif n < 750000:
        M = 128
    else:
        M = 1000

    N = n // M
    dic = {}
    for i in range(len(LRO_v[M])):
        dic[i] = 0
    for i in range(N):
        # Step 2. Divide the sequence
        seq = sequence[i * M:(i + 1) * M]
        print(seq)
        # Step 3. Calculate the frencuency of 1 and adj. 1
        max_adj = 0
        ones = 0
        for j in range(len(seq)):
            if seq[j] == 1:
                ones += 1
            else:
                ones = 0
            if ones > max_adj:
                max_adj = ones
        if max_adj <= LRO_v[M][0]:
            max_adj = LRO_v[M][0]
        elif max_adj >= LRO_v[M][-1]:
            max_adj = LRO_v[M][-1]
        dic[(LRO_v[M].index(max_adj))] += 1
        print(max_adj)
    obs = 0
    for i in range(len(LRO_v[M])):
        obs += ((dic[LRO_v[M].index(LRO_v[M][i])] - LRO_N[M] *
                 LRO_pie[M][i])**2) / (LRO_N[M] * LRO_pie[M][i])
    print(obs)
    # Step . Calculte p-value
    p = 1 - special.gammainc(LRO_k[M] * 1.0 / 2, obs / 2)
    print(p)

    if p > 0.01:
        return p,True
    else:
        return p,False


def ranking(sequence, M, Q):
    # Step 1. Divide the sequence into N matrix(M*Q)
    N = (len(sequence)) // (M * Q)
    matrixs = []
    for i in range(N):
        matrix = []
        for j in range(Q):
            row = sequence[i * M * Q + j * Q:i * M * Q + j * Q + Q]
            print(row)
            matrix.append(row)
        matrixs.append(matrix)
    for i in range(len(matrixs)):
        matrixs[i] = np.matrix(matrixs[i])
    full_rank = np.shape(matrixs[0])[0]
    ranks = {full_rank: 0, full_rank - 1: 0, 0: 0}
    for matrix in matrixs:
        rank = np.linalg.matrix_rank(matrix)
        if rank < (full_rank - 1):
            rank = 0
        ranks[rank] += 1
    print(ranks)
    obs = ((ranks[full_rank] - 0.2888 * N)**2) / (0.2888 * N) + (ranks[full_rank - 1] - 0.5776 * N)**2 / \
        (0.5776 * N) + (ranks[0] - 0.1336 * N)**2 / (0.1336 * N)
    print(obs)

    # Step . Calculate p-value
    p = math.e**(-obs / 2)
    print(p)
    if p > 0.01:
        return p,True
    else:
        return p,False

# Maybe something wrong?
def discreteFourierTransform(sequence):
    seq = sequence[:]
    n = len(seq)
    # Step 1. Change the sequence
    for i in range(len(seq)):
        seq[i] = 2 * seq[i] - 1
    print(seq)
    dft_matrix = np.fft.fft(seq)
    print(dft_matrix)
    m = abs(dft_matrix)[:n/2]
    print(m)
    T = math.sqrt(3*n)
    print(T)
    n0 = 0.95*n/2
    n1 = 0
    for i in range(len(m)):
        if m[i]<T:
            n1 += 1
    print("n1=%s"%n1)
    print("n0=%s"%n0)
    d = (n1-n0)/(math.sqrt(n*0.95*0.05/2))
    p = math.erfc(abs(d)/math.sqrt(2))
    print(p)
    if p > 0.01:
        return p,True
    else:
        return p,False


def nonOverLappingTemplateMatching(M, sequence, B):
    n = len(sequence)
    N = n // M
    m = len(B)
    ws = []
    for i in range(N):
        seq = sequence[i * M:i * M + M]
        w = len(KMP(seq, B))
        ws.append(w)
    miu = (M - m + 1)*1.0 / math.pow(2, m)
    sigima2 = M * (math.pow(2, -m) - (2 * m - 1) * math.pow(2, -2 * m))
    obs = 0
    for w in ws:
        print(w)
        obs += (w - miu)**2 / sigima2

    p = 1 - special.gammainc(N * 1.0 / 2, obs / 2)
    print(p)
    if p > 0.01:
        return p,True
    else:
        return p,False

# Example result different from the guidance
def overLappingTemplateMatching(M,sequence,B):
    n = len(sequence)
    N = n//M
    m = len(B)
    k = 5
    blocks = {}
    for i in range(k+1):
        blocks[i] = 0

    for i in range(N):
        seq = sequence[i*M:i*M+M]
        w = len(KMP(seq,B,True))
        print(w)
        if w > k:
            w = k
        blocks[w] += 1
    print(blocks)
    print(N)
    lb = (M-m+1)/math.pow(2,m)
    obs = 0
    for i in range(k+1):
        print(blocks[i],OLTM_pie[i],N)
        obs += (blocks[i]-N*OLTM_pie[i])**2/(N*OLTM_pie[i])
        print((blocks[i]-N*OLTM_pie[i])**2/(N*OLTM_pie[i]))
        print(obs)
    print(obs)
    p = 1 - special.gammainc(N * 1.0 / 2, obs / 2)
    print(p)
    if p > 0.01:
        return p,True
    else:
        return p,False

def Universal(M,Q,sequence):
    n = len(sequence)
    k = n // M - Q # blocks#
    initial = sequence[:Q]
    test = sequence[Q:]

    values = list(permutations(initial,M))
    initial_blocks = {}
    cumsum = 0
    for value in values:
        initial_blocks[value] = 0
    for i in range(Q):
        block_i = tuple(initial[i*M:i*M+M])
        initial_blocks[block_i] += 1
    for i in range(k):
        block_t = tuple(test[i*M:i*M+M])
        cumsum += math.log(i+Q+1-initial_blocks[block_t],2)
    print(cumsum)
    f = cumsum/k
    #p = math.erfc(abs(f-???))

# Remain to finish
def linearComplexity(M,sequence):
    n = len(sequence)
    pass

# Remain to finish
def serial(m,sequence):
    pass

def approximateEntropy(m,sequence):
    pass

def cumulativeSums(mode,sequence):
    pass

def randomExcursions(sequence):
    pass

def randomExcursionsVariant(sequence):
    pass





def KMP(text, pattern,overlapping=False):
    print("Text = \t%s\nPattern = \t%s"%((str)(text),(str)(pattern)))
    next_a = find_next(pattern)
    j = 0
    i = 0
    indices = []
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        else:
            if j == 0:
                i += 1
            else:
                i += j - next_a[j - 1] - 1
                j = next_a[j - 1]
        if j == len(pattern):
            indices.append(i - len(pattern))
            if overlapping:
                i = i - j + 1
            j = 0

    return indices


def find_next(s):
    next_a = [0]
    k = 0
    for j in range(1, len(s)):
        if s[j] == s[k]:
            k += 1
            next_a.append(k)
        else:
            k = 0
            next_a.append(k)
    return next_a


LRO_v = {8: [1, 2, 3, 4], 128: [4, 5, 6, 7, 8, 9],
         10000: [10, 11, 12, 13, 14, 15, 16]}
LRO_pie = {8: [0.2148, 0.3672, 0.2305, 0.1875], 128: [0.1174, 0.2430, 0.2493, 0.1752,
                                                      0.1027, 0.1124], 10000: [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]}
LRO_k = {8: 3, 128: 5, 10000: 6}
LRO_N = {8: 16, 128: 49, 10000: 75}
OLTM_pie = [0.324651,0.182617,0.142670,0.106645,0.077142,0.166269]

if __name__ == "__main__":
    dic = {}
    sequence = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1,
                1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    # seq = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1,1,0,1,1,0,0,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,0]
    seq = [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    seq = [1,0,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,1,1,0]
    seq = [1,0,0,1,0,1,0,0,1,1]
    seq = [1,0,1,1,1,0,1,1,1,1,0,0,1,0,1,1,0,1,0,0,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1,1,1,0,0,0,0,1,0,1,1,0,1,1,0,1]
    seq = '10111011110010110110011100101110111110000101101001'
    seq = '01011010011101010111'
    # seq = '1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000'
    seq = list(seq)
    for i in range(len(seq)):
        seq[i] = (int)(seq[i])
    # seq.pop(48)
    # for i in range(12*8,13*8):
    #     seq.pop(i)
    # seq[12*8:13*8] = [1,1,0,1,0,1,1,0]
    # print(frequency(sequence))
    # n = len(sequence)
    # M = 11
    # print(blockFrequency(M, n, sequence))
    # print(runs(sequence))
    # print(longestRunOfOnes(sequence))
    # print(ranking(sequence, 3, 3))
    # print(nonOverLappingTemplateMatching(10, sequence, [0, 0, 1]))
    # print(discreteFourierTransform(sequence))
    print(overLappingTemplateMatching(10,sequence,[1,1]))



