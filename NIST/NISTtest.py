#!/usr/local/bin
# -*- coding: utf-8 -*-
# @Author: Jingrou Wu
# @Date:   2018-12-24 19:15:04
# @Last Modified by:   Jingrou Wu
# @Last Modified time: 2018-12-26 14:36:41

import math
import scipy.special as special
from scipy.stats import norm
import pandas as pd
import numpy as np
from itertools import permutations
from itertools import product


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

# Result wrong, need to be fixed
def serial(m,sequence):
    n = len(sequence)
    dics = []
    for i in range(3):
        seq = extend_seq(m-i,sequence)
        dic = get_frequency(m-i,seq)
        dics.append(dic)
    fi = []
    for i in range(len(dics)):
        fi_v = (math.pow(2,m-i)*1.0/n)*sum([(dics[i][key])**2 for key in dics[i].keys()]) - n
        fi.append(fi_v)
    p_values = []
    delta1 = fi[0] - fi[1]
    delta2 = fi[0] - 2*fi[1] + fi[2]
    deltas = [delta1,delta2]
    print(deltas)
    print(fi)
    for i in range(len(fi)-1):
        delta = deltas[i]
        print(math.pow(2,m-2-i),delta*1.0/2)
        p = 1- special.gammainc(math.pow(2,m-2-i),delta*1.0)
        p_values.append(p)
    return p_values

def approximateEntropy(m,sequence):
    n = len(sequence)
    fi1 = get_fi(m,sequence)
    fi2 = get_fi(m+1,sequence)
    print(fi1-fi2)
    print(math.log(2)-(fi1-fi2))
    obs = 2*n*(math.log(2)-(fi1-fi2))
    print(obs)
    p = 1 - special.gammainc(math.pow(2,m-1),obs/2)
    return p,p>=0.01

def extend_seq(m,sequence):
    seq = sequence[:]
    seq.extend(seq[:m-1])
    return seq

def get_frequency(m,sequence):
    n = len(sequence)
    values = list(product([0,1],repeat = m))
    dic = {}
    fi = 0
    for value in values:
        dic[value] = 0
    for i in range(n-m+1):
        block = tuple(sequence[i:i+m])
        dic[block] += 1
    return dic
# Error in the guidance?
# When repeat: φ(4) =0+0+0+0.1(log0.01)+0.1(log0.01)+0.2(log0.02)+0.1(log0.01)+0+0+ 0.1(log 0.01) + 0.3(log 0.03) + 0 + 0 + 0.1(log 0.01) + 0 + 0) = -1.83437197.
# 0.0x should be 0.x
def get_fi(m,sequence):
    seq = extend_seq(m,sequence)
    dic = get_frequency(m,sequence)
    for key in dic.keys():
        c = dic[key]*1.0/(n-m+1)
        if c != 0:
            fi += c*math.log(c)
    return fi


# k's range issue
def cumulativeSums(mode,sequence):
    n = len(sequence)
    seq = normalize(sequence)
    if mode == 0:
        p_sum = [sum(seq[:i+1]) for i in range(n)]
    elif mode == 1:
        p_sum = [sum(seq[i:]) for i in range(n-1,-1,-1)]
    else:
        print('Mode error: mode should only be equal to 0 or 1!')
        return
    z = max(p_sum)
    print(z)
    p1 = 0
    p2 = 0
    for k in range((int)((-n*1.0/z +1)/4),(int)(((n*1.0/z - 1)/4))+1):
        p1 += norm.cdf((4*k+1)*z/math.sqrt(n)) - norm.cdf((4*k-1)*z/math.sqrt(n))
    for k in range((int)((-n*1.0/z -3)/4),(int)(((n*1.0/z -1)/4))+1):
        p2 += norm.cdf((4*k+3)*z/math.sqrt(n)) - norm.cdf((4*k+1)*z/math.sqrt(n))
    p = 1 - p1 + p2

    return p,p>=0.01

def randomExcursions(sequence,k=5):
    seq = normalize(sequence)
    n = len(seq)
    p_sum = [sum(seq[:i+1]) for i in range(n)]
    p_sum.append(0)
    p_sum.insert(0,0)
    index = 0
    values = range(-4,5)
    values.remove(0)
    cycles = []
    while index != -1:
        dic = {}
        for value in values:
            dic[value] = 0
        if 0 not in p_sum[index+1:]:
            break
        index2 = p_sum[index+1:].index(0) + index + 1
        cycle = p_sum[index+1:index2]
        for x in cycle:
            if x in values:
                dic[x] += 1
        cycles.append(dic)
        index = index2
    j = len(cycles)
    dic = {}
    for i in range(len(cycles)):
        dic[i] = []
    table = pd.DataFrame(np.zeros((len(values),k+1)),index = values)
    for x in values:
        for cycle in cycles:
            if cycle[x]>k:
                cycle[x] = k
            table.loc[x,cycle[x]] += 1
    table['test statistic'] = 0
    table['p-value'] = 0
    for x in values:
        obs = 0
        for i in range(k+1):
            obs += (table.loc[x,i] - j*RE_PIE.loc['x = %d'%(abs(x)),'pi_%d(x)'%i])**2/(j*RE_PIE.loc['x = %d'%(abs(x)),'pi_%d(x)'%i])
        table.loc[x,'test statistic'] = obs
        p = 1 - special.gammainc(k/2,obs/2)
        table.loc[x,'p-value'] = p
    return table,table[table['p-value']<0.01]
    

def randomExcursionsVariant(sequence):
    seq = normalize(sequence)
    n = len(seq)
    p_sum = [sum(seq[:i+1]) for i in range(n)]
    p_sum.append(0)
    p_sum.insert(0,0)
    dic = {}
    values = range(-9,10)
    values.remove(0)
    j = -1
    for value in values:
        dic[value] = 0
    for ele in p_sum:
        if dic.has_key(ele):
            dic[ele] += 1
        if ele == 0:
            j += 1
    p_values = {}
    for x in values:
        p = math.erfc(abs(dic[x]-j)/math.sqrt(2*j*(4*abs(x)-2)))
        p_values[x] = p
    fail = 0
    for p in p_values.values():
        if p < 0.01:
            fail+=1
    return p_values,fail*1.0/len(values)

def normalize(sequence):
    seq = sequence[:]
    for i in range(len(seq)):
        seq[i] = 2*seq[i]-1
    return seq





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

col_name = 'pi_%d(x)'
RE_PIE = {col_name%0:[0.5,0.75,0.8333,0.8750,0.9,0.9167,0.9286],col_name%1:[0.25,0.0625,0.0278,0.0156,0.0100,0.0069,0.0051],col_name%2:[0.125,0.0469,0.0231,0.0137,0.009,0.0064,0.0047],col_name%3:[0.0625,0.0352,0.0193,0.0120,0.0081,0.0058,0.0044],col_name%4:[0.0312,0.00264,0.0161,0.0105,0.0073,0.0053,0.0041],col_name%5:[0.0312,0.0791,0.0804,0.0733,0.0656,0.0588,0.0531]}
RE_PIE = pd.DataFrame(RE_PIE,index = ['x = %d'%i for i in range(1,8)])
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
    seq = '0110110101'
    seq = '1011010111'
    seq = '0100110101'
    seq = '0011011101'
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
    # print(overLappingTemplateMatching(10,sequence,[1,1]))
    # print(randomExcursionsVariant(sequence))
    # print(randomExcursions(sequence))
    # print(RE_PIE)
    # print(seq)
    # print(cumulativeSums(0,sequence))
    # print(approximateEntropy(2,sequence))
    print(serial(3,sequence))

