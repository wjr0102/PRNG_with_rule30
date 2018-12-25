#!/usr/local/bin
# -*- coding: utf-8 -*-
# @Author: Jingrou Wu
# @Date:   2018-12-23 17:16:51
# @Last Modified by:   Jingrou Wu
# @Last Modified time: 2018-12-25 18:45:23

import numpy as np
import argparse
import matplotlib.pyplot as plt


def rule30(list_, n):
    indices = range(-n, n + 1)
    cells = [0 for i in range(2 * n + 2 + len(list_))]
    cells[0] = 0
    cells[-1] = 0
    tables = []
    new_state = {"111": 0, "110": 0, "101": 0, "000": 0,
                 "100": 1, "011": 1, "010": 1, "001": 1}
    left = (len(cells) - len(list_)) // 2
    j = 0
    for i in range(left, left + len(list_)):
        cells[i] = list_[j]
        j += 1

    for time in range(n):
        tables.append(cells[:])
        print("time:%d" % time)
        print(cells)
        cells_last = cells[:]
        for i in range(1, len(cells) - 1):
            neighbor = (str)(cells_last[i - 1]) + \
                (str)(cells_last[i]) + (str)(cells_last[i + 1])
            # print(neighbor)
            cells[i] = new_state[neighbor]
            # print(cells[i])
    tables.append(cells[:])
    print(np.array(tables))
    return np.array(tables)


def PRNG(window_size, seed, n):
    table = rule30(seed, n)
    rows = np.shape(table)[0]
    index = np.shape(table)[1] // 2
    sequence = []
    for i in range(rows - window_size):
        number = []
        for j in range(window_size):
            number.append(table[i + j, index])
        num = bi_to_digit(number)
        sequence.append(num)
    return sequence, table


def bi_to_digit(number):
    result = 0
    multi = 1
    for i in range(len(number) - 1, -1, -1):
        result += number[i] * multi
        multi = multi * 2
    return result


def is_random(seq):
    dic = {}
    for i in range(len(seq)):
        num = seq[i]
        if dic.has_key(num):
            dic[num] += 1
        else:
            dic[num] = 1
    return dic


def draw_pic(table):
    count = 1
    name = "fig_%d.png"
    rows = np.shape(table)[0]
    cols = np.shape(table)[1]
    print(rows)
    plt.ion()
    fig = plt.figure(figsize=(10, 5))
    plt.xlim(0, cols)
    plt.ylim(0, rows)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    fig.suptitle("Rule 30 when ws=%d, seed=%s, n=%d" %
                 (window_size, (str)(seed), n))
    plt.grid()
    for i in range(rows):
        for j in range(cols):
            print(i, j, table[i, j])
            if table[i, j] == 1:
                plt.fill([j, j, j + 1, j + 1], [i, i + 1, i + 1, i])
                plt.savefig("pic/" + name % count)
                # plt.pause(0.5)
                count += 1
    # while True:
    #     plt.pause(0.05)
    plt.show()


parser = argparse.ArgumentParser(
    description='Generate pseudo random sequence with Rule 30.')
parser.add_argument('seed', metavar='seed', type=int, nargs='+',
                    help='a list of 0,1 integers')
parser.add_argument('-w', '--window_size', dest='window_size', type=int, default=1,
                    help='set the window_size (default: 1)')
parser.add_argument('-n', dest='n', type=int, default=10,
                    help='set the iteration times (default: 1)')

args = parser.parse_args()
seed = args.seed
window_size = args.window_size
n = args.n
# window_size = sys.argv[1]
# seed = [1]
# n = 100
random_s, table = PRNG(window_size, seed, n)
print(random_s)
dic = is_random(random_s)
# print(dic)
# draw_pic(table)
