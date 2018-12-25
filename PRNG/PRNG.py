#!/usr/local/bin
# -*- coding: utf-8 -*-
# @Author: Jingrou Wu
# @Date:   2018-12-21 18:18:06
# @Last Modified by:   Jingrou Wu
# @Last Modified time: 2018-12-22 12:45:47

'''
3.1）一般选取方法：乘数a满足a=4p+1；增量b满足b=2q+1。其中p，q为正整数。 PS:不要问我为什么，我只是搬运工，没有深入研究过这个问题。

3.2）m值得话最好是选择大的，因为m值直接影响伪随机数序列的周期长短。记得Java中是取得32位2进制数吧。

3.3）a和b的值越大，产生的伪随机数越均匀

3.4）a和m如果互质，产生随机数效果比不互质好
'''


def pseudo1(a, b, m, seed=None):
    x_last = seed
    x_list = []
    for i in range(10):
        x = (a * x_last + b) % m
        x_list.append(x)
        print(x)
        x_last = x

    return x_list


def middle_sqaure(seed, n):
    number = seed
    already_seen = set()
    counter = 0
    left = (2 * n) // 4
    right = 3 * ((2 * n) // 4)

    while number not in already_seen:
        counter += 1
        already_seen.add(number)
        number = int(str(number * number).zfill(2 * n)[left:right])
        print("Counter: \t%d\nNumber:\t%d" % (counter, number))

    print(already_seen)

def Lehmer(seed,g):


if __name__ == "__main__":
    # p = 1
    # q = 1
    # a = 4 * p + 1
    # b = 2 * q + 1
    # m = 21
    # pseudo1(a, b, m, 0)
    middle_sqaure(7, )

    random.seed(0)
