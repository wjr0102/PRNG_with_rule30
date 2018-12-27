#!/usr/local/bin
# -*- coding: utf-8 -*-
# @Author: Jingrou Wu
# @Date:   2018-12-27 10:48:00
# @Last Modified by:   Jingrou Wu
# @Last Modified time: 2018-12-27 11:31:07

import math
import sys
import os
sys.path.append(os.path.abspath('../..'))
import utils.NIST_utils as NU


def frequency(sequence):
    '''Frequency(Monobit) test.

    To determine if the occurancy ratio of 0/1 is close to 0.5.

    Args:
        sequence: the sequence of the numbers to test
    '''

    # Step 1. Change 0 to -1
    seq = NU.normalize(sequence)
    # Get the sum of the sequence
    s = sum(seq)
    # Step 2. Calculate the test statistic
    obs = abs(s) / (math.sqrt(len(seq)))
    # Step 3. Calculate the p-value
    p = math.erfc(obs / (math.sqrt(2)))
    # Retrun the p-value and the result
    return p, p > 0.01


if __name__ == "__main__":
    # The example from the manual.
    '''Processing and Result
    sum = 2
    test statistic obs = 0.632455532
    p-value = 0.527089
    '''
    example = NU.string2list('1011010101')
    print(frequency(example))
    # Another example from the manul.
    '''Processing and Result
    sum = -16
    obs = 1.6
    p-value = 0.109599
    '''
    example = NU.string2list(
        '1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000')
    print(frequency(example))
