#!/usr/local/bin
# -*- coding: utf-8 -*-
# @Author: Jingrou Wu
# @Date:   2018-12-27 11:03:26
# @Last Modified by:   Jingrou Wu
# @Last Modified time: 2018-12-27 11:25:27


def normalize(sequence):
    '''Normalize the 01 sequence to {-1,1} with
    the eqution num = 2*num - 1.

    Args:
        sequence: the sequence to normalize.

    Returns:
        A sequence after normalization.
    '''

    seq = sequence[:]  # Copy the sequence

    for i in range(len(seq)):
        seq[i] = 2 * seq[i] - 1  # Change the sequence

    return seq  # Return the sequence


def string2list(sequence):
    '''Change 01 string to list of interger 01.

    Args:
        sequence: the 01 string to change.

    Returns:
        A list after change.
    '''

    seq = [] # Initialize the list

    for i in range(len(sequence)):
        seq.append((int)(sequence[i])) # Change the type of element
    
    return seq # Return the list
