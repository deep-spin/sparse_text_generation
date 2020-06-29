#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import logging

import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from collections import defaultdict

from itertools import chain


logger = logging.getLogger(__name__)

def f1(generated, reference):
    
    f=[]
    
    for idx, g in enumerate(generated):
        count=0
        for x in g:
            if x in reference[idx]:
                count+=1
        if len(g)>0:
            precision = count/len(g)
        else:
            precision=0
        recall = count/len(reference[idx])
        if recall!=0 and precision!=0:
            f.append(2*precision*recall/(precision+recall))
        else:
            f.append(0)    
    return np.array(f).mean()*100




def pad_sequence(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None):
    
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None):

    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def distinct(generated):
    distinct_1 = distinct_n_corpus_level(generated, 1)
    distinct_2 = distinct_n_corpus_level(generated, 2)
    distinct_3 = distinct_n_corpus_level(generated, 3)
    distinct_4 = distinct_n_corpus_level(generated, 4)

    return distinct_1, distinct_2, distinct_3, distinct_4

def distinct_n_sentence_level(sentence, n):

    if len(sentence) == 0:
        return 0.0 
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):

    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)