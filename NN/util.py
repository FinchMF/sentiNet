from collections import Counter
from typing import ( List, Dict, Any, Type, TypeVar, Union ) 
import numpy as np

def build_word_counters(labels: List[str], text: List[str]) -> Union[Dict[str, int], 
                                                                     Dict[str, int], 
                                                                     Dict[str, int]]:

    severe_counts = Counter()
    nonSevere_counts = Counter()
    total_counts = Counter()

    for i in range(len(text)):

        if labels[i] == 'severe':

            for word in text[i].split(" "):
                severe_counts[word] += 1
                total_counts[word] += 1

        else:

            for word in text[i].split(" "):
                nonSevere_counts[word] += 1
                total_counts[word] += 1

    return ( severe_counts, nonSevere_counts, total_counts )

def build_severe_nonSevere_ratio(severe_counts: Dict[str, int], 
                                 nonSevere_counts: Dict[str, int], 
                                 total_counts: Dict[str, int], count: int = 50) -> Dict[str, float]:

    severe_nonSevere_ratios = Counter()

    for word, cnt in list(total_counts.most_common()):

        if cnt >= count:

            severe_nonSevere_ratio = severe_counts[word] / float(nonSevere_counts[word] + 1)
            severe_nonSevere_ratios[word] = severe_nonSevere_ratio

        for word, ratio in severe_nonSevere_ratios.most_common():

            if ratio > 1:
                severe_nonSevere_ratios[word] = np.log(ratio) 

            else:
                severe_nonSevere_ratios[word] = -np.log((1 / (ratio + 0.01)))

    return severe_nonSevere_ratios

def build_vocab(labels: List[str], texts: List[str],
                total_counts: Dict[str, int], 
                min_count: int, polarity_cutoff: float,
                severe_nonSevere_ratios: Dict[str, float],
                reduce_noise: bool = True) -> Union[List[str], List[str]]:

    text_vocab = set()

    for text in texts:

        for word in text.split(" "):

            if reduce_noise:

                if total_counts[word] > min_count:

                    if word in severe_nonSevere_ratios.keys():

                        if ( severe_nonSevere_ratios[word] >= polarity_cutoff) or (severe_nonSevere_ratios[word] <= -polarity_cutoff):

                            text_vocab.add(word)

            else:

                text_vocab.add(word)

    label_vocab = set()

    for label in labels:

        label_vocab.add(label)

    return ( list(text_vocab), list(label_vocab))

def build_vocab_indicies(vocab: List[str]) -> Dict[str, int]:

    v2i = {}

    for i, word in enumerate(vocab):
        v2i[word] = i

    return v2i


class Data:

    def __init__(self, text: List[str], labels: List[str], 
                 polarity_cutoff: float, 
                 min_count: int, reduce_noise: bool = True):

        (self.severe_counts, 
         self.nonSevere_counts, 
         self.total_counts) = build_word_counters(labels=labels, text=text)

        self.severe_nonSevere_ratios = build_severe_nonSevere_ratio(severe_counts=self.severe_counts, 
                                                                    nonSevere_counts=self.nonSevere_counts, 
                                                                    total_counts=self.total_counts)

        self.text_vocab, self.label_vocab = build_vocab(labels=labels, texts=text, 
                                                        total_counts=self.total_counts, 
                                                        min_count=min_count, polarity_cutoff=polarity_cutoff, 
                                                        severe_nonSevere_ratios=self.severe_nonSevere_ratios)

        self.text_vocab_size = len(self.text_vocab)
        self.label_vocab_size = len(self.label_vocab)

        self.w2i = build_vocab_indicies(vocab=self.text_vocab)
        self.l2i = build_vocab_indicies(vocab=self.label_vocab)