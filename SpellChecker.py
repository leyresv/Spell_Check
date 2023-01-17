import re
import numpy as np
from collections import Counter

import os.path
cur_file_path = os.path.dirname(os.path.realpath(__file__))


class SpellChecker:
    """
    Spell checker class
    """

    def __init__(self, dataset, vocab=None, probs=None):
        self.vocab = vocab      # Set of words
        self.probs = probs      # Dictionary of {word:probability}
        if dataset:
            self.create_vocab(dataset)

    def create_vocab(self, dataset):
        """
        Create vocabulary of the spell-checker from the desired dataset
        :param dataset: list of texts
        """
        # Get words
        words = [w.lower() for w in re.findall(r'\w+', dataset)]
        self.vocab = set(words)

        # Get word counts
        word_counts = Counter(words)

        # Get words probabilities
        total = sum(word_counts.values())
        self.probs = {word: count / total for word, count in word_counts.items()}

    @staticmethod
    def split_word(word, end_split=False):
        """
        Get all possible 2-part splits of the word
        :param word: input string
        :param end_split: True if we want to include the end split ("word", "")
        :return: list of tuples with all the possible word splits (starting with ("", "word"))
        """
        if end_split:
            return [(word[:i], word[i:]) for i in range(len(word) + 1)]
        return [(word[:i], word[i:]) for i in range(len(word))]

    def delete_letter(self, word):
        """
        Get all the variations of a word with one letter deleted

        :param word: string to be edited
        :return: list of all possible strings where we deleted one letter of the original word
        """
        word_splits = self.split_word(word)
        return [start + end[1:] for start, end in word_splits]

    def switch_letter(self, word):
        """
        Get all the variations of a word with two consecutive letters switched

        :param word: string to be edited
        :return: list of all possible strings where we switched two consecutive letters of the original word
        """
        word_splits = self.split_word(word)
        return [start[:-1] + end[0] + start[-1:] + end[1:] for (start, end) in word_splits if len(start) > 0]

    def replace_letter(self, word):
        """
        Get all the variations of a word with one letter replaced by another

        :param word: string to be edited
        :return: list of all possible strings where we replaced one letter from the original word
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        word_splits = self.split_word(word)
        replace_l = []
        options = [start + letter + end[1:] for start, end in word_splits
                   for letter in letters
                   if start + letter + end[1:] != word]
        replace_l.extend(options)

        replace_set = set(replace_l)

        return sorted(list(replace_set))

    def insert_letter(self, word):
        """
        Get all the variations of a word with one letter inserted

        :param word: string to be edited
        :return: list of all possible strings with one new letter inserted at every offset
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        word_splits = self.split_word(word, end_split=True)
        insert_l = []
        options = [start + letter + end for start, end in word_splits
                   for letter in letters
                   if start + letter + end not in insert_l]
        insert_l.extend(options)
        return insert_l

    def edit_one_letter(self, word, allow_switches=True):
        """
        Edit one letter of a word

        :param word: string to be edited
        :param allow_switches: allow switch as edit operation
        :return: set of strings with all possible one letter edits
        """
        edit_one_set = set()
        edit_one_set.update(self.replace_letter(word))
        edit_one_set.update(self.insert_letter(word))
        edit_one_set.update(self.delete_letter(word))
        if allow_switches:
            edit_one_set.update(self.switch_letter(word))
        return edit_one_set

    def edit_k_letters(self, word, k, allow_switches=True):
        """
        Edit k letters of a word

        :param word: input word (string)
        :param k: number of letters to be edited
        :param allow_switches: allow switch as edit operation
        :return: set of strings with all possible k letters edits
        """
        edit_k_set = set()
        for i in range(k):
            if not edit_k_set:
                edit_k_set = self.edit_one_letter(word, allow_switches=allow_switches)
            else:
                edit_one_set = edit_k_set.copy()
                for item in edit_one_set:
                    edit_k_set.update(self.edit_one_letter(item, allow_switches=allow_switches))
        return edit_k_set

    def get_suggestions(self, word, n=2, k=2):
        """
        Get possible corrections for a word

        :param word: string to get suggestions for
        :param n: desired number of suggestions
        :param k: amount of edited letters per suggestion
        :return: list of tuples (word, probability) of the n most probable words
        """
        suggestions = self.vocab.intersection(self.edit_one_letter(word)) or self.vocab.intersection(self.edit_k_letters(word, k))
        sugg_prob = sorted([(word, self.probs[word]) for word in suggestions], key=lambda x: x[1], reverse=True)
        n_best = sugg_prob[:n]
        return n_best

    @staticmethod
    def min_edit_distance(source, target, ins_cost=1, del_cost=1, rep_cost=2):
        """
        Get minimum edit distance between two words

        :param source: starting string
        :param target: string that we want to end with
        :param ins_cost: integer cost of insert operation
        :param del_cost: integer cost of delete operation
        :param rep_cost: integer cost of replace operation
        :return: minimum distance (int) required to convert source to target
        """

        # use deletion and insert cost as  1
        m = len(source)
        n = len(target)

        # initialize cost matrix with zeros and dimensions (m+1,n+1)
        D = np.zeros((m + 1, n + 1), dtype=int)

        # Fill in column 0, from row 1 to row m, both inclusive
        for row in range(1, m + 1):
            D[row, 0] = row

        # Fill in row 0, for all columns from 1 to n, both inclusive
        for col in range(1, n + 1):
            D[0, col] = col

        # Loop through row 1 to row m, both inclusive
        for row in range(1, m + 1):

            # Loop through column 1 to column n, both inclusive
            for col in range(1, n + 1):

                # Initialize r_cost to the 'replace' cost that is passed into this function
                r_cost = rep_cost

                # Check to see if source character at the previous row
                # matches the target character at the previous column,
                if source[row - 1] == target[col - 1]:
                    # Update the replacement cost to 0 if source and target are the same
                    r_cost = 0

                # Update the cost at row, col based on previous entries in the cost matrix
                D[row, col] = min(D[row - 1, col] + del_cost, D[row, col - 1] + ins_cost, D[row - 1, col - 1] + r_cost)

        # Set the minimum edit distance with the cost found at row m, column n
        med = D[m, n]
        return med

    def get_vocab(self):
        return self.vocab

    def get_probs(self):
        return self.probs

