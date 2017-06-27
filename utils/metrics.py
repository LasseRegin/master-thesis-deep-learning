
import os
from math import exp, log
from collections import Counter
from subprocess import Popen, PIPE
import subprocess

def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i           # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j           # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, i, j, s1, s2, substitution_cost=1, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    lev[i][j] = min(a, b, c, d)


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    """
    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.
    Allows specifying the cost of substitution edits (e.g., "a" -> "b"),
    because sometimes it makes sense to assign greater penalties to substitutions.
    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.
    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :type transpositions: bool
    :rtype int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(lev, i + 1, j + 1, s1, s2,
                            substitution_cost=substitution_cost, transpositions=transpositions)
    return lev[len1][len2]


def compute_moses_bleu(candidates, references):
    """ Computes the moses BLEU score from a list of candidates and a list
        of references
    """
    folder_path = os.path.dirname(os.path.realpath(__file__))
    candidates_filename = os.path.join(folder_path, 'tmp_candidates.txt')
    references_filename = os.path.join(folder_path, 'tmp_references.txt')

    for filename, values in [
        (candidates_filename, candidates),
        (references_filename, references)
    ]:
        with open(filename, 'w') as f:
            f.write('\n'.join(values))

    # Compute BLEU score
    bleu_score = moses_bleu(candidates_filename, references_filename)

    # Delete files
    os.remove(candidates_filename)
    os.remove(references_filename)

    return bleu_score


def moses_bleu(translated, reference):
    """Using moses bleu implementation multi-bleu.perl"""
    script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'multi-bleu.perl')
    call = ["%s %s %s %s %s" % ("perl %s" % (script_path), reference, " < ", translated, "| awk '{ print $3 }' | sed 's/,//'")]
    call1 = 'perl %s %s < %s ' % (script_path, reference, translated)

    subprocess.call(call1, shell=True)
    p = Popen(call, stdout=PIPE, shell=True)
    out = p.communicate()[0]
    return float(out)


def corpus_bleu(candidates, references, max_n=4):
    """Corpus bleu supporting a single reference per candidate."""
    p_ns = []
    # compute modified n-gram precision for each n:
    for n in range(1, max_n+1):
        count_clip = 0
        count = 0
        for candidate, reference in zip(candidates, references):
            if len(reference) < n or len(candidate) < n:
                continue

            reference_ngrams = ngrams(reference, n)
            reference_counts = Counter(reference_ngrams)

            candidate_ngrams = list(ngrams(candidate, n))
            candidate_counts = Counter(candidate_ngrams)

            for gram, cnt in candidate_counts.items():
                if gram in reference_counts:
                    count_clip += min(cnt, reference_counts[gram])

            count += len(candidate_ngrams)

        # avoid returning p_n = 0 because log(0) is undefined
        if count_clip == 0:
            if n == 1:
                return 0
            else:
                count_clip = 1

        if count:
            p_ns.append(count_clip/count)

    score = exp(sum([1/len(p_ns) * log(p_n) for p_n in p_ns]))

    # compute brevity penalty (BP)
    c = r = 0
    for candidate, reference in zip(candidates, references):
        c += min(len(candidate), len(reference))
        r += len(reference)
    brevity_penalty = exp(1-r/c)

    return brevity_penalty * score


def sentence_bleu(candidate, reference, weights=4):
    """Sentence bleu supporting a single reference."""
    p_ns = []
    # compute modified n-gram precision for each n:
    for n in range(1, weights+1):
        if len(reference) < n:
            continue
        if len(candidate) < n:
            p_ns.append(1)
            continue

        reference_ngrams = ngrams(reference, n)
        reference_counts = Counter(reference_ngrams)

        candidate_ngrams = list(ngrams(candidate, n))
        candidate_counts = Counter(candidate_ngrams)

        hits = 0
        for gram, count in candidate_counts.items():
            if gram in reference_counts:
                hits += min(count, reference_counts[gram])

        # avoid returning p_n = 0 because log(0) is undefined
        if hits == 0:
            if n == 1:
                return 0
            else:
                hits = 1

        p_ns.append(hits/len(candidate_ngrams))

    score = exp(sum([1/len(p_ns) * log(p_n) for p_n in p_ns]))

    # compute brevity penalty (BP)
    if len(candidate) > len(reference):
        brevity_penalty = 1
    else:
        brevity_penalty = exp(1-len(reference)/len(candidate))

    return brevity_penalty * score


def mean_char_edit_distance(candidates, references):
    total_distance = 0
    total_target_length = 0
    for y, t in zip(candidates, references):
        total_distance += edit_distance(y, t)
        total_target_length += len(t)
    return total_distance/total_target_length



def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:

        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
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


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    """
    Returns a padded sequence of items before ngram extraction.

        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']

    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n-1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n-1))
    return sequence



if __name__ == '__main__':
    candidate = ['this', 'is', 'a', 'test']
    reference = ['here', 'is', 'a', 'test']
    candidate2 = ['this', 'is', 'another', 'test']
    reference2 = ['this', 'is', 'another', 'test']
    print(corpus_bleu([candidate, candidate2], [reference, reference2]))
    print(sentence_bleu(candidate, reference))
    print(sentence_bleu(reference, reference))
    print(edit_distance(' '.join(candidate), ' '.join(reference)))
    print(edit_distance(' '.join(candidate2), ' '.join(reference2)))
    print(edit_distance('This is a test', 'this is far from the real one'))
    print(edit_distance('boom', 'this is far from the real one'))

    ref_filename, cand_filename = 'ref_file.txt', 'cand_filename.txt'
    with open(ref_filename, 'w') as f_ref, open(cand_filename, 'w') as f_cand:
        f_ref.write('\n'.join([
            'this is a test',
            'this is also a test',
            'this is the last test'
        ]))
        f_cand.write('\n'.join([
            'here is a test',
            'this is another test',
            'this is the last test'
        ]))

    print(moses_bleu(cand_filename, ref_filename))
    os.remove(ref_filename)
    os.remove(cand_filename)
