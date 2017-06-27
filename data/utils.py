
import os
import re
import csv
import string
import hashlib


class Tokenizer:
    PAD_ID = 0
    GO_ID  = 1
    EOS_ID = 2
    UNK_ID = 3
    START_VOCAB = [UNK_ID]
    UNK_SYMBOL = '<unk>'

    WHITESPACE = string.whitespace

    def __init__(self, tokens):
        start_vocab_length = len(self.START_VOCAB)
        self.token2idx = {token: idx + start_vocab_length for idx, token in enumerate(sorted(tokens))}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    @classmethod
    def _load_tokens(cls, path):
        if not os.path.isfile(path):
            raise FileNotFoundError('The provided path does not exist!')

        with open(path, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f)
            tokens = [row[0] for row in reader]
        return tokens

    @classmethod
    def load(cls, path):
        return Tokenizer(tokens=cls._load_tokens(path=path))

    def save(self, path):
        with open(path, 'wt', encoding='utf-8') as f:
            writer = csv.writer(f)
            for token in sorted(self.token2idx.keys()):
                writer.writerow([token])

    def encode(self, token):
        return self.token2idx.get(token, self.UNK_ID)

    def decode(self, idx):
        return self.idx2token.get(idx, self.UNK_SYMBOL)

    def encode_seq(self, seq):
        return [self.encode(x) for x in seq]

    def decode_seq(self, seq):
        return [self.decode(x) for x in seq]

    def seq2str(self, seq, decode=True):
        if decode:
            seq = self.decode_seq(seq)

        # Convert to string
        seq_string = ''.join(seq)

        # Remove unknown symbols from end of string
        seq_string = seq_string.rstrip(self.UNK_SYMBOL)

        return seq_string

    def __len__(self):
        return len(self.token2idx) + len(self.START_VOCAB)

    def __hash__(self):
        tokens = '-'.join(list(sorted(self.token2idx.keys())))
        hash_val = str(int(hashlib.md5(tokens.encode('utf-8')).hexdigest(), 16))
        return int(hash_val[0:20])

class Alphabet(Tokenizer):
    PAD_ID = 0
    GO_ID  = 1
    EOS_ID = 2
    UNK_ID = 3
    START_VOCAB = [PAD_ID, GO_ID, EOS_ID, UNK_ID]

    UNK_SYMBOL = '_'

    def __init__(self, characters):
        super().__init__(tokens=characters)

        # Make sure there is a space in the alphabet
        assert ' ' in self.token2idx
        self.whitespace_idx = self.token2idx[' ']

    @staticmethod
    def create_standard_alphabet():
        chars  = ' '                          # Whitespace
        chars += 'abcdefghijklmnopqrstuvwxyz' # Letters
        chars += '0123456789'                 # numbers
        chars += '.,\'?:!&-"()/'              # Symbols
        alphabet = Alphabet(characters=chars)
        return alphabet

    @classmethod
    def load(cls, path):
        return Alphabet(characters=cls._load_tokens(path=path))


class Vocabulary(Tokenizer):
    PAD_ID = 0
    UNK_ID = 1
    START_VOCAB = [PAD_ID, UNK_ID]

    def __init__(self, words):
        super().__init__(tokens=words)

    @classmethod
    def load(cls, path):
        return Vocabulary(words=cls._load_tokens(path=path))


def create_standard_alphabet():
    chars = ' '                           # Whitespace
    chars += 'abcdefghijklmnopqrstuvwxyz' # Letters
    chars += '0123456789'                 # numbers
    chars += '.,\'?:!&-"()/'              # Symbols
    alphabet = Alphabet(characters=chars)
    return alphabet

apostrophe_reg       = re.compile(r'[`´]')
period_reg           = re.compile(r'[.]')
multiperiods_reg     = re.compile(r'[.]+')
multiwhitespace_reg  = re.compile(r'\s+')
leftspacedperiod_reg = re.compile(r'\s[.]+')
def process_text(line):
    # Remove tabs, multiple whitespaces and newlines:
    # http://stackoverflow.com/a/10711166/2538589
    line = ' '.join(line.split())

    # Replace fake apostrophes with correct one
    line = apostrophe_reg.sub('\'', line)

    # Replace multiple consecutive periods with a single period
    line = multiperiods_reg.sub('.', line)

    # Add whitespace on the right of a period
    line = period_reg.sub('. ', line)

    # Remove whitespace left of a period
    line = leftspacedperiod_reg.sub('.', line)

    # Remove multiple whitespaces
    line = multiwhitespace_reg.sub(' ', line)

    line = line.lower()
    line = line.strip()
    return line


def pad_sequence(sequence, max_length, alphabet):
    seq_length = max_length
    tmp_length = len(sequence)
    if tmp_length < max_length:
        sequence.append(alphabet.EOS_ID)
        tmp_length += 1

    diff = max_length - tmp_length
    sequence += [alphabet.PAD_ID] * diff
    seq_length -= diff

    return sequence, seq_length
