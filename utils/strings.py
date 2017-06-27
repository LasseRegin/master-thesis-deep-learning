
import re

_eos_reg = re.compile(r'[\.!?]+')
def truncate_sentences(string, max_sentences=3):
    """ Truncates a string on sentence level.
        Returns the first `max_sentences` sentences of `string` using [.!?] as
        sentence delimiters.
    """
    for i, match in enumerate(_eos_reg.finditer(string)):
        if i == max_sentences-1:
            string = string[0:match.start()+1]
            break
    return string


# Regular expression for words (including words like "don't" and "time-step")
_word_reg  = re.compile(r'[a-zA-Z0-9_\-\']+')

# Regular expression for stand alone symbols e.g. " - " and " _ ".
_symbol_reg = re.compile(r'\s[_\'\-]+\s')
def extract_words(string):
    """ Returns a list of strings in a list ignoring symbols like commands and
        punctuations. Also ignores specific symbols standing alone.
    """
    # Remove symbols standing alone
    string = _symbol_reg.sub(' ', string)

    return _word_reg.findall(string)


def extract_sentences(string):
    return _eos_reg.split(string)

# Regular expression for word delimitters including spaces, tabs, commas, periods
_reg_space = re.compile(r'[\s\.,!?]+')
def truncate_by_word(string, max_length):
    if len(string) <= max_length:
        return string

    to_idx = max_length
    while not _reg_space.match(string[to_idx]):
        to_idx -= 1
        if to_idx == 0:
            break
    string_truncated = string[:to_idx]

    # If the first word is longer than `max_length` we return an empty string
    # in order to prevent word truncating
    if len(string_truncated) > max_length:
        return ''

    return string_truncated


_eow_reg = re.compile(r'[\.!?\s]+')
def word_end_indices(string):
    return [match.start() for match in _eow_reg.finditer(string)]
