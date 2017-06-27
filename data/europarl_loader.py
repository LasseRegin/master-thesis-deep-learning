
import os
import json
import math
import random
import collections

import numpy as np
from sklearn.externals import joblib

from utils.strings import extract_words, truncate_sentences, truncate_by_word
from utils.strings import word_end_indices
from .data_loader import DataLoader
from .utils import Alphabet, Vocabulary, process_text, pad_sequence

# TODO's
# * Process text before tokenizing
# * Truncate to 1 sentence
FILEPATH = os.path.dirname(os.path.abspath(__file__))

class EuroparlLoader(DataLoader):
    def __init__(
            self,
            alphabet,
            source_filename='europarl-v7.fr-en.en',
            target_filename='europarl-v7.fr-en.fr',
            **kwargs
        ):
        """ TODO: Class description.
        """
        # Define data folder
        self.FOLDER = os.path.join(FILEPATH, 'europarl')

        self.source_filename = os.path.join(self.FOLDER, source_filename)
        self.target_filename = os.path.join(self.FOLDER, target_filename)

        # Run parent constructor
        super().__init__(**kwargs)

        # dtype used for specifying data-type in the memmap
        self.dtype = 'int16'

        # Define data row keys and indices
        last_idx = 0
        self.data_indices = {}
        for key, length in [
            ('enc_input',            self.max_enc_seq_length),     # Enc input
            ('dec_input',            self.max_dec_seq_length + 1), # Dec output
            ('dec_target',           self.max_dec_seq_length + 1), # Dec target
            ('enc_input_length',     1),                           # Input length
            ('dec_input_length',     1),                           # Output length
            ('is_lm_mode',           1),                           # Binary mode value
            ('ws_indices',           self.max_words),              # Word indices
            ('word_count',           1),                           # Number of words
        ]:
            self.data_indices[key] = (last_idx, last_idx + length)
            last_idx += length

        # Number of total elements per observation
        self.row_elements = last_idx

        # Create or load data
        # Defines `self.train_data`, `self.val_data`, and `self.test_data`.
        self.setup_data(alphabet=alphabet)


    def setup_data(self, alphabet):

        # Construct mappings if not already exists
        if not self.dependencies_constructed() or self.force_reconstruct:
            self._print('Constructing dependencies..')

            # Count number of observations first
            data_count = 0
            with open(self.source_filename, 'r') as f:
                for _ in f:
                    data_count += 1

            # Split in train, validation and test
            indices = np.random.permutation(data_count)
            test_count = math.floor(self.test_fraction * data_count)
            val_count  = math.floor(self.val_fraction  * data_count)
            train_count = data_count - test_count - val_count
            indices_test  = indices[0:test_count]
            indices_val   = indices[test_count:test_count + val_count]
            indices_train = indices[test_count + val_count:]

            train_data_memmap = np.memmap(
                filename=self.train_memmap_filename,
                dtype=self.dtype,
                mode='w+',
                shape=(train_count, self.row_elements)
            )
            val_data_memmap = np.memmap(
                filename=self.val_memmap_filename,
                dtype=self.dtype,
                mode='w+',
                shape=(val_count, self.row_elements)
            )
            test_data_memmap = np.memmap(
                filename=self.test_memmap_filename,
                dtype=self.dtype,
                mode='w+',
                shape=(test_count, self.row_elements)
            )

            # Create custom alphabet
            chars = collections.Counter()
            for source, target in self.iterate_source_targets():
                input  = process_text(source)
                output = process_text(target)
                for text in [input, output]:
                    for char in text:
                        chars[char] += 1

            # Use the 100 most common characters
            alphabet = Alphabet(characters=[char for char, count in chars.most_common(100)])
            alphabet.save(path=self.char_filename)

            train_count = val_count = test_count = 0
            words = set([])
            permutation = np.random.permutation(data_count)
            for i, (source, target) in enumerate(self.iterate_source_targets()):
                input  = process_text(source)
                output = process_text(target)

                for text in [input, output]:
                    for word in extract_words(string=text):
                        words.add(word)

                # Truncate to max sequence lengths
                input  = truncate_by_word(input,  max_length=self.max_enc_seq_length)
                output = truncate_by_word(output, max_length=self.max_dec_seq_length)

                # Tokenize source and target
                input_int  = alphabet.encode_seq(input)
                output_int = [alphabet.GO_ID] + alphabet.encode_seq(output)

                input_int,  input_length  = pad_sequence(input_int,  max_length=self.max_enc_seq_length,     alphabet=alphabet)
                output_int, output_length = pad_sequence(output_int, max_length=self.max_dec_seq_length + 1, alphabet=alphabet)
                target_int                = output_int[1:] + [0]

                # Create word separation indices
                ws_indices = word_end_indices(string=input)
                ws_indices = ws_indices[:self.max_words]
                word_count = len(ws_indices)
                ws_indices += [alphabet.PAD_ID] * (self.max_words - word_count)

                if word_count == 0:
                    continue

                # Construct data row
                row  = input_int + output_int + target_int
                row += [input_length, output_length]
                row += [0] # For binary value "mode"
                row += ws_indices
                row += [word_count]
                row  = np.asarray(row, dtype=self.dtype)

                # Append
                data_idx = permutation[i]
                if data_idx in indices_val:
                    val_data_memmap[val_count,:] = row
                    val_count += 1
                elif data_idx in indices_test:
                    test_data_memmap[test_count,:] = row
                    test_count += 1
                else:
                    train_data_memmap[train_count,:] = row
                    train_count += 1

                if ((i + 1) % 10000) == 0:
                    self._print('Processed %d/%d observations' % (i+1, data_count))

            # Create word vocabulary
            vocabulary = Vocabulary(words=words)
            vocabulary.save(path=self.vocabulary_filename)

            # Close files
            train_data_memmap.flush()
            val_data_memmap.flush()
            test_data_memmap.flush()
            del train_data_memmap, val_data_memmap, test_data_memmap

            # Save meta data
            with open(self.meta_data_filename, 'w') as f:
                json.dump({
                    'dtype': self.dtype,
                    'row_elements': self.row_elements,
                    'counts': {
                        'total': data_count,
                        'train': train_count,
                        'val':   val_count,
                        'test':  test_count
                    },
                    'shapes': {
                        'train': (train_count, self.row_elements),
                        'val':   (val_count,   self.row_elements),
                        'test':  (test_count,  self.row_elements)
                    },
                }, f, indent=4)

            # Clean-up
            del alphabet
            del indices_train, indices_val, indices_test

        # Load alphabet
        self.alphabet = Alphabet.load(path=self.char_filename)

        # Load vocabulary
        self.vocabulary = Vocabulary.load(path=self.vocabulary_filename)

        # Load meta data
        with open(self.meta_data_filename, 'r') as f:
            self.meta = json.load(f)

        if self.validation == 'train':
            self.data_memmap = np.memmap(
                filename=self.train_memmap_filename,
                dtype=self.meta['dtype'],
                mode='c',
                shape=tuple(self.meta['shapes']['train'])
            )
        elif self.validation == 'val':
            self.data_memmap = np.memmap(
                filename=self.val_memmap_filename,
                dtype=self.meta['dtype'],
                mode='c',
                shape=tuple(self.meta['shapes']['val'])
            )
        elif self.validation == 'test':
            self.data_memmap = np.memmap(
                filename=self.test_memmap_filename,
                dtype=self.meta['dtype'],
                mode='c',
                shape=tuple(self.meta['shapes']['test'])
            )
        else:
            raise KeyError('Invalid validation argument provided')


    def iterate_source_targets(self):
        with open(self.source_filename, 'r') as f_source, open(self.target_filename, 'r') as f_target:
            for source, target in zip(f_source, f_target):
                yield source, target


    def __iter__(self):
        for x in self.data_memmap:
            yield x


    def dependencies_constructed(self):
        if not os.path.isfile(self.train_memmap_filename): return False
        if not os.path.isfile(self.val_memmap_filename):   return False
        if not os.path.isfile(self.test_memmap_filename):  return False
        if not os.path.isfile(self.meta_data_filename):    return False
        if not os.path.isfile(self.char_filename):         return False

        return True

