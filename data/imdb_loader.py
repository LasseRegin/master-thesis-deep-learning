
import os
import csv
import json
import math
import random

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

from utils.strings import extract_words, truncate_sentences, truncate_by_word
from utils.strings import word_end_indices
from .data_loader import DataLoader
from .utils import Alphabet, Vocabulary, process_text, pad_sequence

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class IMDBLoader(DataLoader):
    def __init__(self, alphabet, filename='imbd-data.csv', **kwargs):
        """ TODO: Class description.
        """
        self.dump_filename = filename

        # Define data folder
        self.FOLDER = os.path.join(FILEPATH, 'imdb')

        # Run parent constructor
        super().__init__(**kwargs)

        # dtype used for specifying data-type in the memmap
        self.dtype = 'int16'

        # Define data row keys and indices
        last_idx = 0
        self.data_indices = {}
        for key, length in [
            ('enc_input',            self.max_enc_seq_length),     # Enc input
            ('enc_input_length',     1),                           # Input length
            ('class_count',          1),                           # Number of associated classes
            ('classes',              self.max_classes),            # Associated classes
            ('input_word_features',  self.max_features),           # Word features
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

        # import matplotlib.pyplot as plt
        # lengths = []
        # with open(os.path.join(self.FOLDER, self.dump_filename), 'r') as f:
        #     reader = csv.reader(f)
        #     for line in reader:
        #         lengths.append(len(line[0]))

        # lengths = np.asarray(lengths)
        # lengths = lengths[np.abs(lengths - lengths.mean()) < 2 * lengths.std()]
        # plt.hist(lengths)
        # plt.show()

        # Construct mappings if not already exists
        if not self.dependencies_constructed() or self.force_reconstruct:
            self._print('Constructing dependencies..')

            # Load reviews and class
            reviews, classes = [], []
            with open(os.path.join(self.FOLDER, self.dump_filename), 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    reviews.append(line[0])
                    classes.append(line[1])

            defined_categories = list(set(classes))

            # Create category to idx mapping dicts
            self.category2idx = {x: i for i, x in enumerate(defined_categories)}
            self.idx2category = {i: x for i, x in enumerate(defined_categories)}

            # Setup word count vectorizer
            vectorizer = CountVectorizer(
                stop_words='english',
                #max_df=255,                     # Due to uint8 dtype
                max_features=self.max_features,
                ngram_range=(1, 2)
            )

            # Extract question and answers + truncate them
            inputs = []
            word_documents = []
            words = set([])
            for review in reviews:
                input = process_text(review)

                # Only use the 3 first sentences of the review
                input = truncate_sentences(input, max_sentences=3)

                # Build word document
                word_documents.append(input)

                for word in extract_words(string=input):
                    words.add(word)

                # Append
                inputs.append(input)

            # Create word vocabulary
            vocabulary = Vocabulary(words=words)
            vocabulary.save(path=self.vocabulary_filename)

            # Shuffle data
            data_count = len(inputs)
            permutation = np.random.permutation(data_count)
            def shuffle(x, permutation):
                x = np.asarray(x)
                x = x[permutation]
                return x.tolist()
            inputs  = shuffle(inputs, permutation)
            classes = shuffle(classes, permutation)

            # Build word tokenizer
            vectorizer.fit(word_documents)

            # Save alphabet
            alphabet.save(path=self.char_filename)

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

            train_count = val_count = test_count = 0
            for data_idx, (input, class_idx) in enumerate(zip(inputs, classes)):

                # Build word features
                input_word_features  = vectorizer.transform([input]).toarray().squeeze().clip(0, 255).tolist()

                # Truncate to max sequence lengths (truncate by word)
                input  = truncate_by_word(input,  max_length=self.max_enc_seq_length)

                # Tokenize question and answer
                input_int  = alphabet.encode_seq(input)
                input_int,  input_length  = pad_sequence(input_int,  max_length=self.max_enc_seq_length,     alphabet=alphabet)

                # Create word separation indices
                ws_indices = word_end_indices(string=input)
                ws_indices = ws_indices[:self.max_words]
                word_count = len(ws_indices)
                ws_indices += [alphabet.PAD_ID] * (self.max_words - word_count)

                if word_count == 0:
                    continue

                classes_vec = [-1] * self.max_classes
                classes_vec[0] = class_idx
                classes_count = 1

                # Construct data row
                row  = input_int
                row += [input_length]
                row += [classes_count] # Number of classes
                row += classes_vec     # Review class
                row += input_word_features
                row += ws_indices
                row += [word_count]
                row  = np.asarray(row, dtype=self.dtype)

                # Append
                if data_idx in indices_val:
                    val_data_memmap[val_count,:] = row
                    val_count += 1
                elif data_idx in indices_test:
                    test_data_memmap[test_count,:] = row
                    test_count += 1
                else:
                    train_data_memmap[train_count,:] = row
                    train_count += 1

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
                    'category2idx': self.category2idx,
                    'class_count': len(defined_categories)
                }, f, indent=4)

            # Save vectorizer
            joblib.dump(vectorizer, self.vectorizer_filename)

            # Clean-up
            del alphabet, vectorizer
            del indices_train, indices_val, indices_test
            del word_documents

        # Load alphabet
        self.alphabet = Alphabet.load(path=self.char_filename)

        # Load vocabulary
        self.vocabulary = Vocabulary.load(path=self.vocabulary_filename)

        # Load meta data
        with open(self.meta_data_filename, 'r') as f:
            self.meta = json.load(f)
        self.num_classes = self.meta['class_count']

        # Load vectorizer
        self.vectorizer = joblib.load(self.vectorizer_filename)

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

    def __iter__(self):
        for x in self.data_memmap:
            yield x

    def dependencies_constructed(self):
        if not os.path.isfile(self.train_memmap_filename): return False
        if not os.path.isfile(self.val_memmap_filename):   return False
        if not os.path.isfile(self.test_memmap_filename):  return False
        if not os.path.isfile(self.meta_data_filename):    return False
        if not os.path.isfile(self.char_filename):         return False
        if not os.path.isfile(self.vectorizer_filename):   return False

        return True
