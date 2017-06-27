
import os
import random

import numpy as np

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class DataLoader:
    def __init__(self, validation, val_fraction=0.1, test_fraction=0.2,
                 max_enc_seq_length=100, max_dec_seq_length=100, random_seed=42,
                 force_reconstruct=False, max_features=512, classes_only=False,
                 verbose=False):
        """ TODO: Class description.
        """
        assert validation in ['train', 'val', 'test']
        self.validation         = validation
        self.max_enc_seq_length = max_enc_seq_length
        self.max_dec_seq_length = max_dec_seq_length
        self.random_seed        = random_seed
        self.val_fraction       = val_fraction
        self.test_fraction      = test_fraction
        self.force_reconstruct  = force_reconstruct
        self.max_classes        = 5  # Maximum number of classes per question
        self.classes_only       = classes_only
        self.verbose            = verbose
        self.max_words          = self.max_enc_seq_length // 4

        # Set random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Max number of word features
        self.max_features = max_features

        # Define data identifier string from arguments and type of data loader
        self.name = type(self).__name__
        self.data_id = '-'.join([
           '%s'   % (self.name),
           '%d'   % (self.max_enc_seq_length),
           '%d'   % (self.max_dec_seq_length),
           '%d'   % (self.random_seed),
           '%.2f' % (self.val_fraction),
           '%.2f' % (self.test_fraction),
           '%d'   % (int(self.classes_only)),
        ])

        # Define data filenames and create folders
        self.setup_filenames()

    def setup_data(self):
        raise NotImplementedError()


    def dependencies_constructed(self):
        raise NotImplementedError()


    def setup_filenames(self):
        # Define dataset base folder
        self.data_folder    = os.path.join(self.FOLDER, self.data_id)

        # Define filenames
        self.train_filename = os.path.join(self.data_folder, 'train.csv')
        self.val_filename   = os.path.join(self.data_folder, 'val.csv')
        self.test_filename  = os.path.join(self.data_folder, 'test.csv')
        self.char_filename  = os.path.join(self.data_folder, 'characters.csv')

        self.train_memmap_filename = os.path.join(self.data_folder, 'train.dat')
        self.val_memmap_filename   = os.path.join(self.data_folder, 'val.dat')
        self.test_memmap_filename  = os.path.join(self.data_folder, 'test.dat')

        self.meta_data_filename = os.path.join(self.data_folder, 'meta.json')

        self.vectorizer_filename = os.path.join(self.data_folder, 'vectorizer.pkl')

        self.vocabulary_filename = os.path.join(self.data_folder, 'vocabulary.csv')

        # Create folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)


    def extract_row(self, row):
        return {
            'enc_input': [int(val) for val in row[0].split(' ')],
            'dec_input': [int(val) for val in row[1].split(' ')],
            'dec_target': [int(val) for val in row[2].split(' ')],
            'space_idx_fw': [int(val) for val in row[3].split(' ')],
            'space_idx_bw': [int(val) for val in row[4].split(' ')],
            'enc_input_length': int(row[5]),
            'dec_input_length': int(row[6]),
            'word_count': int(row[7])
        }


    def _print(self, message):
        if self.verbose:
            print('[%s] %s' % (type(self).__name__, message))
