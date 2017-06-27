
import os
import random
import multiprocessing

import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

import data


class RandomForestWrapper:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

    def __init__(self, max_enc_seq_length, max_dec_seq_length, random_seed, n=1000, verbose=False):
        self.max_enc_seq_length = max_enc_seq_length
        self.max_dec_seq_length = max_dec_seq_length
        self.random_seed        = random_seed
        self.n                  = n
        self.verbose            = verbose

        # NOTE: The word features do not depend on this
        self.max_enc_seq_length = 100
        self.max_dec_seq_length = 100

        # Set random seed
        np.random.seed(self.random_seed)

        # Make models folder if doesn't exist
        if not os.path.isdir(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR)

        # Persistency
        self.name = '-'.join([
            '%s' % (type(self).__name__),
            '%d' % (self.max_enc_seq_length),
            '%d' % (self.max_dec_seq_length),
            '%d' % (self.random_seed),
            '%d' % (self.n),
        ])
        self.model_folder = os.path.join(self.MODELS_DIR, self.name)
        self.question_model_filepath = os.path.join(self.model_folder, 'question-model.pkl')
        self.answer_model_filepath   = os.path.join(self.model_folder, 'answer-model.pkl')

        if not self.try_load_models():
            self._print('Training Question/Answer classifiers..')
            self.setup_classifiers()
        else:
            self._print('Classifiers loaded successfully!')

    def predict_question(self, word_features):
        probs = self.question_classifier.predict_proba([word_features])
        return probs.squeeze()

    def predict_answer(self, word_features):
        probs = self.answer_classifier.predict_proba(word_features)
        return probs

    def try_load_models(self):
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
            return False

        if not os.path.isfile(self.question_model_filepath):
            return False

        if not os.path.isfile(self.answer_model_filepath):
            return False

        # Load question classifier
        try:
            self.question_classifier = joblib.load(self.question_model_filepath)
        except pickle.PickleError:
            return False

        # Load answer classifier
        try:
            self.answer_classifier = joblib.load(self.answer_model_filepath)
        except pickle.PickleError:
            return False

        # Everything was loaded without any errors
        return True

    def setup_classifiers(self):
        # Define alphabet
        alphabet = data.Alphabet.create_standard_alphabet()

        # Initialize data loaders
        kwargs = {
            'max_enc_seq_length': self.max_enc_seq_length,
            'max_dec_seq_length': self.max_dec_seq_length,
            'random_seed':        self.random_seed,
            'classes_only':       True,
            'alphabet':           alphabet
        }
        train_loader = data.QALoader(validation='train', **kwargs)
        val_loader   = data.QALoader(validation='val',   **kwargs)
        test_loader  = data.QALoader(validation='test',  **kwargs)

        def construct_classifier(features_name):
            # Construct inputs and target vectors
            def extract_values(data_loader):
                inputs, targets = [], []
                for data_row in data_loader:
                    # Get class vector
                    idx = data_loader.data_indices['class_count']
                    class_idx = random.randint(0, data_row[idx[0]] - 1)
                    idx = data_loader.data_indices['classes']
                    classes = data_row[idx[0]:idx[1]]
                    y = classes[class_idx]

                    # Get word features
                    idx = data_loader.data_indices[features_name]
                    input_word_features = data_row[idx[0]:idx[1]]

                    inputs.append(input_word_features)
                    targets.append(y)
                return np.asarray(inputs), np.asarray(targets)

            X_train, Y_train = extract_values(train_loader)
            X_val, Y_val     = extract_values(val_loader)
            X_test, Y_test   = extract_values(test_loader)

            return self.train_classifier(
                X={
                    'train': X_train,
                    'val':   X_val,
                    'test':  X_test
                },
                Y={
                    'train': Y_train,
                    'val':   Y_val,
                    'test':  Y_test
                }
            )

        self.question_classifier = construct_classifier('input_word_features')
        self.answer_classifier   = construct_classifier('target_word_features')

        # Save models
        joblib.dump(self.question_classifier, self.question_model_filepath)
        joblib.dump(self.answer_classifier,   self.answer_model_filepath)


    def train_classifier(self, X, Y):
        criterion_values = ['gini', 'entropy']
        max_feature_values = ['sqrt', 'log2', None]

        clf_default_kwargs = {
            'max_depth':         None,
            'min_samples_split': 2,
            'min_samples_leaf':  1,
            'n_jobs':            multiprocessing.cpu_count()
        }

        accuracies = np.zeros((len(criterion_values), len(max_feature_values)))
        for i, criterion in enumerate(criterion_values):
            for j, max_features in enumerate(max_feature_values):
                # Initialize random forest
                clf = RandomForestClassifier(
                    n_estimators=self.n // 10,  # Use less trees for grid search
                    criterion=criterion,
                    max_features=max_features,
                    **clf_default_kwargs
                )

                # Train classifier
                clf.fit(X['train'], Y['train'])

                # Evaluate
                accuracies[i,j] = clf.score(X['val'], Y['val'])

        # Get optimal values
        i_opt, j_opt = np.unravel_index(accuracies.argmax(), accuracies.shape)
        criterion_opt = criterion_values[i_opt]
        max_features_opt = max_feature_values[j_opt]

        # Initialize optimal forest
        clf = RandomForestClassifier(
            n_estimators=self.n,
            criterion=criterion_opt,
            max_features=max_features_opt,
            **clf_default_kwargs
        )

        # Train optimal forest
        clf.fit(X['train'], Y['train'])

        # Evaluate on test
        test_accuracy = clf.score(X['test'], Y['test'])
        self._print('Final test accuracy: %g' % (test_accuracy))
        return clf

    def _print(self, message):
        if self.verbose:
            print('[%s] %s' % (type(self).__name__, message))
