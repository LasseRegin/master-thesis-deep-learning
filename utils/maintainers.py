
import os
import sys
import csv
import json
import collections

import numpy as np

from .metrics import sentence_bleu, edit_distance, compute_moses_bleu
from .strings import extract_words

class LossMaintainer:
    def __init__(self, epochs_filename, epochs, include_accuracy=False,
                 verbose=False, **kwargs):
        self.verbose          = verbose
        self.epochs_filename  = epochs_filename
        self.run_epochs       = epochs
        self.include_accuracy = include_accuracy

        # Load epochs
        self.epochs = collections.OrderedDict()
        if os.path.isfile(self.epochs_filename):
            with open(self.epochs_filename, 'r') as f:
                epochs_dict = json.load(f)

                for epoch, values in sorted(epochs_dict['epochs'].items(), key=lambda x: int(x[0])):
                    self.epochs[int(epoch)] = values

        # Initialize empty prediction container
        self.max_epochs = self.run_epochs + len(self.epochs)

    def add_epoch(self, step, train_loss, train_prob_x, train_accuracy, val_loss, val_prob_x, val_accuracy, **kwargs):
        max_step = max([0] + [int(x) for x in self.epochs.keys()])
        if step < max_step:
            raise KeyError('Provided global step is smaller than previous steps!')

        epoch = len(self.epochs) + 1
        self._print('Epoch %d/%d (global step: %d)' % (epoch, self.max_epochs, step))

        self._print('\tTrain loss:\t%g'             % (train_loss))
        self._print('\tTrain prob_x:\t%g'           % (train_prob_x))
        if self.include_accuracy:
            self._print('\tTrain accuracy:\t%g'     % (train_accuracy))

        self._print('\tVal loss:\t%g'   % (val_loss))
        self._print('\tVal prob_x:\t%g' % (val_prob_x))
        if self.include_accuracy:
            self._print('\tVal accuracy:\t%g' % (val_accuracy))

        self.epochs[epoch] = {
            'step':         int(step),
            'train_loss':   float(train_loss),
            'train_prob_x': float(train_prob_x),
            'val_loss':     float(val_loss),
            'val_prob_x':   float(val_prob_x)
        }
        if self.include_accuracy:
            self.epochs[epoch]['train_accuracy'] = float(train_accuracy)
            self.epochs[epoch]['val_accuracy']   = float(val_accuracy)

    def save_epochs(self):
        train_losses, val_losses = [], []
        train_prob_x, val_prob_x = [], []
        if self.include_accuracy:
            train_accuracy, val_accuracy = [], []

        for epoch, values in sorted(self.epochs.items(), key=lambda x: int(x[0])):
            train_losses.append(values['train_loss'])
            train_prob_x.append(values['train_prob_x'])
            val_losses.append(values['val_loss'])
            val_prob_x.append(values['val_prob_x'])
            if self.include_accuracy:
                train_accuracy.append(values['train_accuracy'])
                val_accuracy.append(values['val_accuracy'])

        self._print('')
        self._print('Final training loss:\t\t%g'         % (train_losses[-1]))
        self._print('Final training prob_x:\t\t%g'       % (train_prob_x[-1]))
        if self.include_accuracy:
            self._print('Final training accuracy:\t%g'   % (train_accuracy[-1]))

        self._print('Final validation loss:\t\t%g'       % (val_losses[-1]))
        self._print('Final validation prob_x:\t%g'       % (val_prob_x[-1]))
        if self.include_accuracy:
            self._print('Final validation accuracy:\t%g' % (val_accuracy[-1]))
        self._print('')

        epochs_dict = {
            'epochs': self.epochs,
            'summary': {
                'epochs': len(self.epochs),
                'final_train_loss':   train_losses[-1],
                'final_train_prob_x': train_prob_x[-1],
                'final_val_loss':     val_losses[-1],
                'final_val_prob_x':   val_prob_x[-1],
            }
        }

        if self.include_accuracy:
            epochs_dict['summary']['final_train_accuracy'] = train_accuracy[-1]
            epochs_dict['summary']['final_val_accuracy'] = val_accuracy[-1]

        with open(self.epochs_filename, 'wt') as f:
            json.dump(epochs_dict, f, indent=4, sort_keys=True)

    def reset_loss(self):
        self.epochs = collections.OrderedDict()
        if os.path.isfile(self.epochs_filename):
            os.remove(self.epochs_filename)
        self.max_epochs = self.run_epochs

    def is_improving(self, patience):
        """
            Returns true if the validation error still is improving, and
            False otherwise.

            `patience`  defines the maxmimum number of epochs to wait
                        for the validation loss to improve.
        """
        val_losses = np.array([epoch['val_loss'] for epoch in self.epochs.values()])
        improvement_delay = len(val_losses) - val_losses.argmin()
        _is_improving = improvement_delay < patience
        if not _is_improving:
            self._print('Validation loss has not improved for a while.')
            self._print('Stopping early..')

        # Flush stdout
        sys.stdout.flush()

        return _is_improving

    def _print(self, message):
        if self.verbose:
            print('[%s] %s' % (type(self).__name__, message))



class MixedLossMaintainer:
    def __init__(self, epochs_filename, **kwargs):
        self.epochs_filename  = epochs_filename

        # Load epochs
        self.epochs = collections.OrderedDict()
        if os.path.isfile(self.epochs_filename):
            with open(self.epochs_filename, 'r') as f:
                epochs_dict = json.load(f)

                for epoch, values in sorted(epochs_dict['epochs'].items(), key=lambda x: int(x[0])):
                    self.epochs[int(epoch)] = values

    def add_epoch(self, step, train_loss_1, train_loss_2, val_loss_1, val_loss_2, **kwargs):
        max_step = max([0] + [int(x) for x in self.epochs.keys()])
        if step < max_step:
            raise KeyError('Provided global step is smaller than previous steps!')

        epoch = len(self.epochs) + 1

        self.epochs[epoch] = {
            'step':           int(step),
            'train_loss_1':   float(train_loss_1),
            'train_loss_2':   float(train_loss_2),
            'val_loss_1':     float(val_loss_1),
            'val_loss_2':     float(val_loss_2),
        }

    def save_epochs(self):
        epochs_dict = {
            'epochs': self.epochs
        }

        with open(self.epochs_filename, 'wt') as f:
            json.dump(epochs_dict, f, indent=4, sort_keys=True)

    def reset_loss(self):
        self.epochs = collections.OrderedDict()
        if os.path.isfile(self.epochs_filename):
            os.remove(self.epochs_filename)
        self.max_epochs = self.run_epochs


class PredictionMaintainer:
    def __init__(self, alphabet, prediction_filename, sample_type, include_accuracy=False,
                 verbose=False, **kwargs):
        self.alphabet               = alphabet
        self.verbose                = verbose
        self.include_accuracy       = include_accuracy
        self.prediction_filename    = prediction_filename

        self.predictions = []

    def add_prediction(
        self,
        input,
        target,
        candidates,
        loss_candidates,
        prob_x_candidates,
        optimal_candidate_idx=None,
        class_true=None,
        class_prediction=None
    ):
        # Decode and convert to strings
        input_string      = self.alphabet.seq2str(input,      decode=True)
        target_string     = self.alphabet.seq2str(target,     decode=True)

        candidate_strings = [self.alphabet.seq2str(candidate, decode=True)
                             for candidate in candidates]

        self._print('Input:')
        self._print(input_string)
        self._print('Target:')
        self._print(target_string)

        # Show classification
        if self.include_accuracy:
            self._print('Class: ')
            self._print(class_true)
            self._print('Predicted class: ')
            self._print(class_prediction)

        self._print('Candidates:')
        candidate_dicts = []
        for i, candidate_string in enumerate(candidate_strings):
            candidate_dicts.append({
                'prediction':   candidate_string,
                'loss':         float(loss_candidates[i]),
                'prob_x':       float(prob_x_candidates[i])
            })
            self._print(candidate_string)


        # Get optimal candidate
        if optimal_candidate_idx:
            optimal_candidate_string = candidate_strings[optimal_candidate_idx]
            self._print('Optimal candidate:')
            self._print(optimal_candidate_string)

        self._print('')

        prediction_dict = {
            'input':                    input_string,
            'target':                   target_string,
            'candidates':               candidate_dicts,
            'optimal_candidate_idx':    int(optimal_candidate_idx) if optimal_candidate_idx else None
        }

        if self.include_accuracy:
            prediction_dict['class'] = int(class_true)
            prediction_dict['class_prediction'] = int(class_prediction)

        self.predictions.append(prediction_dict)

    def save_predictions(self):
        with open(self.prediction_filename, 'wt') as f:
            json.dump({
                'predictions': self.predictions
            }, f, indent=4, sort_keys=True)

    def _print(self, message):
        if self.verbose:
            print('[%s] %s' % (type(self).__name__, message))
