
import json
import collections

import scipy        # Import these before tensorflow to prevent segmentationfault
import numpy as np  # Import these before tensorflow to prevent segmentationfault
import tensorflow as tf

import helpers
import utils

# Get flags
FLAGS = helpers.get_flags()

# Initialize data loader
data_loader_dict = helpers.setup_batch_loaders(
    data_type=FLAGS.data,
    mix_data_loaders=FLAGS.mix_data,
    max_enc_seq_length=FLAGS.max_enc_seq_length,
    max_dec_seq_length=FLAGS.max_dec_seq_length,
    batch_size=FLAGS.batch_size,
    random_seed=FLAGS.random_seed,
    model=FLAGS.model.lower(),
    mix_ratio=FLAGS.mix_ratio,
    verbose=FLAGS.verbose
)
vocabulary = data_loader_dict['vocabulary']

# Setup config
config = helpers.setup_config(
    flags=FLAGS,
    data_name=data_loader_dict['name'],
    alphabet=data_loader_dict['alphabet'],
    num_classes=data_loader_dict['num_classes']
)

with open(config.prediction_filename, 'r') as f:
    predictions_dict = json.load(f)

references_lists, references_tokens = [], []
candidates_tokens = [[] for _ in range(0, FLAGS.beam_size)]
candidates_lists  = [[] for _ in range(0, FLAGS.beam_size)]
classes, class_predictions = [], []

stats = collections.defaultdict(lambda: [])
for prediction_dict in predictions_dict['predictions']:
    opt_idx       = prediction_dict.get('optimal_candidate_idx', None)
    target_list = utils.extract_words(prediction_dict['target'])
    references_lists.append(target_list)
    references_tokens.append(' '.join(map(str, vocabulary.encode_seq(target_list))))

    if 'class' in prediction_dict:
        classes.append(prediction_dict['class'])
    if 'class_prediction' in prediction_dict:
        class_predictions.append(prediction_dict['class_prediction'])

    for i, candidate_dict in enumerate(prediction_dict['candidates']):
        stats['losses'].append(candidate_dict['loss'])
        stats['prob_x_values'].append(candidate_dict['prob_x'])

        candidate_list = utils.extract_words(candidate_dict['prediction'])
        candidates_lists[i].append(candidate_list)
        candidates_tokens[i].append(' '.join(map(str, vocabulary.encode_seq(candidate_list))))

moses_bleu_scores, bleu_scores = [], []
for i in range(0, FLAGS.beam_size):
    bleu = utils.corpus_bleu(candidates_lists[i], references_lists)
    moses_bleu = utils.compute_moses_bleu(candidates_tokens[i], references_tokens)

    bleu_scores.append(bleu)
    moses_bleu_scores.append(moses_bleu)
bleu_scores, moses_bleu_scores = np.asarray(bleu_scores), np.asarray(moses_bleu_scores)

mean_bleu       = float(bleu_scores.mean())
std_bleu        = float(bleu_scores.std())
mean_moses_bleu = float(moses_bleu_scores.mean())
std_moses_bleu  = float(moses_bleu_scores.std())

mean_loss   = float(np.asarray(stats['losses']).mean())
mean_prob_x = float(np.asarray(stats['prob_x_values']).mean())


predictions_dict['summary'] = {
    'mean_bleu':        mean_bleu,
    'std_bleu':         std_bleu,
    'mean_moses_bleu':  mean_moses_bleu,
    'std_moses_bleu':   std_moses_bleu,
    'mean_loss':        mean_loss,
    'mean_prob_x':      mean_prob_x
}

# Compute accuracy if needed
if len(classes) > 0 and len(class_predictions) > 0 and len(classes) == len(class_predictions):
    classes = np.asarray(classes)
    class_predictions = np.asarray(class_predictions)

    has_class = (classes >= 0)
    has_class_count = has_class.astype('int').sum()
    if has_class_count > 0:
        accuracy = (classes[has_class] == class_predictions[has_class]).astype('int').sum() / has_class_count
        predictions_dict['summary']['accuracy'] = accuracy

# Sort predictions by loss
predictions_dict['predictions'] = sorted(predictions_dict['predictions'], key=lambda x: sum(c['loss'] for c in x['candidates']))

with open(config.prediction_filename, 'wt') as f:
    json.dump(predictions_dict, f, indent=4, sort_keys=True)
