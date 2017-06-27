
import scipy        # Import these before tensorflow to prevent segmentationfault
import numpy as np  # Import these before tensorflow to prevent segmentationfault
import tensorflow as tf

import data
import utils
import helpers
import classifier

def main():
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

    # Setup model
    graph = tf.Graph()
    with graph.as_default():
        model = helpers.setup_model(
            flags=FLAGS,
            prediction_mode=True,
            alphabet=data_loader_dict['alphabet'],
            vocabulary=data_loader_dict['vocabulary'],
            data_name=data_loader_dict['name'],
            num_classes=data_loader_dict['num_classes']
        )

    if FLAGS.lm_support:
        lm_graph = tf.Graph()

        with lm_graph.as_default():
            lm_flags = helpers.get_lm_flags(FLAGS)
            lm_config = helpers.setup_config(
                flags=lm_flags,
                data_name=data.WikiLoader.__name__,
                alphabet=data_loader_dict['alphabet'],
                num_classes=None
            )
            lm_predict_func = helpers.load_frozen_network(config=lm_config)
    else:
        lm_predict_func = None


    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    if FLAGS.lm_support:
        lm_session_config = tf.ConfigProto(device_count={'GPU': 0})
        with tf.Session(config=config, graph=graph) as sess, tf.Session(config=lm_session_config, graph=lm_graph) as lm_sess:
            predict(model, FLAGS, data_loader_dict, graph, session=sess, lm_session=lm_sess, lm_predict_func=lm_predict_func)
    else:
        with tf.Session(config=config, graph=graph) as sess:
            predict(model, FLAGS, data_loader_dict, graph, session=sess)


def predict(model, FLAGS, data_loader_dict, graph, session, lm_session=None, lm_predict_func=None):
    # Initialize model
    with graph.as_default():
        model.init(session)

    if FLAGS.beam_size > 1:
        # Initialize classifier
        clf = classifier.RandomForestWrapper(
            max_enc_seq_length=FLAGS.max_enc_seq_length,
            max_dec_seq_length=FLAGS.max_dec_seq_length,
            n=100,
            random_seed=FLAGS.random_seed,
            verbose=FLAGS.verbose
        )
        vectorizer = data_loader_dict['test_loader'].data_loader.vectorizer

    for data_dict in data_loader_dict['test_loader']:
        # Predict
        result = model.predict(session, lm_session=lm_session, lm_predict_func=lm_predict_func, **data_dict)

        for i, candidates in enumerate(result['candidates']):
            opt_candidate_idx = None
            if FLAGS.beam_size > 1:
                opt_candidate_idx = utils.optimal_candidate_idx(
                    candidates=candidates,
                    alphabet=data_loader_dict['alphabet'],
                    vectorizer=vectorizer,
                    classifier=clf,
                    input_word_features=data_dict['input_word_features'][i]
                )

            prediction_dict = {
                'input':                    data_dict['enc_input'][i],
                'target':                   data_dict['dec_target'][i],
                'candidates':               candidates,
                'loss_candidates':          result['loss_candidates'][i],
                'prob_x_candidates':        result['prob_x_candidates'][i],
                'optimal_candidate_idx':    opt_candidate_idx
            }

            if model.prediction_maintainer.include_accuracy:
                prediction_dict['class_true'] = data_dict['class_idx'][i]
                prediction_dict['class_prediction'] = result['class_predictions'][i]

            # Add prediction
            model.prediction_maintainer.add_prediction(**prediction_dict)
    model.prediction_maintainer.save_predictions()


if __name__ == '__main__':
    main()
