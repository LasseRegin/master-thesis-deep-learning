
import os
import shutil
import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

from utils.maintainers import LossMaintainer, PredictionMaintainer
from utils.rnn import sequence_losses, sequence_probability
from utils.prediction import BeamSearchPredictor, SamplingPredictor, ArgmaxPredictor
from utils.ops import pack_state_tuple


class RNN:
    # Name of output nodes for frozen graph
    output_node_names = [
        'decoder_2/decoder_output',
        'decoder_2/decoder_probs',
        'decoder_2/decoder_final_state',
    ]

    def __init__(
        self,
        config,
        alphabet,
        vocabulary,
        ss_decay_func,
        learning_rate,
        sample_type=None,
        beam_size=None,
        clean=False,
        prediction_mode=False,
        parallel_iterations=32,
        swap_memory=False,
        create_summary=False,
        verbose=False,
        **kwargs
    ):
        assert learning_rate > 0.0
        assert alphabet is not None
        assert isinstance(prediction_mode, bool)
        assert beam_size > 0

        # Alphabet class
        self.alphabet = alphabet

        # Vocabulary class
        self.vocabulary = vocabulary

        # Prediction mode - for feeding decoder with previous input, instead of
        # using a given decoder input.
        self.prediction_mode = prediction_mode

        # Network config
        self.config = config

        # Parameters
        self.learning_rate        = learning_rate
        self.verbose              = verbose
        self.clean                = clean
        self.parallel_iterations  = parallel_iterations
        self.swap_memory          = swap_memory
        self.beam_size            = beam_size
        self.sample_type          = sample_type
        self.create_summary       = create_summary

        # Function mapping global_step->probability
        # where `probability` is the probability of using the correct
        # token instead of the predicted token at previous time-step during
        # inference in training.
        self.ss_decay_func = ss_decay_func

        # Determine if dropout should be added
        self.add_dropout = self.config.keep_prob < 1.0 and not self.prediction_mode

        # Input placeholders
        self.encoder_input_chars = tf.placeholder(tf.int32, shape=[None, self.config.max_enc_seq_length],     name='encoder_input_chars')
        self.decoder_input_chars = tf.placeholder(tf.int32, shape=[None, self.config.max_dec_seq_length + 1], name='decoder_input_chars')
        self.target_chars        = tf.placeholder(tf.int32, shape=[None, self.config.max_dec_seq_length + 1], name='target_chars')

        self.enc_word_indices = tf.placeholder(tf.int32, shape=[None, self.config.max_words], name='enc_word_indices')

        self.encoder_sequence_length = tf.placeholder(tf.int32, shape=[None], name='encoder_sequence_length')
        self.decoder_sequence_length = tf.placeholder(tf.int32, shape=[None], name='decoder_sequence_length')
        self.word_seq_lengths        = tf.placeholder(tf.int32, shape=[None], name='encoder_word_sequence_length')

        # Categories
        self.class_idx = tf.placeholder(tf.int32, shape=[None], name='class_idx')

        # Placeholder for binary value telling the decoder whether to be in
        # "language model mode" or "question answering mode"
        self.is_lm_mode   = tf.placeholder(tf.int32, shape=[None], name='is_language_model_mode')

        # Placeholder for decoder output used for computing loss
        self.decoder_logits = tf.placeholder(tf.float32, shape=[None, self.config.max_dec_seq_length + 1, self.config.alphabet_size], name='decoder_logits')

        # Other placeholders
        self.global_step           = tf.Variable(0,   trainable=False, name='global_step')
        self.keep_prob_ph          = tf.Variable(1.0, trainable=False, name='input_keep_ph', dtype=tf.float32)
        self.probs_decay_parameter = tf.placeholder(tf.float64, shape=(), name='probs_decay_parameter')
        self.infer_token_prob      = tf.placeholder(tf.float64, shape=(), name='infer_token_prob')

        # Define function mapping global_step->probability
        # where `probability` is the probability of using the correct
        # token instead of the predicted token at previous time-step during
        # inference in training.
        # self.infer_token_prob_func = lambda x: 1.0 - (x * 6.2446 + 139.9846)

        # #
        # x * 6.2446 + 139.9846
        # n / batch_size

        # Define persistencies
        if self.prediction_mode:
            self.prediction_maintainer = PredictionMaintainer(
                alphabet=self.alphabet,
                sample_type=self.sample_type,
                prediction_filename=self.config.prediction_filename,
                verbose=self.verbose,
                **kwargs
            )
        else:
            self.loss_maintainer = LossMaintainer(
                epochs_filename = self.config.epochs_filename,
                verbose=self.verbose,
                **kwargs
            )

    def setup_network(self):
        raise NotImplementedError()

    def setup_character_embedding(self):
        with tf.device('/cpu:0'), tf.variable_scope(name_or_scope='embedding'):
            self.embedding_matrix = tf.get_variable(
                shape=[self.config.alphabet_size, self.config.embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                name='W'
            )

            # Gather slices from `params` according to `indices`
            enc_input_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.encoder_input_chars, name='enc_input')
            dec_input_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_input_chars, name='dec_input')

        # Expand binary mode value to all time_steps and append to decoder input
        is_lm_mode = tf.to_float(self.is_lm_mode)
        is_lm_mode = tf.expand_dims(is_lm_mode, axis=1)
        is_lm_mode = tf.expand_dims(is_lm_mode, axis=2)
        is_lm_mode = tf.tile(is_lm_mode, [1, self.config.max_dec_seq_length + 1, 1], name='is_language_model_mode')

        dec_input_embedded = tf.concat([dec_input_embedded, is_lm_mode], axis=2, name='merged_dec_input')

        # Use tf.gather since this can be used on a GPU
        def embed_func(input_chars):
            input_embedded = tf.gather(self.embedding_matrix, input_chars)
            is_lm_mode = tf.to_float(self.is_lm_mode)
            is_lm_mode = tf.expand_dims(is_lm_mode, axis=1)
            return tf.concat([input_embedded, is_lm_mode], axis=1, name='input_embedded')

        return enc_input_embedded, dec_input_embedded, embed_func

    def setup_losses(self, dec_outputs, target_chars, decoder_sequence_length):
        # Compute masked sequence losses
        self.losses = sequence_losses(
            logits=dec_outputs,
            labels=target_chars,
            seq_lengths=decoder_sequence_length
        )

        # Define mean loss
        self.mean_loss_batch = tf.reduce_mean(self.losses, axis=1,  name='mean_loss_batch')
        self.mean_loss       = tf.reduce_mean(self.mean_loss_batch, name='mean_loss')

        # Compute probabilities
        self.mean_prob_x_batch = sequence_probability(
            logits=dec_outputs,
            labels=target_chars
        )
        self.mean_prob_x = tf.reduce_mean(self.mean_prob_x_batch, name='mean_prob_x')

        if self.create_summary:
            summary = tf.summary.scalar(self.mean_loss.name, self.mean_loss)
            self.train_summaries.append(summary)
            self.val_summaries.append(summary)

    def setup_optimizer(self):
        # Get trainable variables
        params = tf.trainable_variables()

        # Define optimizer algorithm
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Get and clip gradients
        self.gradients = tf.gradients(self.losses, params)
        clipped_gradients, norm = tf.clip_by_global_norm(
            t_list=self.gradients,
            clip_norm=self.config.max_gradient_norm,
            name='gradient-clipping'
        )

        if self.create_summary:
            for variable in params:
                self.train_summaries.append(tf.summary.histogram(variable.name, variable))

            for gradient in self.gradients:
                if gradient is not None:
                    self.train_summaries.append(tf.summary.histogram(gradient.name, gradient))

        # Define update operation
        self.update_op = optimizer.apply_gradients(
            grads_and_vars=zip(clipped_gradients, params),
            global_step=self.global_step
        )

    def init(self, session):
        # Delete saved checkpoints if `self.clean=True`
        if self.clean:
            shutil.rmtree(self.config.model_folder, ignore_errors=True)
            self._print('Deleted saved checkpoints for: %s' % (self.config.name))

            # Remake folders
            self.config.create_folders()

        self.train_summaries = []
        self.val_summaries = []

        # Setup network
        self.setup_network()

        if not self.prediction_mode:
            # Define optimizer (defines `self.gradients` and `self.updates_op`)
            self.setup_optimizer()

        # Define saver operation
        self.saver = tf.train.Saver(tf.global_variables())

        # Define variable initializer
        self.init_op = tf.global_variables_initializer()

        # Setup summary
        if self.create_summary:
            self.train_summary = tf.summary.merge(self.train_summaries)
            self.val_summary = tf.summary.merge(self.val_summaries)

            self.train_writer = tf.summary.FileWriter(
                logdir=self.config.summary_train_path,
                graph=session.graph
            )
            self.val_writer = tf.summary.FileWriter(
                logdir=self.config.summary_val_path,
                graph=session.graph
            )
        else:
            self.train_summary = tf.constant(-1)
            self.val_summary = tf.constant(-1)

        checkpoint = tf.train.get_checkpoint_state(self.config.model_folder)

        loading_error = None
        if checkpoint:

            # Create model filepath from checkpoint (this is done because the
            # checkpoint path is saved as an absolute path hence it would not work
            # when imported from another system)
            model_filename = os.path.basename(checkpoint.model_checkpoint_path)
            model_filepath = os.path.join(self.config.model_folder, model_filename)

            try:
                self.saver.restore(session, save_path=model_filepath)
                step = session.run(self.global_step)
                self._print('Restored model from %s' % (model_filepath))
                self._print('Continuing from global step: %d' % (step))
            except tf.errors.NotFoundError:
                self._print('Found model does not fit current model!')
                loading_error = True

        if not checkpoint or loading_error:
            self._print('Initialized model: %s' % (self.config.name))
            session.run(self.init_op)
            if not self.prediction_mode:
                self.loss_maintainer.reset_loss()


    def save(self, session):
        # Save model
        filepath = self.saver.save(
            sess=session,
            save_path=self.config.save_path,
            global_step=self.global_step
        )

        # Save loss
        self.loss_maintainer.save_epochs()

        if hasattr(self, 'mixed_loss_maintainer'):
            self.mixed_loss_maintainer.save_epochs()

        self._print('Saved model to: %s' % (filepath))

    def load(self, session):
        self.saver.restore(
            sess=session,
            save_path=self.config.model_folder
        )

    def get_global_step(self, session):
        return session.run(self.global_step)

    def build_feed_dict(self, validation, session, enc_input, dec_target,
                        enc_input_length, dec_input_length, is_lm_mode,
                        **kwargs):
        assert validation in ['train', 'val', 'test']
        feed_dict = {}

        # Get global step
        global_step = self.get_global_step(session)

        # Overwrite some values
        if validation == 'test':
            # Pass `<PAD_ID>` symbols as decoder input with the first element
            # being the `<GO>` symbol
            dec_input = np.full((len(enc_input), self.config.max_dec_seq_length + 1), self.alphabet.PAD_ID)
            dec_input[:,0] = self.alphabet.GO_ID
        else:
            dec_input = kwargs['dec_input']

        # Values all operations need
        feed_dict[self.encoder_input_chars]     = enc_input
        feed_dict[self.decoder_input_chars]     = dec_input
        feed_dict[self.target_chars]            = dec_target
        feed_dict[self.encoder_sequence_length] = enc_input_length
        feed_dict[self.decoder_sequence_length] = dec_input_length

        # Keep probability
        if validation == 'train':
            feed_dict[self.keep_prob_ph] = self.config.keep_prob
        else:
            feed_dict[self.keep_prob_ph] = 1.0

        # Scheduled sampling
        if validation == 'train':
            # feed_dict[self.infer_token_prob] = 0.0
            # feed_dict[self.infer_token_prob] = 0.1
            # feed_dict[self.infer_token_prob] = 0.5 # TODO: Fix
            # feed_dict[self.infer_token_prob] = 1.0
            feed_dict[self.infer_token_prob] = self.ss_decay_func(global_step)
        else:
            feed_dict[self.infer_token_prob] = 1.0

        # Language model mode
        feed_dict[self.is_lm_mode] = is_lm_mode

        # If class vector is provided
        if 'class_idx' in kwargs:
            feed_dict[self.class_idx] = kwargs['class_idx']
        else:
            feed_dict[self.class_idx] = np.zeros(len(enc_input))

        # If using word features
        if hasattr(self, 'input_word_features'):
            feed_dict[self.input_word_features] = kwargs['input_word_features']

        # If using char-2-word encoder
        if 'ws_indices' in kwargs and hasattr(self, 'enc_word_indices'):
            feed_dict[self.enc_word_indices] = kwargs['ws_indices']
        if 'word_count' in kwargs and hasattr(self, 'word_seq_lengths'):
            feed_dict[self.word_seq_lengths] = kwargs['word_count']

        return feed_dict

    def train_op(self, session, **kwargs):
        assert not self.prediction_mode

        _, mean_loss, mean_prob_x, summary, global_step = session.run(
            fetches=[self.update_op, self.mean_loss, self.mean_prob_x, self.train_summary, self.global_step],
            feed_dict=self.build_feed_dict('train', session, **kwargs)
        )

        return {
            'mean_loss':    mean_loss,
            'mean_prob_x':  mean_prob_x,
            'summary':      summary,
            'global_step':  global_step
        }

    def val_op(self, session, **kwargs):
        assert not self.prediction_mode

        mean_loss, mean_prob_x, summary, global_step = session.run(
            fetches=[self.mean_loss, self.mean_prob_x, self.val_summary, self.global_step],
            feed_dict=self.build_feed_dict('val', session, **kwargs)
        )

        return {
            'mean_loss':    mean_loss,
            'mean_prob_x':  mean_prob_x,
            'summary':      summary,
            'global_step':  global_step
        }

    def predict(self, session, lm_session=None, lm_predict_func=None, **kwargs):
        assert self.prediction_mode

        # Encode sequences and extract final states
        initial_state = session.run(
            fetches=self.enc_final_state_tensor,
            feed_dict=self.build_feed_dict('test', session, **kwargs)
        )

        def decode_func(inputs, state, is_lm_mode, lm_state=None, probs_decay_parameter=None):
            output, state, probs_decayed = session.run(
                fetches=[self.decoder_output, self.decoder_final_state, self.decoder_probs_decayed],
                feed_dict={
                    self.decoder_inputs:        inputs,
                    self.decoder_state:         state,
                    self.is_lm_mode:            is_lm_mode,
                    self.probs_decay_parameter: probs_decay_parameter
                }
            )

            # Use Language model
            if lm_predict_func is not None:
                lm_output, lm_probs, lm_state = lm_predict_func(lm_session, inputs, lm_state)

                # Merge logits
                output    = output.reshape(output.shape       + (1,))
                lm_output = lm_output.reshape(lm_output.shape + (1,))
                output = np.concatenate((output, lm_output,), axis=-1)
                output = np.mean(output, axis=-1)

                # Merge probabilities
                probs = probs_decayed
                probs    = probs.reshape(probs.shape       + (1,))
                lm_probs = lm_probs.reshape(lm_probs.shape + (1,))
                probs = np.concatenate((probs, lm_probs), axis=-1)
                probs = np.prod(probs, axis=-1).astype(np.float64)

                # Renormalize
                probs_sums = probs.sum(axis=1)
                probs = probs / probs_sums[:,None]

                probs_decayed = probs

            output_dict = {
                'output':           output,
                'state':            state,
                'probs_decayed':    probs_decayed
            }

            if lm_predict_func is not None:
                output_dict['lm_state'] = lm_state

            return output_dict

        def loss_func(logits, targets, input_length):
            return session.run(
                fetches=[self.mean_loss_batch, self.mean_prob_x_batch],
                feed_dict={
                    self.decoder_logits:          logits,
                    self.target_chars:            targets,
                    self.decoder_sequence_length: input_length
                }
            )

        # Construct vector of <GO_ID> tokens as initial input
        batch_size       = kwargs['enc_input'].shape[0]
        dec_target       = kwargs['dec_target']
        dec_input_length = kwargs['dec_input_length']
        max_iterations   = self.config.max_dec_seq_length + 1

        extra_features = {
            'is_lm_mode':   kwargs['is_lm_mode']
        }

        # Use Language model
        if lm_predict_func is not None:
            # TODO: Pass LM config
            lm_num_cells = 1
            lm_num_units = 1024

            lm_initial_input = np.full(shape=(batch_size,), fill_value=self.alphabet.GO_ID, dtype=np.float32)
            lm_initial_state = np.zeros([batch_size, 2, lm_num_cells, lm_num_units])

            _, _, lm_state = lm_predict_func(lm_session, lm_initial_input, lm_initial_state)
            extra_features['lm_state'] = lm_state

        # Initialize predictor
        if self.sample_type == 'beam':
            predictor = BeamSearchPredictor(
                batch_size=batch_size,
                max_length=max_iterations,
                alphabet=self.alphabet,
                decode_func=decode_func,
                loss_func=loss_func,
                beam_size=self.beam_size
            )
        elif self.sample_type == 'sample':
            predictor = SamplingPredictor(
                batch_size=batch_size,
                max_length=max_iterations,
                alphabet=self.alphabet,
                decode_func=decode_func,
                loss_func=loss_func,
                num_samples=self.beam_size,
            )
        elif self.sample_type == 'argmax':
            predictor = ArgmaxPredictor(
                batch_size=batch_size,
                max_length=max_iterations,
                alphabet=self.alphabet,
                decode_func=decode_func,
                loss_func=loss_func
            )
        else:
            raise KeyError('Invalid sample_type provided!')

        # Predict sequence candidates
        final_candidates, final_logits, loss_candidates, prob_x_candidates = predictor.predict_sequences(
            initial_state=initial_state,
            target=dec_target,
            input_length=dec_input_length,
            features=extra_features
        )

        # Remove predictions after the `<EOS>` id
        for i, j, k in zip(*np.where(final_candidates == self.alphabet.EOS_ID)):
            final_candidates[i, j, k+1:] = 0

        return {
            'candidates':           final_candidates,
            'loss_candidates':      loss_candidates,
            'prob_x_candidates':    prob_x_candidates
        }

    def encode(self, session, **kwargs):
        assert self.prediction_mode

        enc_final_state = session.run(
            fetches=[self.enc_final_state_tensor],
            feed_dict=self.build_feed_dict('test', session, **kwargs)
        )

        return enc_final_state

    def add_epoch(self, session, **kwargs):
        step = self.get_global_step(session)
        return self.loss_maintainer.add_epoch(step=step, **kwargs)

    def is_improving(self, **kwargs):
        return self.loss_maintainer.is_improving(**kwargs)

    def _print(self, message):
        if self.verbose:
            print('[%s] %s' % (type(self).__name__, message))
