
import math

import numpy as np
import tensorflow as tf

from utils.encoder import BidirectionalEncoder, Char2WordEncoder
from utils.ops import is_sequence, flatten, intialize_projections, projection
from utils.rnn import rnn_cell
from .rnn import RNN


class RNN_BLSTM_Classifier(RNN):
    def __init__(self, **kwargs):

        super().__init__(include_accuracy=True, **kwargs)

        # Define encoder
        with tf.variable_scope('encoder'):
            cell_encoder = rnn_cell(
                num_units=self.config.num_units,
                num_cells=self.config.num_cells,
                keep_prob=self.keep_prob_ph,
                add_dropout=self.add_dropout
            )

            # self.encoder = BidirectionalEncoder(
            #     cell=cell_encoder,
            #     parallel_iterations=self.parallel_iterations,
            #     swap_memory=self.swap_memory
            # )

            self.encoder = Char2WordEncoder(
                cell=cell_encoder,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )

    def setup_network(self):
        # Setup character embedding
        embedded_encoder_input, embedded_decoder_input, embed_func = self.setup_character_embedding()

        # Output projection
        with tf.variable_scope('alphabet_projection') as scope:
            self.projection_W, self.projection_b = intialize_projections(
                input_size=4 * self.config.num_units, # Because we use bidirectional encoder
                output_size=self.config.alphabet_size,
                scope=scope
            )

        # Encoder
        with tf.variable_scope('encoder') as scope:
            # Normalize batch
            # TODO: What axis should it be?
            embedded_encoder_input = tf.layers.batch_normalization(
                inputs=embedded_encoder_input,
                center=True,
                scale=True,
                training=not self.prediction_mode,
                trainable=True,
            )

            enc_outputs, enc_final_state = self.encoder.encode(
                inputs=embedded_encoder_input,
                seq_lengths=self.encoder_sequence_length,
                enc_word_indices=self.enc_word_indices,
                word_seq_lengths=self.word_seq_lengths,
                max_words=self.config.max_words,
                scope=scope
            )

        # Predict question categories
        with tf.variable_scope('question') as scope:

            # Convert StateTuple to vector
            state_vector = tf.concat(flatten(enc_final_state), axis=1, name='combined-state-vec')

            # Add dense layer
            W, b = intialize_projections(
                input_size=4 * self.config.num_units * self.config.num_cells,
                output_size=128
            )
            layer = tf.nn.relu(tf.matmul(state_vector, W) + b)
            if self.add_dropout:
                layer = tf.nn.dropout(
                    x=layer,
                    keep_prob=self.keep_prob_ph
                )

            # Compute L2-weight decay
            W_penalty = tf.contrib.layers.apply_regularization(
                regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.W_lambda),
                weights_list=[W]
            )

            class_logits = projection(
                x=layer,
                input_size=128,
                output_size=self.config.num_classes
            )

        # Define loss
        self.setup_losses(
            class_logits=class_logits,
            class_idx=self.class_idx,
            W_penalty=W_penalty
        )

    def setup_losses(self, class_logits, class_idx, W_penalty):

        # Create on-hot-encoded vectors
        class_one_hot = tf.one_hot(
            indices=self.class_idx,
            depth=self.config.num_classes,
            on_value=1.0,
            off_value=0.0,
            axis=-1,
            dtype=tf.float32,
            name='class-one-hot-encoded'
        )

        # Compute cross entropy manually
        class_probs = tf.nn.softmax(class_logits)
        class_probs_clipped = tf.clip_by_value(
            class_probs,
            clip_value_min=1e-8,
            clip_value_max=1.0 - 1e-8
        )
        self.losses = -tf.reduce_mean(class_one_hot * tf.log(class_probs_clipped), axis=1)
        self.losses += W_penalty

        # Compute mean loss
        self.mean_loss = tf.reduce_mean(self.losses, name='class-loss')

        # Compute accuracy
        self.class_predictions = tf.to_int32(tf.argmax(class_probs, axis=1))
        # class_predictions_filtered = tf.to_int32(tf.argmax(class_probs_filtered, axis=1))
        # correct_pred_filtered = tf.equal(class_predictions_filtered, class_idx_filtered)

        correct_predictions = tf.equal(self.class_predictions, class_idx)

        # NOTE: This does not take into account missing classes
        self.accuracy = tf.reduce_mean(tf.to_float(correct_predictions), name='class-accuracy')


        if self.create_summary:
            accuracy_summary = tf.summary.scalar(self.accuracy.name, self.accuracy)
            loss_summary = tf.summary.scalar(self.mean_loss.name, self.mean_loss)
            self.train_summaries.append(accuracy_summary)
            self.train_summaries.append(loss_summary)
            self.val_summaries.append(accuracy_summary)
            self.val_summaries.append(loss_summary)

        # class_is_known = tf.greater_equal(self.class_idx, 0)

        # # Only compute class loss for observations where the question class
        # # is known
        # class_indices         = tf.where(class_is_known)
        # class_logits_filtered = tf.gather(class_logits, class_indices)
        # class_idx_filtered    = tf.gather(class_idx,    class_indices)

        # # Compute classification loss
        # class_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=class_logits_filtered,
        #     labels=class_idx_filtered,
        #     name='cross-entropy-class-losses'
        # )
        # self.losses = class_losses

        # # Compute mean loss
        # self.mean_loss = tf.reduce_mean(self.losses)

        # # Compute class probabilities
        # class_probs = tf.nn.softmax(class_logits)
        # class_probs_filtered = tf.nn.softmax(class_logits_filtered)

        # # Compute accuracy
        # self.class_predictions = tf.to_int32(tf.argmax(class_probs, axis=1))
        # class_predictions_filtered = tf.to_int32(tf.argmax(class_probs_filtered, axis=1))
        # correct_pred_filtered = tf.equal(class_predictions_filtered, class_idx_filtered)
        # self.accuracy = tf.reduce_mean(tf.to_float(correct_pred_filtered), name='class-accuracy')


    def setup_optimizer(self):
        # Get trainable variables
        params = tf.trainable_variables()

        # Define optimizer algorithm
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Get and clip gradients for class prediction
        self.gradients = tf.gradients(self.losses, params)

        clipped_gradients, norm_class = tf.clip_by_global_norm(
            t_list=self.gradients,
            clip_norm=self.config.max_gradient_norm
        )

        if self.create_summary:
            for variable in params:
                self.train_summaries.append(
                    tf.summary.histogram(variable.name, variable)
                )

            for gradient in self.gradients:
                if gradient is not None:
                    self.train_summaries.append(
                        tf.summary.histogram(gradient.name, gradient)
                    )

        # Define update operation
        self.update_op = optimizer.apply_gradients(
            grads_and_vars=zip(clipped_gradients, params),
            global_step=self.global_step
        )

    def build_feed_dict(self, validation, session, enc_input, enc_input_length,
                        **kwargs):
        assert validation in ['train', 'val', 'test']
        feed_dict = {}

        # Get global step
        global_step = self.get_global_step(session)

        # Values all operations need
        feed_dict[self.encoder_input_chars]     = enc_input
        feed_dict[self.encoder_sequence_length] = enc_input_length

        # Keep probability
        if validation == 'train':
            feed_dict[self.keep_prob_ph] = self.config.keep_prob
        else:
            feed_dict[self.keep_prob_ph] = 1.0

        # If class vector is provided
        feed_dict[self.class_idx] = kwargs['class_idx']

        # If using char-2-word encoder
        if 'ws_indices' in kwargs and hasattr(self, 'enc_word_indices'):
            feed_dict[self.enc_word_indices] = kwargs['ws_indices']
        if 'word_count' in kwargs and hasattr(self, 'word_seq_lengths'):
            feed_dict[self.word_seq_lengths] = kwargs['word_count']

        return feed_dict

    def train_op(self, session, **kwargs):
        assert not self.prediction_mode

        _, mean_loss, accuracy, summary, global_step = session.run(
            fetches=[self.update_op, self.mean_loss, self.accuracy, self.train_summary, self.global_step],
            feed_dict=self.build_feed_dict('train', session, **kwargs)
        )

        return {
            'mean_loss':    mean_loss,
            'mean_prob_x':  0.0,
            'accuracy':     accuracy,
            'summary':      summary,
            'global_step':  global_step
        }

    def val_op(self, session, **kwargs):
        assert not self.prediction_mode

        mean_loss, accuracy, summary, global_step = session.run(
            fetches=[self.mean_loss, self.accuracy, self.val_summary, self.global_step],
            feed_dict=self.build_feed_dict('val', session, **kwargs)
        )

        return {
            'mean_loss':    mean_loss,
            'mean_prob_x':  0.0,
            'accuracy':     accuracy,
            'summary':      summary,
            'global_step':  global_step
        }

    def predict(self, session, **kwargs):
        assert self.prediction_mode

        mean_loss, class_predictions = session.run(
            fetches=[self.mean_loss, self.class_predictions],
            feed_dict=self.build_feed_dict('test', session, **kwargs)
        )

        return {
            'mean_loss':            mean_loss,
            'class_predictions':    class_predictions
        }



class RNN_BLSTM_Classifier_Regular(RNN_BLSTM_Classifier):
    def __init__(self, **kwargs):
        RNN.__init__(self, include_accuracy=True, **kwargs)

        # Define encoder
        with tf.variable_scope('encoder'):
            cell_encoder = rnn_cell(
                num_units=self.config.num_units,
                num_cells=self.config.num_cells,
                keep_prob=self.keep_prob_ph,
                add_dropout=self.add_dropout
            )

            self.encoder = BidirectionalEncoder(
                cell=cell_encoder,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )


    def setup_network(self):
        # Setup character embedding
        embedded_encoder_input, embedded_decoder_input, embed_func = self.setup_character_embedding()

        # Output projection
        with tf.variable_scope('alphabet_projection') as scope:
            self.projection_W, self.projection_b = intialize_projections(
                input_size=4 * self.config.num_units, # Because we use bidirectional encoder
                output_size=self.config.alphabet_size,
                scope=scope
            )

        # Encoder
        with tf.variable_scope('encoder') as scope:
            # Normalize batch
            embedded_encoder_input = tf.layers.batch_normalization(
                inputs=embedded_encoder_input,
                center=True,
                scale=True,
                training=not self.prediction_mode,
                trainable=True,
            )

            enc_outputs, enc_final_state = self.encoder.encode(
                inputs=embedded_encoder_input,
                seq_lengths=self.encoder_sequence_length,
                scope=scope
            )

        # Predict question categories
        with tf.variable_scope('question') as scope:
            # Convert StateTuple to vector
            state_vector = tf.concat(flatten(enc_final_state), axis=1, name='combined-state-vec')

            # Add dense layer
            layer = projection(
                x=state_vector,
                input_size=4 * self.config.num_units * self.config.num_cells,
                output_size=128,
                nonlinearity=tf.nn.relu
            )
            if self.add_dropout:
                layer = tf.nn.dropout(
                    x=layer,
                    keep_prob=self.keep_prob_ph
                )

            class_logits = projection(
                x=layer,
                input_size=128,
                output_size=self.config.num_classes
            )

        W_penalty = 0.0

        # Define loss
        self.setup_losses(
            class_logits=class_logits,
            class_idx=self.class_idx,
            W_penalty=W_penalty
        )

