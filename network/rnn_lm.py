
import numpy as np
import tensorflow as tf

from utils.ops import intialize_projections, projection
from utils.ops import pack_state_tuple, unpack_state_tensor
from utils.encoder import Encoder
from utils.decoder import Decoder
from utils.rnn import rnn_cell
from utils.prediction import BeamSearchPredictor, SamplingPredictor
from .rnn import RNN


class RNN_LM(RNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define decoder
        with tf.variable_scope('decoder'):
            cell_decoder = rnn_cell(
                num_units=self.config.num_units,
                num_cells=self.config.num_cells,
                keep_prob=self.keep_prob_ph,
                add_dropout=self.add_dropout
            )

            self.decoder = Decoder(
                cell=cell_decoder,
                max_iterations=self.config.max_dec_seq_length + 1,
                infer_token_prob=self.infer_token_prob,
                alphabet=self.alphabet,
                use_scheduled_sampling=False,
                check_seq_lengths=True,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )

        # Inputs for the one-step-at-a-time decoding
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None], name='decoder_inputs')
        self.decoder_state  = tf.placeholder(tf.float32, shape=[None, 2, self.config.num_cells, self.config.num_units], name='decoder_state')

    def setup_network(self):
        # Setup character embedding (defines `self.embedding_matrix`)
        with tf.device('/cpu:0'), tf.variable_scope(name_or_scope='embedding'):
            self.embedding_matrix = tf.get_variable(
                shape=[self.config.alphabet_size, self.config.embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                name='W'
            )

            # Gather slices from `params` according to `indices`
            embedded_decoder_input = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_input_chars, name='dec_input')

            def embed_func(input_chars):
                return tf.gather(self.embedding_matrix, input_chars)

        # Output projection
        with tf.variable_scope('alphabet_projection') as scope:
            self.projection_W, self.projection_b = intialize_projections(
                input_size=self.config.num_units,
                output_size=self.config.alphabet_size,
                scope=scope
            )

            # Define alphabet projection function
            def project_func(output):
                return projection(output, W=self.projection_W, b=self.projection_b)

        # Define initial state as zero states
        self.enc_final_state = self.decoder.cell.zero_state(batch_size=tf.shape(embedded_decoder_input)[0], dtype=tf.float32)

        # Define decoder
        with tf.variable_scope('decoder'):
            dec_outputs, dec_final_state = self.decoder.decode(
                inputs=embedded_decoder_input,
                initial_state=self.enc_final_state,
                seq_length=self.decoder_sequence_length,
                embed_func=embed_func,
                project_func=project_func
            )

            # Project output to alphabet size and reshape
            dec_outputs = tf.reshape(dec_outputs, [-1, self.config.num_units])
            dec_outputs = projection(dec_outputs, W=self.projection_W, b=self.projection_b)
            dec_outputs = tf.reshape(dec_outputs, [-1, self.config.max_dec_seq_length + 1, self.config.alphabet_size])

            # self.packed_dec_final_state = pack_state_tuple(dec_final_state)

        if self.prediction_mode:
            dec_outputs = self.decoder_logits

        # Define loss
        self.setup_losses(
            dec_outputs=dec_outputs,
            target_chars=self.target_chars,
            decoder_sequence_length=self.decoder_sequence_length
        )

        if self.prediction_mode:
            # Pack state to tensor
            self.enc_final_state_tensor = pack_state_tuple(self.enc_final_state)

            # Look up inputs
            decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_inputs, name='decoder_input')

            # Unpack state
            initial_state = unpack_state_tensor(self.decoder_state)

            with tf.variable_scope('decoder', reuse=True):
                decoder_output, decoder_final_state = self.decoder.predict(
                    inputs=decoder_inputs_embedded,
                    initial_state=initial_state
                )

                # Project output to alphabet size
                self.decoder_output = projection(decoder_output, W=self.projection_W, b=self.projection_b, name='decoder_output')
                self.decoder_probs  = tf.nn.softmax(self.decoder_output, name='decoder_probs')
                self.probs_decay_parameter = tf.placeholder(tf.float64, shape=(), name='probs_decay_parameter')
                self.decoder_probs_decayed = tf.pow(tf.cast(self.decoder_probs, tf.float64), self.probs_decay_parameter)
                decoder_probs_sum = tf.expand_dims(tf.reduce_sum(self.decoder_probs_decayed, axis=1), axis=1)
                decoder_probs_sum = tf.tile(decoder_probs_sum, [1, self.config.alphabet_size])
                self.decoder_probs_decayed = self.decoder_probs_decayed / decoder_probs_sum

                # Pack state to tensor
                self.decoder_final_state = pack_state_tuple(decoder_final_state, name='decoder_final_state')

    def setup_optimizer(self):
        # Get trainable variables
        params = tf.trainable_variables()

        # Define optimizer algorithm
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=0.99,
            use_locking=False,
            name='Momentum',
            use_nesterov=False
        )

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


    def train_op(self, session, enc_input, dec_input, dec_target,
                 enc_input_length, dec_input_length, is_lm_mode, **kwargs):
        assert not self.prediction_mode

        _, mean_loss, mean_prob_x, summary, global_step = session.run([self.update_op, self.mean_loss, self.mean_prob_x, self.train_summary, self.global_step], {
            self.decoder_input_chars: dec_input,
            self.target_chars: dec_target,

            self.decoder_sequence_length: dec_input_length,

            self.keep_prob_ph: self.config.keep_prob,
        })

        return {
            'mean_loss':    mean_loss,
            'mean_prob_x':  mean_prob_x,
            'summary':      summary,
            'global_step':  global_step
        }

    def val_op(self, session, enc_input, dec_input, dec_target,
               enc_input_length, dec_input_length, is_lm_mode, **kwargs):
        assert not self.prediction_mode

        mean_loss, mean_prob_x, summary, global_step = session.run([self.mean_loss, self.mean_prob_x, self.val_summary, self.global_step], {
            self.decoder_input_chars: dec_input,
            self.target_chars: dec_target,

            self.decoder_sequence_length: dec_input_length,

            self.keep_prob_ph: 1.0,
        })

        return {
            'mean_loss':    mean_loss,
            'mean_prob_x':  mean_prob_x,
            'summary':      summary,
            'global_step':  global_step
        }

    def predict(self, session, lm_predict_func=None, **kwargs):
        assert self.prediction_mode

        def decode_func(inputs, state):
            output, probs, state = session.run(
                fetches=[self.decoder_output, self.decoder_probs, self.decoder_final_state],
                feed_dict={
                    self.decoder_inputs: inputs,
                    self.decoder_state:  state
                }
            )

            if lm_predict_func is not None:
                lm_output, lm_probs, lm_state = lm_predict_func(inputs, state)

            return {
                'output':       output,
                'probs':        probs,
                'state':        state
            }

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
        initial_inputs   = np.full(shape=(batch_size,), fill_value=self.alphabet.GO_ID, dtype=np.float32)
        max_iterations   = self.config.max_dec_seq_length + 1

        # Define initial state
        initial_state = self.decoder.cell.zero_state(batch_size, dtype=tf.float32)
        initial_state = pack_state_tuple(initial_state)
        initial_state = session.run(initial_state)

        extra_features = {
        }

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

    def compute_probabilities(self, session, inputs, state):
        assert self.prediction_mode

        output, probs, state = session.run(
            fetches=[self.decoder_output, self.decoder_probs, self.decoder_final_state],
            feed_dict={
                self.decoder_inputs: inputs,
                self.decoder_state:  state
            }
        )
        return {
            'output':   output,
            'probs':    probs,
            'state':    state
        }
