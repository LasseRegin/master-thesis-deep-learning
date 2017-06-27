
import tensorflow as tf

from utils.rnn import rnn_cell, select_decoder_inputs
from utils.encoder import BidirectionalEncoder
from utils.decoder import Decoder
from utils.ops import intialize_projections, projection
from utils.ops import compute_decayed_probs
from utils.ops import pack_state_tuple, unpack_state_tensor
from .rnn import RNN


class RNN_BLSTM(RNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        # Define decoder
        with tf.variable_scope('decoder'):
            cell_decoder = rnn_cell(
                num_units=2 * self.config.num_units, # Because we use bidirectional encoder
                num_cells=self.config.num_cells,
                keep_prob=self.keep_prob_ph,
                add_dropout=self.add_dropout
            )
            self.decoder = Decoder(
                cell=cell_decoder,
                max_iterations=self.config.max_dec_seq_length + 1,
                infer_token_prob=self.infer_token_prob,
                use_scheduled_sampling=self.config.scheduled_sampling,
                alphabet=self.alphabet,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )

        # Inputs for the one-step-at-a-time decoding
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None], name='decoder_inputs')
        self.decoder_state  = tf.placeholder(tf.float32, shape=[None, 2, self.config.num_cells, 2 * self.config.num_units], name='decoder_state')

    def setup_network(self):
        # Setup character embedding
        embedded_encoder_input, embedded_decoder_input, embed_func = self.setup_character_embedding()

        # Output projection
        with tf.variable_scope('alphabet_projection') as scope:
            self.projection_W, self.projection_b = intialize_projections(
                input_size=2 * self.config.num_units, # Because we use bidirectional encoder
                output_size=self.config.alphabet_size,
                scope=scope
            )

            # Define alphabet projection function
            def project_func(output):
                return projection(output, W=self.projection_W, b=self.projection_b)

        # Encoder
        with tf.variable_scope('encoder') as scope:
            enc_outputs, enc_final_state = self.encoder.encode(
                inputs=embedded_encoder_input,
                seq_lengths=self.encoder_sequence_length,
                scope=scope
            )

        # Set decoder initial state and encoder outputs based on the binary
        # mode input value
        # - If `self.is_lm_mode=0` Use the passed initial state from encoder
        # - If `self.is_lm_mode=1` Use the zero vector
        enc_outputs, self.enc_final_state = select_decoder_inputs(
            is_lm_mode=self.is_lm_mode,
            enc_outputs=enc_outputs,
            initial_state=enc_final_state
        )

        # Pack state to tensor
        self.enc_final_state_tensor = pack_state_tuple(self.enc_final_state)

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
            dec_outputs = tf.reshape(dec_outputs, [-1, 2 * self.config.num_units])
            dec_outputs = projection(dec_outputs, W=self.projection_W, b=self.projection_b)
            dec_outputs = tf.reshape(dec_outputs, [-1, self.config.max_dec_seq_length + 1, self.config.alphabet_size])

        if self.prediction_mode:
            dec_outputs = self.decoder_logits

        # Define loss
        self.setup_losses(
            dec_outputs=dec_outputs,
            target_chars=self.target_chars,
            decoder_sequence_length=self.decoder_sequence_length
        )

        if self.prediction_mode:
            # Look up inputs
            decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_inputs, name='decoder_input')
            is_lm_mode_tensor = tf.to_float(tf.expand_dims(self.is_lm_mode, axis=1))
            decoder_inputs = tf.concat([decoder_inputs_embedded, is_lm_mode_tensor], axis=1)

            # Unpack state
            initial_state = unpack_state_tensor(self.decoder_state)

            with tf.variable_scope('decoder', reuse=True):
                decoder_output, decoder_final_state = self.decoder.predict(
                    inputs=decoder_inputs,
                    initial_state=initial_state
                )

            # Project output to alphabet size
            self.decoder_output = projection(decoder_output, W=self.projection_W, b=self.projection_b, name='decoder_output')

            # Compute decayed logits
            self.decoder_probs_decayed = compute_decayed_probs(
                logits=self.decoder_output,
                decay_parameter_ph=self.probs_decay_parameter
            )

            # Pack state to tensor
            self.decoder_final_state = pack_state_tuple(decoder_final_state, name='decoder_final_state')
