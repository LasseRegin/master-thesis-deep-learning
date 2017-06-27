
import tensorflow as tf

from .ops import state_cell_count

class Encoder:
    def __init__(self, cell, dtype=tf.float32, parallel_iterations=32,
                 swap_memory=False):
        self.cell = cell
        self.dtype = dtype
        self.parallel_iterations = parallel_iterations
        self.swap_memory = swap_memory

        # Infer number of cells
        self.num_cells = state_cell_count(cell)
        self.num_units = cell.output_size

    def encode(self, inputs, seq_lengths, scope=None):
        outputs, final_state = tf.nn.dynamic_rnn(
            cell=self.cell,
            inputs=inputs,
            sequence_length=seq_lengths,
            dtype=self.dtype,
            scope=scope,
            parallel_iterations=self.parallel_iterations,
            swap_memory=self.swap_memory
        )

        return outputs, final_state


class BidirectionalEncoder(Encoder):
    def encode(self, inputs, seq_lengths, scope=None):
        # Forward direction
        with tf.variable_scope('FW') as fw_scope:
            output_fw, final_state_fw = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=inputs,
                sequence_length=seq_lengths,
                dtype=self.dtype,
                scope=fw_scope,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )

        # Backward direction
        time_dim = 1
        batch_dim = 0

        with tf.variable_scope('BW') as bw_scope:
            # Reverse the input
            inputs_reverse = tf.reverse_sequence(
                input=inputs,
                seq_lengths=seq_lengths,
                seq_dim=time_dim,
                batch_dim=batch_dim
            )

            output_bw, final_state_bw = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=inputs_reverse,
                sequence_length=seq_lengths,
                dtype=self.dtype,
                scope=bw_scope,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )

        # Reverse the backwards output to get the correct order again
        output_bw = tf.reverse_sequence(
            input=output_bw,
            seq_lengths=seq_lengths,
            seq_dim=time_dim,
            batch_dim=batch_dim
        )

        # Concatenate forward/backward outputs
        outputs = tf.concat([output_fw, output_bw], axis=2, name='outputs')

        if self.num_cells == 1:
            final_state_fw = (final_state_fw,)
            final_state_bw = (final_state_bw,)

        # Concatenate forward/backward state
        states = []
        for i in range(0, self.num_cells):
            c = tf.concat([final_state_fw[i].c, final_state_bw[i].c], axis=1, name='c_%d' % (i))
            h = tf.concat([final_state_fw[i].h, final_state_bw[i].h], axis=1, name='h_%d' % (i))
            states.append(tf.contrib.rnn.LSTMStateTuple(c=c, h=h))

        if self.num_cells == 1:
            final_state = states[0]
        else:
            final_state = tuple(states)

        return outputs, final_state


class Char2WordEncoder(Encoder):
    def encode(self, inputs, seq_lengths, word_seq_lengths, enc_word_indices, max_words, scope=None):
        # Character encoding
        with tf.variable_scope('char-encoding') as char_scope:
            output_char, final_state_char = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=inputs,
                sequence_length=seq_lengths,
                dtype=self.dtype,
                scope=char_scope,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )

        # Backward direction
        time_dim = 1
        batch_dim = 0

        # Word encoding
        with tf.variable_scope('word-encoding') as word_scope:
            def _grid_gather(params, indices):
                with tf.variable_scope('grid-gather'):
                    params_shape = tf.shape(params)

                    # Reshape params
                    flat_params_dim0 = tf.reduce_prod(params_shape[:2])
                    flat_params = tf.reshape(params, [-1, self.num_units])

                    # Fix indices
                    rng = tf.expand_dims(tf.range(flat_params_dim0, delta=params_shape[1]), 1)
                    ones_shape_list = [
                        tf.expand_dims(tf.constant(1), axis=0),
                        tf.expand_dims(max_words,      axis=0)
                    ]
                    ones_shape = tf.concat(ones_shape_list, axis=0)
                    ones = tf.ones(ones_shape, dtype=tf.int32)
                    rng_array = tf.matmul(rng, ones)
                    indices = tf.to_int32(indices) + tf.to_int32(rng_array)

                    # Gather and return
                    return tf.gather(flat_params, indices)

            # Extract word states
            word_states = _grid_gather(output_char, enc_word_indices)

        # Word encoder
        with tf.variable_scope('word-encoding-fw') as word_scope:
            output_word_fw, final_state_word_fw = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=word_states,
                sequence_length=word_seq_lengths,
                dtype=self.dtype,
                scope=word_scope,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )

        with tf.variable_scope('word-encoding-bw') as word_scope:
            # Reverse the input
            word_states_reverse = tf.reverse_sequence(
                input=word_states,
                seq_lengths=word_seq_lengths,
                seq_dim=time_dim,
                batch_dim=batch_dim
            )

            output_word_bw, final_state_word_bw = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=word_states_reverse,
                sequence_length=word_seq_lengths,
                dtype=self.dtype,
                scope=word_scope,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )

        # Reverse the backwards output to get the correct order again
        output_word_bw = tf.reverse_sequence(
            input=output_word_bw,
            seq_lengths=word_seq_lengths,
            seq_dim=time_dim,
            batch_dim=batch_dim
        )

        # Concatenate forward/backward outputs
        outputs = tf.concat([output_word_fw, output_word_bw], axis=2, name='outputs')

        # Concatenate forward/backward state
        states = []
        if self.num_cells == 1:
            final_state_word_fw = (final_state_word_fw,)
            final_state_word_bw = (final_state_word_bw,)

        for i in range(0, self.num_cells):
            c = tf.concat([final_state_word_fw[i].c, final_state_word_bw[i].c], axis=1, name='c_%d' % (i))
            h = tf.concat([final_state_word_fw[i].h, final_state_word_bw[i].h], axis=1, name='h_%d' % (i))
            states.append(tf.contrib.rnn.LSTMStateTuple(c=c, h=h))

        if self.num_cells == 1:
            final_state = states[0]
        else:
            final_state = tuple(states)

        #return outputs, final_state, word_states
        return outputs, final_state
