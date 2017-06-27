
import tensorflow as tf

from .ops import state_cell_count, is_sequence, flatten, projection
from .ops import pack_state_tuple, unpack_state_tensor
from .exceptions import NotInitializedError



class Decoder:
    def __init__(self, cell, max_iterations, infer_token_prob, alphabet,
                 use_scheduled_sampling=False, check_seq_lengths=True,
                 parallel_iterations=32, swap_memory=False):
        self.cell                   = cell
        self.max_iterations         = max_iterations
        self.infer_token_prob       = infer_token_prob
        self.alphabet               = alphabet
        self.parallel_iterations    = parallel_iterations
        self.swap_memory            = swap_memory
        self.use_scheduled_sampling = use_scheduled_sampling
        self.check_seq_lengths      = check_seq_lengths

        # Infer number of cells
        self.num_cells  = state_cell_count(cell)
        self.state_size = cell.output_size

    def scheduled_sampling_input(self, time, dec_input, prev_logits, embed_func):
        # Create logits from the probability of infering token (from scheduled
        # sample paper 1506.03099v3.pdf)
        # batch_size = dec_input.get_shape().as_list()[0]
        batch_size = tf.shape(dec_input)[0]

        shape = tf.shape(dec_input)[0:1]
        use_pred_prob = tf.fill(shape, value=self.infer_token_prob)
        use_true_prob = tf.fill(shape, value=1 - self.infer_token_prob)
        use_pred_prob = tf.expand_dims(use_pred_prob, axis=1)
        use_true_prob = tf.expand_dims(use_true_prob, axis=1)
        probs = tf.concat([use_pred_prob, use_true_prob], axis=1)

        # Flip a coin for each observation. The outcomes are:
        # 0 => use last predicted value
        # 1 => use true token
        infer_token_outcome = tf.multinomial(logits=tf.log(probs), num_samples=1)
        infer_token_outcome = tf.reshape(infer_token_outcome, [batch_size])
        infer_token = tf.equal(infer_token_outcome, 0)

        # Determine what to use as input
        infered_tokens_sampled = tf.multinomial(logits=prev_logits, num_samples=1)
        infered_tokens_sampled = tf.reshape(infered_tokens_sampled, [batch_size])

        # Look up embedding etc.
        infered_input = embed_func(input_chars=infered_tokens_sampled)

        next_input = tf.where(
            condition=infer_token,
            x=infered_input,
            y=dec_input
        )

        return next_input

    def _pass_through(self, time, output, zero_output, prev_state, new_state, seq_length):
        # If t >= length of the given sequence just pass zeros as the new output
        # and the previous state as state
        sequence_done = tf.greater_equal(time, seq_length)
        output = tf.where(
            condition=sequence_done,
            x=zero_output,  # If condition is true
            y=output        # Else
        )
        state = tf.where(
            condition=sequence_done,
            x=prev_state,   # If condition is true
            y=new_state     # Else
        )

        return output, state

    def decode(self, inputs, initial_state, seq_length, embed_func, project_func):
        """ Computes decoder outputs in parallel for training.

        Args:
            input: Decoder input of shape [batch_size, max_seq_length, embedding_size]
            initial_state: Initial cell state tuple of length `num_cells`

        Returns:
            outputs: Outputs of decoder
            final_state: Final state of decoder
        """
        # Make time the first dimension
        inputs = tf.transpose(inputs, [1, 0, 2])

        # Pack state tuple to tensor
        initial_state = pack_state_tuple(initial_state)

        # Define decoder input TensorArray
        inputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True
        )
        inputs_ta = inputs_ta.unstack(inputs)

        # Define decoder output TensorArray
        outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=self.max_iterations,
            dynamic_size=False
        )

        # Determine min and max lengths
        min_seq_length = tf.reduce_min(seq_length)
        max_seq_length = tf.reduce_max(seq_length)

        def cond(time, prev_state, outputs_ta, prev_logits):
            return tf.less(time, self.max_iterations)

        def body(time, prev_state, outputs_ta, prev_logits):
            # Read input
            dec_input = inputs_ta.read(time)

            if self.use_scheduled_sampling:
                # If t > 0 use scheduled sampling
                dec_input = tf.cond(
                    pred=time > 0,
                    fn1=lambda: self.scheduled_sampling_input(time, dec_input, prev_logits, embed_func),
                    fn2=lambda: dec_input
                )

            # Compute time cell value
            output, new_state = self.predict(
                inputs=dec_input,
                initial_state=unpack_state_tensor(prev_state)
            )
            new_state = pack_state_tuple(new_state)

            if self.check_seq_lengths:
                # Create zero output
                zero_output = tf.zeros(shape=[tf.shape(inputs)[1], self.state_size])

                # If t >= max_seq_length just pass zeros as the new output and
                # previous state as the new state.
                output, new_state = tf.cond(
                    pred=time >= max_seq_length,
                    fn1=lambda: (zero_output, new_state),
                    fn2=lambda: self._pass_through(time, output, zero_output, prev_state, new_state, seq_length)
                )

            if self.use_scheduled_sampling:
                # Project to alphabet_size
                logits = project_func(output)
            else:
                logits = prev_logits

            # Write output
            outputs_ta = outputs_ta.write(time, output, name='dec-output')

            return (time + 1, new_state, outputs_ta, logits)

        # Create empty output tensor
        prev_logits = tf.zeros(shape=[tf.shape(inputs)[1], len(self.alphabet)])

        # Define initial loop variables
        loop_vars = (
            tf.constant(0, dtype=tf.int32),
            initial_state,
            outputs_ta,
            prev_logits
        )

        # Execute loop
        _, final_state, outputs_ta, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=loop_vars,
            parallel_iterations=self.parallel_iterations,
            swap_memory=self.swap_memory
        )

        # Convert TensorArray of outputs to Tensor
        final_outputs = outputs_ta.stack()
        final_outputs = tf.transpose(final_outputs, [1, 0, 2])
        final_outputs = tf.reshape(final_outputs, [-1, self.max_iterations, self.state_size],
                                   name='outputs')

        # Unpack state tensor to state tuple
        final_state = unpack_state_tensor(final_state)

        return final_outputs, final_state


    def predict(self, inputs, initial_state):
        """ Predicts a single time sequence from given inputs and initial states.

        Args:
            input: Decoder input of shape [batch_size, embedding_size]
            initial_state: Initial cell state tuple of length `num_cells`

        Returns:
            outputs: Outputs of decoder
            final_state: Final state of decoder
        """

        # Compute time cell value
        output, state = self.cell(inputs, initial_state)

        return output, state


class AttentionDecoder(Decoder):
    """
        Implements decoder using the attention mechanism introduced in
        https://arxiv.org/abs/1412.7449
    """
    def __init__(self, initial_state_attention=False, **kwargs):
        self.initial_state_attention = initial_state_attention

        super().__init__(**kwargs)

    def initialize_attention_func(self, input_size, attention_states):
        # Get shape of attention states (the outputs from the encoder cell)
        attention_states_shape = attention_states.get_shape().as_list()
        attention_size = attention_states_shape[-1]
        attention_length = attention_states_shape[1]

        # Define W_2
        with tf.variable_scope('attention'):
            # Since we unroll the cell state tuples we will have two vectors
            # for each rnn cell (the hidden state vector c_t and the output
            # vector h_t)
            unrolled_state_length = 2 * self.state_size * self.num_cells

            W_2 = tf.get_variable(
                name='W_2',
                shape=[unrolled_state_length, attention_size],
                initializer=tf.uniform_unit_scaling_initializer(),
                dtype=tf.float32
            )

            b_2 = tf.get_variable(
                name='b_2',
                shape=[attention_size],
                initializer=tf.constant_initializer(),
                dtype=tf.float32
            )

            W_3 = tf.get_variable(
                name='W_3',
                shape=[input_size + attention_size, input_size],
                initializer=tf.uniform_unit_scaling_initializer(),
                dtype=tf.float32
            )

            b_3 = tf.get_variable(
                name='b_3',
                shape=[input_size],
                initializer=tf.constant_initializer(),
                dtype=tf.float32
            )

        # Reshape hidden encoder state `h_t`.
        h_t = tf.reshape(attention_states, shape=[-1, attention_length, 1, attention_size])

        k = tf.get_variable(
            shape=[1, 1, attention_size, attention_size],
            name='attention_W'
        )

        v = tf.get_variable(
            shape=[attention_size],
            name='attention_v'
        )

        # Compute W_1 * h_t using a 1-by-1 convolution
        W1_ht = tf.nn.conv2d(
            input=h_t,
            filter=k,
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='W1_ht'
        )

        # Define attention function
        def attention_func(state):
            '''
                Computes attention-weighted context vector c_t from a given
                RNN StateTuple.
            '''
            # If the query is a tuple, flatten it
            # (e.g. when using bidirectional encoder).
            if is_sequence(state):
                query_list = flatten(state)
                state = tf.concat(query_list, axis=1)

            with tf.variable_scope('attention'):
                # Compute W_2 * d_t
                W2_dt = projection(state, W=W_2, b=b_2)
                W2_dt = tf.reshape(W2_dt, [-1, 1, 1, attention_size])

                # Compute attention mask:
                #   v.T * tanh(W_1 * h_t + W_2 * d_t)
                u = tf.reduce_sum(v * tf.tanh(W1_ht + W2_dt), [2, 3])

                # Compute attention mask - alphas
                alpha = tf.nn.softmax(u, name='alpha-weights')

                # Compute the attention-weighted context vector c_t.
                c_t = tf.reduce_sum(tf.reshape(alpha, [-1, attention_length, 1, 1]) * h_t, [1, 2])

            return c_t
        self._attention_func = attention_func
        self._W_3 = W_3
        self._b_3 = b_3

    def attention_func(self, *args, **kwargs):
        if self._attention_func is None:
            raise NotInitializedError('Attention function should be initialized \
                                       before using!')
        return self._attention_func(*args, **kwargs)

    @property
    def W_3(self):
        if self._W_3 is None:
            raise NotInitializedError('Attention function should be initialized \
                                       before using!')
        return self._W_3

    @property
    def b_3(self):
        if self._b_3 is None:
            raise NotInitializedError('Attention function should be initialized \
                                       before using!')
        return self._b_3

    def decode(self, inputs, initial_state, seq_length, embed_func, project_func,
               additional_state_units=0):
        """ Computes decoder outputs in parallel for training.

        Args:
            input: Decoder input of shape [batch_size, max_seq_length, embedding_size]
            initial_state: Initial cell state tuple of length `num_cells`

        Returns:
            outputs: Outputs of decoder
            final_state: Final state of decoder
        """
        batch_size = tf.shape(inputs)[0]
        attention_size = self.state_size - additional_state_units

        # Initialize the attentions from the initial state and attention
        # states if `initial_state_attention` is True
        # Ueful when we wish to resume decoding from a previously stored
        # decoder state and attention states.
        if self.initial_state_attention:
            attentions = self.attention_func(initial_state)
        else:
            attentions = tf.zeros([batch_size, attention_size], dtype=tf.float32)

        # Pack state tuple to tensor
        initial_state = pack_state_tuple(initial_state)

        # Make time the first dimension
        inputs = tf.transpose(inputs, [1, 0, 2])

        # Define decoder input TensorArray
        inputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True
        )
        inputs_ta = inputs_ta.unstack(inputs)

        # Define decoder output TensorArray
        outputs_ta = tf.TensorArray(
            dtype=tf.float32,
            size=self.max_iterations,
            dynamic_size=False
        )

        # Determine min and max lengths
        min_seq_length = tf.reduce_min(seq_length)
        max_seq_length = tf.reduce_max(seq_length)

        def cond(time, prev_state, outputs_ta, attentions, prev_logits):
            return tf.less(time, self.max_iterations)

        def body(time, prev_state, outputs_ta, attentions, prev_logits):
            # Read input
            dec_input = inputs_ta.read(time)

            if self.use_scheduled_sampling:
                # If t > 0 use scheduled sampling
                dec_input = tf.cond(
                    pred=time > 0,
                    fn1=lambda: self.scheduled_sampling_input(time, dec_input, prev_logits, embed_func),
                    fn2=lambda: dec_input
                )

            # Predict
            output_merged, new_state, attentions = self.predict(
                inputs=dec_input,
                initial_state=unpack_state_tensor(prev_state),
                attention_states=attentions
            )
            new_state = pack_state_tuple(new_state)

            if self.check_seq_lengths:
                # Create zero output
                zero_output = tf.zeros(shape=[tf.shape(inputs)[1], self.state_size + attention_size])

                # If t >= max_seq_length just pass zeros as the new output and
                # previous state as the new state.
                output_merged, new_state = tf.cond(
                    pred=time >= max_seq_length,
                    fn1=lambda: (zero_output, new_state),
                    fn2=lambda: self._pass_through(time, output_merged, zero_output, prev_state, new_state, seq_length)
                )

            if self.use_scheduled_sampling:
                # Project to alphabet_size
                logits = project_func(output_merged)
            else:
                logits = prev_logits

            # Write output
            outputs_ta = outputs_ta.write(time, output_merged, name='dec-output')

            return (time + 1, new_state, outputs_ta, attentions, logits)

        # Create empty output tensor
        prev_logits = tf.zeros(shape=[tf.shape(inputs)[1], len(self.alphabet)])

        # Define initial loop variables
        loop_vars = (
            tf.constant(0, dtype=tf.int32),
            initial_state,
            outputs_ta,
            attentions,
            prev_logits
        )

        # Execute loop
        _, final_state, outputs_ta, _, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=loop_vars,
            parallel_iterations=self.parallel_iterations,
            swap_memory=self.swap_memory
        )

        # Convert TensorArray of outputs to Tensor
        final_outputs = outputs_ta.stack()
        final_outputs = tf.transpose(final_outputs, [1, 0, 2]) # TODO: Is this right?
        final_outputs = tf.reshape(final_outputs, [-1, self.max_iterations, self.state_size + attention_size],
                                   name='outputs')

        # Unpack state tensor to state tuple
        final_state = unpack_state_tensor(final_state)

        return final_outputs, final_state


    def predict(self, inputs, initial_state, attention_states):
        """ Predicts a single time sequence from given inputs and initial states.

        Args:
            input: Decoder input of shape [batch_size, embedding_size]
            initial_state: Initial cell state tuple of length `num_cells`

        Returns:
            outputs: Outputs of decoder
            final_state: Final state of decoder
        """
        # Concatenate attention masked states with decoder input and project
        # down to the decoder input size.
        inputs_merged = tf.concat([inputs, attention_states], axis=1)
        inputs = projection(inputs_merged, W=self.W_3, b=self.b_3)

        # Compute time cell value
        output, state = self.cell(inputs, initial_state)

        # Apply attention
        attentions = self.attention_func(state)

        # Merge decoder output and attention vector
        output_merged = tf.concat([output, attentions], axis=1)

        return output_merged, state, attentions
