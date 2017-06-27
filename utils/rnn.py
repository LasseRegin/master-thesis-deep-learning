
import tensorflow as tf

from .ops import state_size

def rnn_cell(num_units, num_cells, keep_prob, add_dropout=False, scope=None):
    """
        Returns `num_cells` RNN cells with `num_units` units stacked on top of
        eachother. If `add_dropout=True` the cells will be wrapped with dropout
        and the value of `keep_prob` will be the %% to keep in input and output.
        `keep_prob` can be a `tensor` or `float`.
    """
    with tf.variable_scope(name_or_scope=scope, default_name='rnn_cell') as scope:
        # Create cell
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)

        # Add dropout if input and output keep
        if add_dropout:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=keep_prob,
                output_keep_prob=keep_prob
            )

        # Stack cells
        if num_cells > 1:
            cell = tf.contrib.rnn.MultiRNNCell(cells=[cell] * num_cells, state_is_tuple=True)

        return cell


def sequence_losses(logits, labels, seq_lengths=None):
    """
        Computes the cross-entropy losses for given sequence logits with
        corresponding target labels.

    Args:
        logits: Tensor of logits for the sequence classes of shape
                [batch_size, max_seq_length, alphabet_size]
        labels: Tensor of target labels of shape
                [batch_size, max_seq_length]
        seq_lengths (optional): Tensor of sequence lengths of target sequences
                used for masking the sequence losses of shape [batch_size].
    """
    # Compute losses
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        name='cross_entropy_losses'
    )

    if seq_lengths is not None:
        # Infer max sequence length
        max_seq_length = labels.get_shape().as_list()[-1]

        # Construct mask for targets
        mask = tf.sequence_mask(
            lengths=seq_lengths,
            dtype=tf.float32,
            maxlen=max_seq_length,
            name='loss_mask'
        )

        # Do not use loss for padded symbols
        losses = tf.identity(losses * mask, name='masked_cross_entropy_losses')

    return losses


def sequence_probability(logits, labels):
    """
        Computes the sequence losses for given sequence logits with
        corresponding target labels.

    Args:
        logits: Tensor of logits for the sequence classes of shape
                [batch_size, max_seq_length, alphabet_size]
        labels: Tensor of target labels of shape
                [batch_size, max_seq_length]
    """
    # Compute probabilities
    prob = tf.nn.softmax(logits)

    # Infer alphabet_size
    logits_shape_list = logits.get_shape().as_list()
    alphabet_size = logits_shape_list[-1]

    # Create mask for only extracting the probability of the correct
    # target label
    mask = tf.one_hot(indices=labels, depth=alphabet_size, on_value=1.0, off_value=0.0)
    prob_masked = prob * mask
    prob_x_t = tf.reduce_sum(prob_masked, axis=2)

    # Compute sequence probablity as the mean of probabilities
    # This differs from the definition in Graves' paper
    # https://arxiv.org/pdf/1308.0850.pdf
    # but makes probabilities of sequences with different length more
    # compareable.
    mean_prob_x_batch = tf.reduce_mean(prob_x_t, axis=1, name='prob_x')

    return mean_prob_x_batch


def select_decoder_inputs(is_lm_mode, enc_outputs, initial_state, scope=None):
    # Infer state cell count
    num_cells = state_size(initial_state)

    with tf.variable_scope(scope or 'decoder_inputs_selection') as scope:
        # Set decoder initial state based on the binary input value
        # - If `self.is_lm_mode=0` Use the passed initial state from encoder
        # - If `self.is_lm_mode=1` Use the zero vector
        is_lm_mode_cond = tf.equal(is_lm_mode, 1)

        if num_cells == 1:
            initial_state = (initial_state,)

        new_initial_states = []
        for i, state_tuple in enumerate(initial_state):

            # Construct zero state vector for language model decoder initial state
            zero_vec_c = tf.zeros(shape=tf.shape(state_tuple.c))
            zero_vec_h = tf.zeros(shape=tf.shape(state_tuple.h))

            c = tf.where(is_lm_mode_cond, zero_vec_c, state_tuple.c, name='c_%d' % (i))
            h = tf.where(is_lm_mode_cond, zero_vec_h, state_tuple.h, name='h_%d' % (i))

            new_initial_states.append(tf.contrib.rnn.LSTMStateTuple(c=c, h=h))

        if num_cells == 1:
            initial_state = new_initial_states[0]
        else:
            initial_state = tuple(new_initial_states)

        # Overwrite encoder outputs to zeros if we are in language model mode
        # (we don't need the values and hopefully we can prevent the computation)
        zero_vec = tf.zeros(shape=tf.shape(enc_outputs))
        enc_outputs = tf.where(is_lm_mode_cond, zero_vec, enc_outputs, name='enc_outputs')

    return enc_outputs, initial_state
