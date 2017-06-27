
from __future__ import division

import collections

import six
import tensorflow as tf
import numpy as np

def state_cell_count(cell):
    return state_size(cell.state_size)

def state_size(state):
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        return 1
    else:
        return len(state)

def _yield_flat_nest(nest):
    for n in nest:
        if is_sequence(n):
            for ni in _yield_flat_nest(n):
                yield ni
        else:
            yield n

def is_sequence(seq):
    """Returns a true if its input is a collections.Sequence (except strings).
    Args:
    seq: an input sequence.
    Returns:
    True if the sequence is a not a string and is a collections.Sequence.
    """
    return (isinstance(seq, collections.Sequence)
            and not isinstance(seq, six.string_types))

def flatten(nest):
    """Returns a flat sequence from a given nested structure.
    If `nest` is not a sequence, this returns a single-element list: `[nest]`.
    Args:
    nest: an arbitrarily nested structure or a scalar object.
      Note, numpy arrays are considered scalars.
    Returns:
    A Python list, the flattened version of the input.
    """
    return list(_yield_flat_nest(nest)) if is_sequence(nest) else [nest]

def intialize_projections(input_size, output_size, scope=None):
    """
        Returns the tuple `(W, b)` with `W` being a matrix of shape
        `(input_size, output_size)` and `b` being a vector of shape `(output_size)`.
    """
    with tf.variable_scope(name_or_scope=scope, default_name='rnn_cell') as scope:
        W = tf.get_variable(
            shape=[input_size, output_size],
            initializer=tf.uniform_unit_scaling_initializer(),
            name='W'
        )
        b = tf.get_variable(
            shape=[output_size],
            initializer=tf.constant_initializer(0.0),
            name='b',
        )
        return (W, b)

def projection(x, input_size=None, output_size=None, W=None, b=None,
               nonlinearity=None, name=None, scope=None):
    """
        Returns the projection `Wx + b`. If `W=None` or `b=None` they will
        be created with the shape `(input_size, output_size)` and `(output_size)`.
        If `nonlinearity` is not None the function will return
        `nonlinearity(Wx + b)`.
    """
    assert ((input_size is None or output_size is None) and (W is not None and b is not None)) or \
           ((input_size is not None and output_size is not None) and (W is None or b is None))

    with tf.variable_scope(name_or_scope=scope, default_name='projection') as scope:
        if W is None or b is None:
            W, b = intialize_projections(input_size, output_size, scope)

        # Compute projection
        result = tf.matmul(x, W) + b
        if nonlinearity is not None:
            result = nonlinearity(result)

    if name is not None:
        result = tf.identity(result, name=name)

    return result

def np_softmax(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


def compute_decayed_logits(logits, decay_parameter_ph):
    """
        Computes the decayed logits from the log-probabilities `logits`
        with the decay parameter being `decay_parameter_ph`.
    """
    logits_decayed = tf.cast(logits, tf.float64) * (1.0 + decay_parameter_ph)
    logits_decayed = tf.identity(logits_decayed, name='decayed_logits')
    return logits_decayed

def compute_decayed_probs(logits, decay_parameter_ph):
    """
        Computes the decayed probabilities from the log-probabilities `logits`
        with the decay parameter being `decay_parameter_ph`.
    """
    logits_decayed = compute_decayed_logits(logits, decay_parameter_ph)
    probs_decayed = tf.nn.softmax(logits_decayed, name='decayed_probabilities')
    return probs_decayed

def pack_state_tuple(state_tuple, name=None):
    """
        Converts a StateTuple to a tensor of shape
        [batch_size, 2, num_cells, num_units]
        with the 3rd dimension containing the c and h vectors of the given
        cell.
    """
    if isinstance(state_tuple, tf.contrib.rnn.LSTMStateTuple):
        state_tuple = (state_tuple,)
    state_c = tf.concat([tf.expand_dims(state.c, axis=1) for state in state_tuple], axis=1)
    state_h = tf.concat([tf.expand_dims(state.h, axis=1) for state in state_tuple], axis=1)
    state = tf.concat([tf.expand_dims(state_c, axis=1), tf.expand_dims(state_h, axis=1)], axis=1)
    if name is not None:
        state = tf.identity(state, name=name)
    return state


def unpack_state_tensor(state_tensor):
    """
        Converts a state tensor of shape [batch_size, 2, num_cells, num_units]
        to a tuple of StateTuples for using as initial state in rnn cell.
    """
    states = tf.unstack(state_tensor, axis=2)
    state_tuples = []
    for i, state in enumerate(states):
        state_c, state_h = tf.unstack(state, axis=1)
        state_c = tf.identity(state_c, name='c-%d' % (i))
        state_h = tf.identity(state_h, name='h-%d' % (i))
        state_tuples.append(tf.contrib.rnn.LSTMStateTuple(c=state_c, h=state_h))
    packed_state = tuple(state_tuples)
    if len(states) == 1:
        packed_state = packed_state[0]
    return packed_state
