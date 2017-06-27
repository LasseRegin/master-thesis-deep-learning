
import numpy as np
import tensorflow as tf

from utils.rnn import rnn_cell, sequence_losses, sequence_probability, select_decoder_inputs
from utils.encoder import BidirectionalEncoder, Char2WordEncoder
from utils.decoder import AttentionDecoder
from utils.maintainers import MixedLossMaintainer
from utils.ops import intialize_projections, projection
from utils.ops import pack_state_tuple, unpack_state_tensor
from utils.ops import compute_decayed_probs, flatten
from utils.prediction import BeamSearchPredictor, SamplingPredictor, ArgmaxPredictor
from .rnn import RNN

class RNN_BLSTM_Attention_Classifier(RNN):
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

        # Define decoder
        with tf.variable_scope('decoder'):
            cell_decoder = rnn_cell(
                num_units=2 * self.config.num_units, # Because we use bidirectional encoder
                num_cells=self.config.num_cells,
                keep_prob=self.keep_prob_ph,
                add_dropout=self.add_dropout
            )
            self.decoder = AttentionDecoder(
                cell=cell_decoder,
                max_iterations=self.config.max_dec_seq_length + 1,
                infer_token_prob=self.infer_token_prob,
                use_scheduled_sampling=self.config.scheduled_sampling,
                alphabet=self.alphabet,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )

        # Inputs for the one-step-at-a-time decoding
        self.decoder_inputs    = tf.placeholder(tf.int32,   shape=[None], name='decoder_inputs')
        self.decoder_state     = tf.placeholder(tf.float32, shape=[None, 2, self.config.num_cells, 2 * self.config.num_units], name='decoder_state')
        self.decoder_attention = tf.placeholder(tf.float32, shape=[None, 2 * self.config.num_units], name='decoder_attention')

        if not self.prediction_mode:
            self.mixed_loss_maintainer = MixedLossMaintainer(
                epochs_filename = self.config.epochs_mixed_filename,
                verbose=False
            )

    def add_epoch(self, session, **kwargs):
        step = self.get_global_step(session)
        self.mixed_loss_maintainer.add_epoch(step=step, **kwargs)
        return self.loss_maintainer.add_epoch(step=step, **kwargs)


    def setup_network(self):
        # Setup character embedding
        embedded_encoder_input, embedded_decoder_input, embed_func = self.setup_character_embedding()

        # Output projection
        with tf.variable_scope('alphabet_projection') as scope:
            self.projection_W, self.projection_b = intialize_projections(
                input_size=4 * self.config.num_units,
                output_size=self.config.alphabet_size,
                scope=scope
            )

            # Define alphabet projection function
            def project_func(output):
                return projection(output, W=self.projection_W, b=self.projection_b)

        # Encoder
        with tf.variable_scope('encoder') as scope:
            # Normalize batch
            embedded_encoder_input = tf.layers.batch_normalization(
                inputs=embedded_encoder_input,
                center=True,
                scale=True,
                # training=not self.prediction_mode,
                training=True,  # I think this should be true always, because in training
                                # and inference we have the entire question text.
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

        # Set decoder initial state and encoder outputs based on the binary
        # mode input value
        # - If `self.is_lm_mode=0` Use the passed initial state from encoder
        # - If `self.is_lm_mode=1` Use the zero vector
        self.enc_outputs, enc_final_state = select_decoder_inputs(
            is_lm_mode=self.is_lm_mode,
            enc_outputs=enc_outputs,
            initial_state=enc_final_state,
        )

        # If an observation has a class -> Pass the true class as 1-hot-encoded
        # vector to the decoder input.
        # If an observation doesn't have a class -> Pass the class logits for
        # the given observation to the decoder input.
        class_is_known = tf.greater_equal(self.class_idx, 0)

        # Create one-hot-encoded vectors
        class_one_hot = tf.one_hot(
            indices=self.class_idx,
            depth=self.config.num_classes,
            on_value=1.0,
            off_value=0.0,
            axis=-1,
            dtype=tf.float32,
            name='class-one-hot-encoded'
        )

        # Compute class probabilities
        class_probs = tf.nn.softmax(class_logits)

        # Select what to pass on
        self.class_info_vec = tf.where(
            condition=class_is_known,
            x=class_one_hot,
            y=class_probs
        )

        # Concatenate class info vector with decoder input
        _class_info_vec = tf.expand_dims(self.class_info_vec, axis=1)
        _class_info_vec = tf.tile(_class_info_vec, multiples=[1, self.config.max_dec_seq_length + 1, 1])
        decoder_input = tf.concat([embedded_decoder_input, _class_info_vec], axis=2)

        # Pack state to tensor
        self.enc_final_state_tensor = pack_state_tuple(enc_final_state)

        # Initialize decoder attention function using encoder outputs
        self.decoder.initialize_attention_func(
            input_size=decoder_input.get_shape().as_list()[-1],
            attention_states=self.enc_outputs
        )

        # Define decoder
        with tf.variable_scope('decoder'):
            dec_outputs, dec_final_state = self.decoder.decode(
                inputs=decoder_input,
                initial_state=enc_final_state,
                seq_length=self.decoder_sequence_length,
                embed_func=embed_func,
                project_func=project_func
            )

            # Project output to alphabet size and reshape
            dec_outputs = tf.reshape(dec_outputs, [-1, 4 * self.config.num_units])
            dec_outputs = projection(dec_outputs, W=self.projection_W, b=self.projection_b)
            dec_outputs = tf.reshape(dec_outputs, [-1, self.config.max_dec_seq_length + 1, self.config.alphabet_size])

        if self.prediction_mode:
            dec_outputs = self.decoder_logits

        # Define loss
        self.setup_losses(
            dec_outputs=dec_outputs,
            target_chars=self.target_chars,
            decoder_sequence_length=self.decoder_sequence_length,
            class_probs=class_probs,
            class_idx=self.class_idx,
            class_is_known=class_is_known,
            class_one_hot=class_one_hot,
            W_penalty=W_penalty
        )

        if self.prediction_mode:
            # Define initial attention tensor
            self.initial_attention = self.decoder.attention_func(enc_final_state)

            # Look up inputs
            decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_inputs, name='decoder_input')
            is_lm_mode_tensor = tf.to_float(tf.expand_dims(self.is_lm_mode, axis=1))
            decoder_inputs = tf.concat([decoder_inputs_embedded, is_lm_mode_tensor], axis=1)

            # Concatenate class info vector
            decoder_inputs = tf.concat([decoder_inputs, self.class_info_vec], axis=1)

            # Unpack state
            initial_state = unpack_state_tensor(self.decoder_state)

            with tf.variable_scope('decoder', reuse=True):
                decoder_output, decoder_final_state, self.decoder_new_attention = self.decoder.predict(
                    inputs=decoder_inputs,
                    initial_state=initial_state,
                    attention_states=self.decoder_attention
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

    def setup_losses(self, dec_outputs, target_chars, decoder_sequence_length,
                     class_probs, class_idx, class_is_known, class_one_hot,
                     W_penalty):

        # Compute cross entropy manually
        class_probs_clipped = tf.clip_by_value(
            class_probs,
            clip_value_min=1e-8,
            clip_value_max=1.0 - 1e-8
        )
        class_losses = -tf.reduce_mean(class_one_hot * tf.log(class_probs_clipped), axis=1)

        # Compute accuracy
        self.class_predictions = tf.to_int32(tf.argmax(class_probs, axis=1))
        correct_predictions = tf.equal(self.class_predictions, class_idx)
        # NOTE: This does not take into account missing classes
        self.accuracy = tf.reduce_mean(tf.to_float(correct_predictions), name='class-accuracy')

        # Compute masked sequence losses
        seq_losses = sequence_losses(
            logits=dec_outputs,
            labels=target_chars,
            seq_lengths=decoder_sequence_length
        )

        # Define mean loss
        self.mean_loss_batch = tf.reduce_mean(seq_losses, axis=1)

        # Combine sequence loss and class prediction loss
        self.losses = self.config.alpha * self.mean_loss_batch + (1 - self.config.alpha) * class_losses
        self.losses += W_penalty

        # Define mean loss
        self.mean_loss = tf.reduce_mean(self.losses, name='loss')
        # Only use loss from known indices to compute mean
        self.mean_class_loss = tf.reduce_mean(tf.gather(class_losses, tf.where(class_is_known)), name='class-loss')
        # self.mean_class_loss = tf.reduce_mean(class_losses)
        self.mean_seq_loss = tf.reduce_mean(self.mean_loss_batch, name='seq-loss')

        # Compute probabilities
        self.mean_prob_x_batch = sequence_probability(
            logits=dec_outputs,
            labels=target_chars
        )
        self.mean_prob_x = tf.reduce_mean(self.mean_prob_x_batch)

        if self.create_summary:
            for variable in [
                self.mean_loss,
                self.mean_class_loss,
                self.mean_seq_loss,
                self.accuracy
            ]:
                variable_summary = tf.summary.scalar(variable.name, variable)
                self.train_summaries.append(variable_summary)
                self.val_summaries.append(variable_summary)

    def setup_optimizer(self):
        # Get trainable variables
        params = tf.trainable_variables()

        # Define optimizer algorithm
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Get and clip gradients
        self.gradients = tf.gradients(self.losses, params)

        clipped_gradients, norm = tf.clip_by_global_norm(
            t_list=self.gradients,
            clip_norm=self.config.max_gradient_norm
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

    def train_op(self, session, **kwargs):
        assert not self.prediction_mode

        _, mean_loss, mean_prob_x, accuracy, summary, global_step, mean_class_loss, mean_seq_loss = session.run(
            fetches=[self.update_op, self.mean_loss, self.mean_prob_x, self.accuracy, self.train_summary, self.global_step, self.mean_class_loss, self.mean_seq_loss],
            feed_dict=self.build_feed_dict('train', session, **kwargs)
        )

        return {
            'mean_loss':        mean_loss,
            'mean_class_loss':  mean_class_loss,
            'mean_seq_loss':    mean_seq_loss,
            'mean_prob_x':      mean_prob_x,
            'accuracy':         accuracy,
            'summary':          summary,
            'global_step':      global_step
        }

    def val_op(self, session, **kwargs):
        assert not self.prediction_mode

        mean_loss, mean_prob_x, accuracy, summary, global_step, mean_class_loss, mean_seq_loss = session.run(
            fetches=[self.mean_loss, self.mean_prob_x, self.accuracy, self.train_summary, self.global_step, self.mean_class_loss, self.mean_seq_loss],
            feed_dict=self.build_feed_dict('val', session, **kwargs)
        )

        return {
            'mean_loss':        mean_loss,
            'mean_class_loss':  mean_class_loss,
            'mean_seq_loss':    mean_seq_loss,
            'mean_prob_x':      mean_prob_x,
            'accuracy':         accuracy,
            'summary':          summary,
            'global_step':      global_step
        }

    def predict(self, session, lm_session=None, lm_predict_func=None, **kwargs):
        assert self.prediction_mode

        # Encode sequences and extract final states
        initial_state, initial_attention, enc_outputs, class_predictions, class_info_vec = session.run(
            fetches=[self.enc_final_state_tensor, self.initial_attention, self.enc_outputs, self.class_predictions, self.class_info_vec],
            feed_dict=self.build_feed_dict('test', session, **kwargs)
        )

        def decode_func(inputs, state, is_lm_mode, attention, enc_outputs, lm_state=None, probs_decay_parameter=None):
            output, state, attention, probs_decayed = session.run(
                fetches=[self.decoder_output, self.decoder_final_state, self.decoder_new_attention, self.decoder_probs_decayed],
                feed_dict={
                    self.decoder_inputs:        inputs,
                    self.decoder_state:         state,
                    self.decoder_attention:     attention,
                    self.is_lm_mode:            is_lm_mode,
                    self.enc_outputs:           enc_outputs,
                    self.probs_decay_parameter: probs_decay_parameter,
                    self.class_info_vec:        class_info_vec
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
                'attention':        attention,
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
        initial_inputs   = np.full(shape=(batch_size,), fill_value=self.alphabet.GO_ID, dtype=np.float32)
        max_iterations   = self.config.max_dec_seq_length + 1

        # Define extra features to pass on
        extra_features = {
            'is_lm_mode':   kwargs['is_lm_mode'],
            'attention':    initial_attention,
            'enc_outputs':  enc_outputs
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
            'prob_x_candidates':    prob_x_candidates,
            'class_predictions':    class_predictions
        }



class RNN_BLSTM_Attention_Classifier_Regular(RNN_BLSTM_Attention_Classifier):
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

        # Define decoder
        with tf.variable_scope('decoder'):
            cell_decoder = rnn_cell(
                num_units=2 * self.config.num_units, # Because we use bidirectional encoder
                num_cells=self.config.num_cells,
                keep_prob=self.keep_prob_ph,
                add_dropout=self.add_dropout
            )
            self.decoder = AttentionDecoder(
                cell=cell_decoder,
                max_iterations=self.config.max_dec_seq_length + 1,
                infer_token_prob=self.infer_token_prob,
                use_scheduled_sampling=self.config.scheduled_sampling,
                alphabet=self.alphabet,
                parallel_iterations=self.parallel_iterations,
                swap_memory=self.swap_memory
            )

        # Inputs for the one-step-at-a-time decoding
        self.decoder_inputs    = tf.placeholder(tf.int32,   shape=[None], name='decoder_inputs')
        self.decoder_state     = tf.placeholder(tf.float32, shape=[None, 2, self.config.num_cells, 2 * self.config.num_units], name='decoder_state')
        self.decoder_attention = tf.placeholder(tf.float32, shape=[None, 2 * self.config.num_units], name='decoder_attention')

        if not self.prediction_mode:
            self.mixed_loss_maintainer = MixedLossMaintainer(
                epochs_filename = self.config.epochs_mixed_filename,
                verbose=False
            )

    def setup_network(self):
        # Setup character embedding
        embedded_encoder_input, embedded_decoder_input, embed_func = self.setup_character_embedding()

        # Output projection
        with tf.variable_scope('alphabet_projection') as scope:
            self.projection_W, self.projection_b = intialize_projections(
                input_size=4 * self.config.num_units,
                output_size=self.config.alphabet_size,
                scope=scope
            )

            # Define alphabet projection function
            def project_func(output):
                return projection(output, W=self.projection_W, b=self.projection_b)

        # Encoder
        with tf.variable_scope('encoder') as scope:
            # Normalize batch
            embedded_encoder_input = tf.layers.batch_normalization(
                inputs=embedded_encoder_input,
                center=True,
                scale=True,
                # training=not self.prediction_mode,
                training=True,  # I think this should be true always, because in training
                                # and inference we have the entire question text.
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

        # Set decoder initial state and encoder outputs based on the binary
        # mode input value
        # - If `self.is_lm_mode=0` Use the passed initial state from encoder
        # - If `self.is_lm_mode=1` Use the zero vector
        self.enc_outputs, enc_final_state = select_decoder_inputs(
            is_lm_mode=self.is_lm_mode,
            enc_outputs=enc_outputs,
            initial_state=enc_final_state,
        )

        # If an observation has a class -> Pass the true class as 1-hot-encoded
        # vector to the decoder input.
        # If an observation doesn't have a class -> Pass the class logits for
        # the given observation to the decoder input.
        class_is_known = tf.greater_equal(self.class_idx, 0)

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

        # Select what to pass on
        self.class_info_vec = tf.where(
            condition=class_is_known,
            x=class_one_hot,
            y=class_logits
        )

        # Concatenate class info vector with decoder input
        _class_info_vec = tf.expand_dims(self.class_info_vec, axis=1)
        _class_info_vec = tf.tile(_class_info_vec, multiples=[1, self.config.max_dec_seq_length + 1, 1])
        decoder_input = tf.concat([embedded_decoder_input, _class_info_vec], axis=2)

        # Pack state to tensor
        self.enc_final_state_tensor = pack_state_tuple(enc_final_state)

        # Initialize decoder attention function using encoder outputs
        self.decoder.initialize_attention_func(
            input_size=decoder_input.get_shape().as_list()[-1],
            attention_states=self.enc_outputs
        )

        # Define decoder
        with tf.variable_scope('decoder'):
            dec_outputs, dec_final_state = self.decoder.decode(
                inputs=decoder_input,
                initial_state=enc_final_state,
                seq_length=self.decoder_sequence_length,
                embed_func=embed_func,
                project_func=project_func
            )

            # Project output to alphabet size and reshape
            dec_outputs = tf.reshape(dec_outputs, [-1, 4 * self.config.num_units])
            dec_outputs = projection(dec_outputs, W=self.projection_W, b=self.projection_b)
            dec_outputs = tf.reshape(dec_outputs, [-1, self.config.max_dec_seq_length + 1, self.config.alphabet_size])

        if self.prediction_mode:
            dec_outputs = self.decoder_logits

        # Define loss
        self.setup_losses(
            dec_outputs=dec_outputs,
            target_chars=self.target_chars,
            decoder_sequence_length=self.decoder_sequence_length,
            class_logits=class_logits,
            class_idx=self.class_idx,
            class_is_known=class_is_known,
            class_one_hot=class_one_hot,
            W_penalty=W_penalty
        )

        if self.prediction_mode:
            # Define initial attention tensor
            self.initial_attention = self.decoder.attention_func(enc_final_state)

            # Look up inputs
            decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_inputs, name='decoder_input')
            is_lm_mode_tensor = tf.to_float(tf.expand_dims(self.is_lm_mode, axis=1))
            decoder_inputs = tf.concat([decoder_inputs_embedded, is_lm_mode_tensor], axis=1)

            # Concatenate class info vector
            decoder_inputs = tf.concat([decoder_inputs, self.class_info_vec], axis=1)

            # Unpack state
            initial_state = unpack_state_tensor(self.decoder_state)

            with tf.variable_scope('decoder', reuse=True):
                decoder_output, decoder_final_state, self.decoder_new_attention = self.decoder.predict(
                    inputs=decoder_inputs,
                    initial_state=initial_state,
                    attention_states=self.decoder_attention
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
