
import collections

import numpy as np

from .ops import np_softmax


class Predictor:
    def __init__(self, batch_size, max_length, alphabet, decode_func, loss_func):
        self.batch_size    = batch_size
        self.max_length    = max_length
        self.alphabet      = alphabet
        self.alphabet_size = len(self.alphabet)
        self.decode_func   = decode_func
        self.loss_func     = loss_func


class ArgmaxPredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def predict_sequences(self, initial_state, target, input_length,
                          features={}):
        """ Predict sequences from given inputs

        Args:


        Returns:
            candidates: Predicted candidates of shape
                [batch_size, 1, max_length]
            logits: Corresponding logits for each predicted class of shape
                [batch_size, 1, max_length, alphabet_size]
        """

        # Define initial input
        inputs = np.full(shape=(self.batch_size,), fill_value=self.alphabet.GO_ID, dtype=np.float32)
        state = initial_state

        predictions_list = []
        logits_list = []
        for i in range(0, self.max_length):
            decoder_dict = self.decode_func(
                inputs=inputs,
                state=state,
                probs_decay_parameter=0,
                **features
            )
            decoder_logits = decoder_dict['output']
            decoder_probs_decayed  = decoder_dict['probs_decayed']
            predictions = decoder_probs_decayed.argmax(axis=1)

            predictions_list.append(predictions)
            logits_list.append(decoder_logits)

            # Set new iteration input
            inputs = predictions
            state = decoder_dict['state']

        candidates = np.asarray(predictions_list, dtype=np.uint8)
        logits     = np.asarray(logits_list,      dtype=np.float32)
        candidates = np.transpose(candidates, [1, 0])
        logits     = np.transpose(logits, [1, 0, 2])

        # Compute loss and probabilities
        mean_losses, mean_prob_x = self.loss_func(
            logits=logits,
            targets=target,
            input_length=input_length
        )

        # Reshape results
        loss_candidates   = mean_losses.reshape((self.batch_size, 1))
        prob_x_candidates = mean_prob_x.reshape((self.batch_size, 1))
        final_candidates  = candidates.reshape((self.batch_size, 1, self.max_length))
        final_logits      = logits.reshape((self.batch_size, 1, self.max_length, self.alphabet_size))

        return final_candidates, final_logits, loss_candidates, prob_x_candidates


class SamplingPredictor(Predictor):
    def __init__(self, num_samples, **kwargs):
        super().__init__(**kwargs)

        self.num_samples = num_samples


    def predict_sequences(self, initial_state, target, input_length,
                          features={}):
        """ Predict sequences from given inputs

        Args:


        Returns:
            candidates: Predicted candidates of shape
                [batch_size, samples, max_length]
            logits: Corresponding logits for each predicted class of shape
                [batch_size, samples, max_length, alphabet_size]
        """

        # Define initial input
        initial_inputs = np.full(shape=(self.batch_size,), fill_value=self.alphabet.GO_ID, dtype=np.float32)

        # Repeat for multiple samples
        inputs           = np.repeat(initial_inputs, repeats=self.num_samples, axis=0)
        state            = np.repeat(initial_state,  repeats=self.num_samples, axis=0)
        dec_target       = np.repeat(target,         repeats=self.num_samples, axis=0)
        dec_input_length = np.repeat(input_length,   repeats=self.num_samples, axis=0)

        # Repeat extra features as well
        features = {key: np.repeat(value, repeats=self.num_samples, axis=0)
                    for key, value in features.items()}

        predictions_list = []
        logits_list = []
        for i in range(0, self.max_length):
            decoder_dict = self.decode_func(
                inputs=inputs,
                state=state,
                probs_decay_parameter=3.0 * (i / self.max_length),
                **features
            )
            decoder_logits         = decoder_dict['output']
            decoder_probs_decayed  = decoder_dict['probs_decayed']
            for key, value in features.items():
                if key in decoder_dict:
                    features[key] = decoder_dict[key]

            logits_list.append(decoder_logits)

            predictions = []
            for batch_num in range(0, self.batch_size):
                batch_offset = self.num_samples * batch_num
                for sample_num in range(0, self.num_samples):
                    idx = batch_offset + sample_num

                    # Parse to float64 to avoid numerically instabilities
                    sample_probs = decoder_probs_decayed[idx, :].astype(np.float64)
                    symbol = np.random.multinomial(1, pvals=sample_probs).argmax()
                    predictions.append(symbol)
            predictions = np.asarray(predictions, dtype=np.uint8)
            predictions_list.append(predictions)

            # Set new iteration input
            inputs = predictions
            state = decoder_dict['state']

            # Add some uniform noise to the probabilties to increase diversity
            # of generated sequences
            # diversity = 1.0
            # noise =  np.random.uniform(low=-1, high=1, size=beam_probs.shape)
            # beam_probs += diversity * noise

            # TODO: Make it possible to add a certain amount of uniformly dist. noise
            #       to this probability in order to get more diverse predictions
        candidates = np.asarray(predictions_list, dtype=np.uint8)
        logits     = np.asarray(logits_list,      dtype=np.float32)
        candidates = np.transpose(candidates, [1, 0])
        logits     = np.transpose(logits, [1, 0, 2])

        # Compute loss and probabilities
        mean_losses, mean_prob_x = self.loss_func(
            logits=logits,
            targets=dec_target,
            input_length=dec_input_length
        )

        # Reshape results
        loss_candidates   = mean_losses.reshape((self.batch_size, self.num_samples))
        prob_x_candidates = mean_prob_x.reshape((self.batch_size, self.num_samples))
        final_candidates  = candidates.reshape((self.batch_size, self.num_samples, self.max_length))
        final_logits      = logits.reshape((self.batch_size, self.num_samples, self.max_length, self.alphabet_size))

        return final_candidates, final_logits, loss_candidates, prob_x_candidates



class BeamSearchPredictor(Predictor):
    def __init__(self, beam_size, **kwargs):
        super().__init__(**kwargs)

        self.beam_size       = beam_size


    def predict_sequences(self, initial_state, target, input_length,
                          features={}):
        """ Predict sequences from given inputs

        Args:


        Returns:
            candidates: Predicted candidates of shape
                [batch_size, samples, max_length]
            logits: Corresponding logits for each predicted class of shape
                [batch_size, samples, max_length, alphabet_size]
        """

        # Get initial probabilties
        initial_inputs = np.full(shape=(self.batch_size,), fill_value=self.alphabet.GO_ID, dtype=np.float32)
        decoder_dict = self.decode_func(
            inputs=initial_inputs,
            state=initial_state,
            **features
        )
        decoder_output = decoder_dict['output']
        # NOTE: using logits instead of probs should also work, but if
        #       something wierd is going on maybe try changing it back
        # decoder_probs  = decoder_dict['probs']
        decoder_probs  = decoder_dict['output']
        for key, value in features.items():
            if key in decoder_dict:
                features[key] = decoder_dict[key]


        # Extract top beam_size probabilities
        top_idx = np.argsort(decoder_probs, axis=1)
        top_idx = top_idx[:,::-1]
        top_idx = top_idx[:,:self.beam_size]

        inputs = top_idx.reshape((self.batch_size * self.beam_size))
        state  = np.repeat(decoder_dict['state'], repeats=self.beam_size, axis=0)

        # Repeat for beam search
        dec_target       = np.repeat(target,        repeats=self.beam_size, axis=0)
        dec_input_length = np.repeat(input_length,  repeats=self.beam_size, axis=0)

        # Repeat extra features as well
        features = {key: np.repeat(value, repeats=self.beam_size, axis=0)
                    for key, value in features.items()}

        # Placeholders for candidates
        beam_logits        = np.zeros((self.batch_size, self.beam_size, self.max_length, self.alphabet_size))
        beam_probabilities = np.zeros((self.batch_size, self.beam_size))
        beam               = np.zeros((self.batch_size, self.beam_size, self.max_length)).astype(np.uint8)

        # Extract top probabilities
        batch_idx_vec = np.arange(0, self.batch_size).repeat(repeats=self.beam_size)
        top_idx_vec   = top_idx.reshape((self.batch_size * self.beam_size))
        top_probs     = decoder_probs[batch_idx_vec, top_idx_vec]
        top_probs     = top_probs.reshape((self.batch_size, self.beam_size))

        # Repeat logits
        logits = decoder_output.reshape((self.batch_size, 1, 1, self.alphabet_size))
        logits = np.repeat(logits, repeats=self.beam_size, axis=1)

        # Set new candidate symbols, logits, and probabilities
        beam[:,:,:1]            = inputs.reshape((self.batch_size, self.beam_size, 1))
        beam_logits[:,:,:1,:]   = logits
        beam_probabilities[:,:] = top_probs

        sequence_prob_pairs = collections.defaultdict(lambda: {
            'sequences': [],
            'probs': [],
            'logits': []
        })

        for i in range(1, self.max_length):
            decoder_dict = self.decode_func(
                inputs=inputs,
                state=state,
                **features
            )
            decoder_output = decoder_dict['output']
            # NOTE: using logits instead of probs should also work, but if
            #       something wierd is going on maybe try changing it back
            # decoder_probs  = decoder_dict['probs']
            decoder_probs  = decoder_dict['output']
            for key, value in features.items():
                if key in decoder_dict:
                    features[key] = decoder_dict[key]

            # Group results on observations
            decoder_probs = decoder_probs.reshape((self.batch_size, self.beam_size * self.alphabet_size))

            # Set probabilities of sequences coming from ended sequence to -inf
            parent_probs = beam_probabilities.repeat(repeats=self.alphabet_size, axis=1)
            is_done_idx = np.where(parent_probs == -float('inf'))
            for batch_idx, idx in zip(*is_done_idx):
                decoder_probs[batch_idx, idx] = -float('inf')

            # Extract top beam_size probabilities
            top_idx = np.argsort(decoder_probs, axis=1)
            top_idx = top_idx[:,::-1]
            top_idx = top_idx[:,:self.beam_size]

            # Determine beam_idx and symbols
            beam_idx   = top_idx // self.alphabet_size
            symbols    = top_idx %  self.alphabet_size

            # Extract top probabilities
            top_idx_vec = top_idx.reshape((self.batch_size * self.beam_size))
            # top_probs = beam_probs[batch_idx, self.batch_size]
            top_probs = decoder_probs[batch_idx_vec, top_idx_vec]
            top_probs = top_probs.reshape((self.batch_size, self.beam_size))

            # Define new candidates
            new_candidates = symbols.reshape((self.batch_size, self.beam_size, 1))

            # Set new input and states
            beam_idx_vec = beam_idx.reshape((self.batch_size * self.beam_size))
            unrolled_idx = batch_idx_vec * self.beam_size + beam_idx_vec
            inputs = symbols.reshape((self.batch_size * self.beam_size))
            state  = decoder_dict['state'][unrolled_idx]
            for key, value in features.items():
                features[key] = value[unrolled_idx]


            logits = decoder_output[unrolled_idx]
            logits = logits.reshape((self.batch_size, self.beam_size, 1, self.alphabet_size))

            ###################################################################
            # Get corresponding parent sequences and logits
            parent_symbols = beam[batch_idx_vec, beam_idx_vec, 0:i].reshape((self.batch_size, self.beam_size, i))
            parent_logits  = beam_logits[batch_idx_vec, beam_idx_vec, :i, :].reshape((self.batch_size, self.beam_size, i, self.alphabet_size))

            # Concatenate with new candidate symbols
            new_candidates = np.concatenate((parent_symbols, new_candidates), axis=2)

            # Concatenate with new candidate logits
            logits = np.concatenate((parent_logits, logits), axis=2)


            # Set new candidate symbols, logits, and probabilities
            beam[:,:,:i+1]          = new_candidates
            beam_logits[:,:,:i+1,:] = logits
            beam_probabilities[:,:] = top_probs

            # Find and remove done sequences
            done_seq_idx = np.where(new_candidates == self.alphabet.EOS_ID)
            for batch_numb, beam_numb, _ in zip(*done_seq_idx):
                candidate = new_candidates[batch_numb, beam_numb]

                zeros = np.zeros(self.max_length - i - 1).astype(np.uint8)
                candidate = np.concatenate((candidate, zeros), axis=0)
                prob = beam_probabilities[batch_numb, beam_numb]
                candidate_logits = beam_logits[batch_numb, beam_numb]
                beam_probabilities[batch_numb, beam_numb] = -float('inf')

                sequence_prob_pairs[batch_numb]['probs'].append(prob)
                sequence_prob_pairs[batch_numb]['logits'].append(candidate_logits)
                sequence_prob_pairs[batch_numb]['sequences'].append(candidate)


        # Add final sequences
        for batch_numb in range(0, self.batch_size):
            for beam_numb in range(0, self.beam_size):
                candidate = beam[batch_numb, beam_numb]
                logits    = beam_logits[batch_numb, beam_numb]
                prob      = beam_probabilities[batch_numb, beam_numb]
                sequence_prob_pairs[batch_numb]['probs'].append(prob)
                sequence_prob_pairs[batch_numb]['logits'].append(logits)
                sequence_prob_pairs[batch_numb]['sequences'].append(candidate)

        # Extract top candidates
        final_candidates = np.zeros((self.batch_size, self.beam_size, self.max_length)).astype(np.uint8)
        final_logits     = np.zeros((self.batch_size, self.beam_size, self.max_length, self.alphabet_size))
        for batch_numb, candidates_dict in sequence_prob_pairs.items():
            probs      = np.asarray(candidates_dict['probs'])
            logits     = np.asarray(candidates_dict['logits'])
            candidates = np.asarray(candidates_dict['sequences'])

            # Extract top values
            sorted_idx = probs.argsort()[::-1]
            top_idx = sorted_idx[0:self.beam_size]

            top_probs      = probs[top_idx]
            top_logits     = logits[top_idx]
            top_candidates = candidates[top_idx]

            final_candidates[batch_numb,:,:] = top_candidates
            final_logits[batch_numb,:,:]     = top_logits

        # Compute loss and probabilities
        mean_losses, mean_prob_x = self.loss_func(
            logits=final_logits.reshape((self.batch_size * self.beam_size, self.max_length, self.alphabet_size)),
            targets=dec_target.reshape((self.batch_size * self.beam_size, self.max_length)),
            input_length=dec_input_length.reshape((self.batch_size * self.beam_size))
        )

        loss_candidates   = mean_losses.reshape((self.batch_size, self.beam_size))
        prob_x_candidates = mean_prob_x.reshape((self.batch_size, self.beam_size))

        return final_candidates, final_logits, loss_candidates, prob_x_candidates
