
import os

class Config:
    CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoints')
    SUMMARY_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'summary')

    def __init__(
        self,
        network_class_name,
        data_name,
        max_enc_seq_length,
        max_dec_seq_length,
        embedding_size,
        alphabet,
        num_cells,
        num_units,
        scheduled_sampling,
        max_gradient_norm,
        keep_prob,
        random_seed,
        lm_support,
        sample_type=None,
        beam_size=None,
        max_features=None,
        alpha=None,
        question_state_units=None,
        num_classes=None,
        W_lambda=None
    ):
        assert max_enc_seq_length >= 0
        assert max_dec_seq_length >= 0
        assert embedding_size > 0
        assert max_gradient_norm > 0.0
        assert num_cells > 0 and isinstance(num_cells, int)
        assert num_units > 0 and isinstance(num_cells, int)

        # Parameters
        self.model                = network_class_name.lower().strip()
        self.data_name            = data_name
        self.max_enc_seq_length   = max_enc_seq_length
        self.max_dec_seq_length   = max_dec_seq_length
        self.alphabet_size        = len(alphabet)
        self.embedding_size       = embedding_size
        self.max_gradient_norm    = max_gradient_norm
        self.num_units            = num_units
        self.num_cells            = num_cells
        self.keep_prob            = keep_prob
        self.num_classes          = num_classes
        self.scheduled_sampling   = scheduled_sampling
        self.max_features         = max_features
        self.alpha                = alpha
        self.question_state_units = question_state_units
        self.random_seed          = random_seed
        self.lm_support           = lm_support
        self.W_lambda             = W_lambda
        self.max_words            = self.max_enc_seq_length // 4

        # Model filenames
        self.name              = '%s-%s' % (network_class_name, self.data_name)
        self.model_type_folder = os.path.join(self.CHECKPOINT_DIR, '%s/' % (self.name))
        self.model_folder      = os.path.join(self.model_type_folder, '%s/' % (self.architecture_string))
        self.save_path         = os.path.join(self.model_folder, 'model')

        # Encoded states files
        self.encoder_states_filename   = os.path.join(self.model_folder, 'encoder_states.json')
        self.projected_states_filename = os.path.join(self.model_folder, 'projected_states.json')

        # Summary folder
        self.summary_path = os.path.join(self.SUMMARY_DIR, '%s-%s' % (self.name, self.architecture_string))
        self.summary_train_path = os.path.join(self.summary_path, 'train')
        self.summary_val_path = os.path.join(self.summary_path, 'val')

        # Frozen graph file
        self.frozen_model_path = os.path.join(self.model_folder, 'frozen_model.pb')

        # Epochs filename
        self.epochs_filename = os.path.join(self.model_folder, 'epochs.json')
        self.epochs_mixed_filename = os.path.join(self.model_folder, 'epochs-mixed.json')

        # Define prediction files
        if sample_type == 'sample':
            prediction_filename = 'sample-predictions-%d' % (beam_size)
        elif sample_type == 'beam':
            prediction_filename = 'beam-predictions-%d' % (beam_size)
        elif sample_type == 'argmax':
            prediction_filename = 'argmax-predictions-%d' % (beam_size)

        if self.lm_support:
            prediction_filename = '%s-with-lm' % (prediction_filename)

        self.prediction_filename = os.path.join(self.model_folder, '%s.json' % (prediction_filename))

        self.create_folders()

    def create_folders(self):
        # Make folders if not existing already
        for folder in [self.CHECKPOINT_DIR, self.SUMMARY_DIR, self.model_type_folder, self.model_folder, self.summary_path]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

    @property
    def architecture_string(self):
        if self.model == 'rnn_lstm':
            return '-'.join([
                '%d'    % (self.max_enc_seq_length),
                '%d'    % (self.max_dec_seq_length),
                '%d'    % (self.embedding_size),
                '%.2f'  % (self.max_gradient_norm),
                '%d'    % (self.alphabet_size),
                '%d'    % (self.num_cells),
                '%d'    % (self.num_units),
                '%.2f'  % (self.keep_prob),
                '%d'    % (self.random_seed)
            ])
        elif self.model == 'rnn_blstm':
            return  '-'.join([
                '%d'    % (self.max_enc_seq_length),
                '%d'    % (self.max_dec_seq_length),
                '%d'    % (self.embedding_size),
                '%.2f'  % (self.max_gradient_norm),
                '%d'    % (self.alphabet_size),
                '%d'    % (self.num_cells),
                '%d'    % (self.num_units),
                '%.2f'  % (self.keep_prob),
                '%d'    % (self.random_seed)
            ])
        elif self.model in ['rnn_blstm_attention', 'rnn_blstm_attention_c2w', 'rnn_blstm_attention_c2w_bn']:
            return  '-'.join([
                '%d'    % (self.max_enc_seq_length),
                '%d'    % (self.max_dec_seq_length),
                '%d'    % (self.embedding_size),
                '%.2f'  % (self.max_gradient_norm),
                '%d'    % (self.alphabet_size),
                '%d'    % (self.num_cells),
                '%d'    % (self.num_units),
                '%.2f'  % (self.keep_prob),
                '%d'    % (self.random_seed)
            ])
        elif self.model == 'rnn_blstm_classifier':
            return '-'.join([
                '%d'    % (self.max_enc_seq_length),
                '%d'    % (self.max_dec_seq_length),
                '%d'    % (self.embedding_size),
                '%.2f'  % (self.max_gradient_norm),
                '%d'    % (self.alphabet_size),
                '%d'    % (self.num_cells),
                '%d'    % (self.num_units),
                '%.2f'  % (self.keep_prob),
                '%d'    % (self.max_features),
                '%.6f'  % (self.W_lambda),
                '%d'    % (self.random_seed)
            ])
        elif self.model in ['rnn_blstm_classifier_regular']:
            return '-'.join([
                '%d'    % (self.max_enc_seq_length),
                '%d'    % (self.max_dec_seq_length),
                '%d'    % (self.embedding_size),
                '%.2f'  % (self.max_gradient_norm),
                '%d'    % (self.alphabet_size),
                '%d'    % (self.num_cells),
                '%d'    % (self.num_units),
                '%.2f'  % (self.keep_prob),
                '%d'    % (self.max_features),
                '%.6f'  % (self.W_lambda),
                '%d'    % (self.random_seed)
            ])
        elif self.model == 'rnn_blstm_attention_classifier':
            return '-'.join([
                '%d'    % (self.max_enc_seq_length),
                '%d'    % (self.max_dec_seq_length),
                '%d'    % (self.embedding_size),
                '%.2f'  % (self.max_gradient_norm),
                '%d'    % (self.alphabet_size),
                '%d'    % (self.num_cells),
                '%d'    % (self.num_units),
                '%.2f'  % (self.keep_prob),
                '%.2f'  % (self.alpha),
                '%d'    % (self.question_state_units),
                '%.6f'  % (self.W_lambda),
                '%d'    % (self.random_seed)
            ])
        elif self.model == 'rnn_lm':
            return '-'.join([
                '%d'    % (self.max_dec_seq_length),
                '%d'    % (self.embedding_size),
                '%.2f'  % (self.max_gradient_norm),
                '%d'    % (self.alphabet_size),
                '%d'    % (self.num_cells),
                '%d'    % (self.num_units),
                '%.2f'  % (self.keep_prob),
                '%d'    % (self.random_seed)
            ])
        else:
            raise KeyError('Invalid model name provided!')


    def str_representation(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.str_representation()

    def __str__(self):
        return self.str_representation()
