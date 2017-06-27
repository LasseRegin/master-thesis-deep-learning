
import collections

import tensorflow as tf

import network
import data
import utils

model2class = {
    'rnn_lstm': network.RNN_LSTM,
    'rnn_blstm': network.RNN_BLSTM,
    'rnn_blstm_attention': network.RNN_BLSTM_Attention,
    'rnn_blstm_attention_c2w': network.RNN_BLSTM_Attention_C2W,
    'rnn_blstm_attention_c2w_bn': network.RNN_BLSTM_Attention_C2W_BN,
    'rnn_blstm_classifier': network.RNN_BLSTM_Classifier,
    'rnn_blstm_classifier_regular': network.RNN_BLSTM_Classifier_Regular,
    'rnn_blstm_attention_classifier': network.RNN_BLSTM_Attention_Classifier,
    'rnn_blstm_attention_classifier_regular': network.RNN_BLSTM_Attention_Classifier_Regular,
    'rnn_lm': network.RNN_LM
}

_models_with_classes = [
    'rnn_blstm_classifier',
    'rnn_blstm_classifier_regular',
    'rnn_blstm_attention_classifier',
    'rnn_blstm_attention_classifier_regular',
]

def load_frozen_network(config):
    # Load frozen graph
    with tf.gfile.FastGFile(config.frozen_model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    decoder_inputs, decoder_state, decoder_final_state, decoder_outputs, decoder_probs = tf.import_graph_def(
        graph_def,
        name='',
        return_elements=[
            'decoder_inputs:0',
            'decoder_state:0',
            'decoder_2/decoder_final_state:0',
            'decoder_2/decoder_output:0',
            'decoder_2/decoder_probs:0'
        ]
    )

    def prediction_func(session, inputs, state):
        return session.run(
            fetches=[decoder_outputs, decoder_probs, decoder_final_state],
            feed_dict={
                decoder_inputs: inputs,
                decoder_state:  state
            }
        )

    # tf.reset_default_graph()

    return prediction_func


def setup_config(flags, data_name, alphabet, num_classes):
    network_class = model2class.get(flags.model.lower(), None)
    if network_class is None:
        raise KeyError('Invalid model parameter provided')

    # Define config
    return network.Config(
        network_class_name=network_class.__name__,
        data_name=data_name,
        max_enc_seq_length=flags.max_enc_seq_length,
        max_dec_seq_length=flags.max_dec_seq_length,
        embedding_size=flags.embedding_size,
        alphabet=alphabet,
        num_cells=flags.num_cells,
        num_units=flags.num_units,
        scheduled_sampling=flags.scheduled_sampling,
        max_gradient_norm=flags.max_gradient_norm,
        keep_prob=flags.keep_prob,
        max_features=flags.max_features,
        alpha=flags.alpha,
        question_state_units=flags.q_units,
        num_classes=num_classes,
        sample_type=flags.sample_type,
        beam_size=flags.beam_size,
        random_seed=flags.random_seed,
        lm_support=flags.lm_support,
        W_lambda=flags.W_lambda
    )


def setup_model(flags, prediction_mode, alphabet, vocabulary, data_name,
                num_classes, ss_decay_func=lambda global_step: 1.0):
    network_class = model2class.get(flags.model.lower(), None)
    if network_class is None:
        raise KeyError('Invalid model parameter provided')

    # Define config
    config = setup_config(
        flags=flags,
        data_name=data_name,
        alphabet=alphabet,
        num_classes=num_classes
    )

    # Initialize network object
    return network_class(
        config=config,
        learning_rate=flags.learning_rate,
        prediction_mode=prediction_mode,
        alphabet=alphabet,
        vocabulary=vocabulary,
        clean=flags.clean,
        verbose=flags.verbose,
        epochs=flags.epochs,
        beam_size=flags.beam_size,
        sample_type=flags.sample_type,
        swap_memory=flags.swap_memory,
        create_summary=flags.create_summary,
        parallel_iterations=flags.parallel_iterations,
        ss_decay_func=ss_decay_func,
    )


def get_flags():
    tf.app.flags.DEFINE_integer('max_enc_seq_length', 50, 'Maximum encoder sequence length')
    tf.app.flags.DEFINE_integer('max_dec_seq_length', 50, 'Maximum decoder sequence length')
    tf.app.flags.DEFINE_integer('embedding_size', 32, 'Dimensionality of embedding')
    tf.app.flags.DEFINE_integer('num_units', 16, 'Number of units in the RNN cell')
    tf.app.flags.DEFINE_integer('batch_size', 64, 'Size of mini-batches during training')
    tf.app.flags.DEFINE_integer('epochs', 10, 'Number of epochs to train')
    tf.app.flags.DEFINE_integer('num_cells', 2, 'Number of RNN cells to use')
    tf.app.flags.DEFINE_integer('save_epochs', 10, 'Save model on every `save_epochs` epochs')
    tf.app.flags.DEFINE_integer('patience', 20, 'Allowed epochs without improving validation.')
    tf.app.flags.DEFINE_integer('random_seed', 42, 'Define random seed used for splitting data.')
    tf.app.flags.DEFINE_integer('max_features', 512, 'Number of word features.')
    tf.app.flags.DEFINE_integer('q_units', 32, 'Number of units in question state projections.')
    tf.app.flags.DEFINE_integer('update_every', 100, 'Update tensorboard summary on every \
                                                    `update_every` training batch.')
    tf.app.flags.DEFINE_integer('beam_size', 1, 'Size of beam in beam search.')
    tf.app.flags.DEFINE_integer('parallel_iterations', 32, 'Number of parallel iterations \
                                                            used in the tf.while_loop.')
    tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate to use during training')
    tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Maximum allowed norm of gradients')
    tf.app.flags.DEFINE_float('keep_prob', 1.0, 'Input and output keep probability (inverse dropout)')
    tf.app.flags.DEFINE_float('alpha', 0.5, 'Trade-off parameter between learning sequences \
                                             and learning class predictions (1.0 is sequences only).')
    tf.app.flags.DEFINE_float('W_lambda', 0.0001, 'L2-regularization parameter.')
    tf.app.flags.DEFINE_float('mix_ratio', 0.50, 'Ratio of QA observations in the mixed batch loader.')
    tf.app.flags.DEFINE_string('model', 'RNN_LSTM', 'Choose model to train.')
    tf.app.flags.DEFINE_string('data', 'QA', 'Choose data to train on.')
    tf.app.flags.DEFINE_string('mode', 'train', 'Determine what mode to run the script in. \
                                                 Options are "train", "test", "plot".')
    tf.app.flags.DEFINE_string('sample_type', 'argmax', 'Determine what type of prediction sampling \
                                                         is going to be used.')
    tf.app.flags.DEFINE_boolean('scheduled_sampling', False, 'Use scheduled sampling during traning.')
    tf.app.flags.DEFINE_boolean('clean', False, 'Reset saved model checkpoints.')
    tf.app.flags.DEFINE_boolean('verbose', True, 'Print information during training.')
    tf.app.flags.DEFINE_boolean('lm_support', False, 'Whether to use the corresponding \
                                                      language model to support predictions.')
    tf.app.flags.DEFINE_boolean('swap_memory', False, 'Whether to use the `swap_memory` options \
                                                       in the tf.while_loop.')
    tf.app.flags.DEFINE_boolean('create_summary', False, 'Add summary statistics for using \
                                                          Tensorboard.')
    tf.app.flags.DEFINE_boolean('mix_data', False, 'Mix QA data with Wiki data.')

    return tf.app.flags.FLAGS


def get_lm_flags(FLAGS):
    FLAGS._parse_flags()
    flags_dict = FLAGS.__dict__['__flags'].copy()

    # Set LM config values
    flags_dict['model']              = 'RNN_LM'
    flags_dict['embedding_size']     = 256
    flags_dict['num_units']          = 1024
    flags_dict['num_cells']          = 1
    flags_dict['keep_prob']          = 1.00
    flags_dict['data']               = 'wiki'
    # flags_dict['max_dec_seq_length'] = 100
    flags_dict['max_dec_seq_length'] = 500

    flags_dict['random_seed'] = 42

    LM_FLAGS = collections.namedtuple('FLAGS', flags_dict.keys())
    flags = LM_FLAGS(**flags_dict)

    return flags


def setup_batch_loaders(data_type, mix_data_loaders, max_enc_seq_length, max_dec_seq_length, batch_size, random_seed, model,
                        mix_ratio=0.5, verbose=False):
    data_type = data_type.lower().strip()
    assert data_type in ['qa', 'qa_ext', 'qa_healthtap', 'wiki', 'europarl', 'imdb']
    assert model in list(model2class.keys())

    classes_only = False
    # if model in ['rnn_blstm_classifier']:
    #     classes_only = True
    # else:
    #     classes_only = False

    # Define alphabet
    alphabet = data.Alphabet.create_standard_alphabet()

    kwargs = {
        'max_enc_seq_length': max_enc_seq_length,
        'max_dec_seq_length': max_dec_seq_length,
        'random_seed':        random_seed,
        'classes_only':       classes_only,
        'verbose':            verbose
    }

    vocabulary   = None
    num_classes  = None
    category2idx = None
    idx2category = None

    if data_type in ['qa', 'qa_ext', 'qa_healthtap']:
        if data_type == 'qa':
            QA_class = data.QALoader
        elif data_type == 'qa_healthtap':
            QA_class = data.HealthtapQALoader
        else:
            QA_class = data.ExtendedQALoader

        qa_data_loader_train = QA_class(validation='train', alphabet=alphabet, **kwargs)
        qa_data_loader_val   = QA_class(validation='val',   alphabet=alphabet, **kwargs)
        qa_data_loader_test  = QA_class(validation='test',  alphabet=alphabet, **kwargs)
        name = qa_data_loader_train.name
        num_classes = qa_data_loader_train.num_classes
        category2idx = qa_data_loader_train.meta['category2idx']
        idx2category = {idx: cat for cat, idx in category2idx.items()}
        vocabulary = qa_data_loader_train.vocabulary

    if data_type == 'wiki' or mix_data_loaders:
        # kwargs['max_features'] = 0 # Row lengths must match to concatenate
        wiki_data_loader_train = data.WikiLoader(validation='train', alphabet=alphabet, **kwargs)
        wiki_data_loader_val   = data.WikiLoader(validation='val',   alphabet=alphabet, **kwargs)
        wiki_data_loader_test  = data.WikiLoader(validation='test',  alphabet=alphabet, **kwargs)
        name = wiki_data_loader_train.name

    # Initialize batch loaders
    if data_type in ['qa', 'qa_ext', 'qa_healthtap', 'wiki'] and mix_data_loaders:
        name = ''.join([qa_data_loader_train.name, wiki_data_loader_train.name, str(mix_ratio)])
        train_loader = utils.MixedBatchLoader(
            data_loader_1=qa_data_loader_train,
            data_loader_2=wiki_data_loader_train,
            batch_size=batch_size,
            mix_ratio=mix_ratio
        )

        # NOTE: We only evaluate on the QA data to be able to compare models
        # TODO: Is this the correct thing to do?
        val_loader   = utils.BatchLoader(data_loader=qa_data_loader_val,   batch_size=batch_size)
        test_loader  = utils.BatchLoader(data_loader=qa_data_loader_test,  batch_size=batch_size)
    elif data_type in ['qa', 'qa_ext', 'qa_healthtap']:
        train_loader = utils.BatchLoader(data_loader=qa_data_loader_train, batch_size=batch_size)
        val_loader   = utils.BatchLoader(data_loader=qa_data_loader_val,   batch_size=batch_size)
        test_loader  = utils.BatchLoader(data_loader=qa_data_loader_test,  batch_size=batch_size)
    elif data_type == 'wiki':
        train_loader = utils.BatchLoader(data_loader=wiki_data_loader_train, batch_size=batch_size)
        val_loader   = utils.BatchLoader(data_loader=wiki_data_loader_val,   batch_size=batch_size)
        test_loader  = utils.BatchLoader(data_loader=wiki_data_loader_test,  batch_size=batch_size)
    elif data_type == 'europarl':
        train_loader = data.EuroparlLoader(validation='train', alphabet=None, **kwargs)
        val_loader   = data.EuroparlLoader(validation='val',   alphabet=None, **kwargs)
        test_loader  = data.EuroparlLoader(validation='test',  alphabet=None, **kwargs)
        name = train_loader.name
        vocabulary = train_loader.vocabulary

        # Use custom alphabet when using europarl dataset
        alphabet = train_loader.alphabet

        train_loader = utils.BatchLoader(data_loader=train_loader, batch_size=batch_size)
        val_loader   = utils.BatchLoader(data_loader=val_loader,   batch_size=batch_size)
        test_loader  = utils.BatchLoader(data_loader=test_loader,  batch_size=batch_size)
    elif data_type == 'imdb':
        train_loader = data.IMDBLoader(validation='train', alphabet=alphabet, **kwargs)
        val_loader   = data.IMDBLoader(validation='val',   alphabet=alphabet, **kwargs)
        test_loader  = data.IMDBLoader(validation='test',  alphabet=alphabet, **kwargs)
        name = train_loader.name
        vocabulary = train_loader.vocabulary
        num_classes = train_loader.num_classes

        train_loader = utils.BatchLoader(data_loader=train_loader, batch_size=batch_size)
        val_loader   = utils.BatchLoader(data_loader=val_loader,   batch_size=batch_size)
        test_loader  = utils.BatchLoader(data_loader=test_loader,  batch_size=batch_size)
    else:
        raise KeyError('Invalid data_type provided')

    # Sample classes and define "class_idx" key
    if model in _models_with_classes:
        train_loader = utils.ClassSamplingBatchLoader(train_loader)
        val_loader   = utils.ClassSamplingBatchLoader(val_loader)
        test_loader  = utils.ClassSamplingBatchLoader(test_loader)

    return {
        'train_loader': train_loader,
        'val_loader':   val_loader,
        'test_loader':  test_loader,
        'alphabet':     alphabet,
        'vocabulary':   vocabulary,
        'num_classes':  num_classes,
        'name':         name,
        'category2idx': category2idx,
        'idx2category': idx2category
    }
