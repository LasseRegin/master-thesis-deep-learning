
import os

import tensorflow as tf
import numpy as np

import helpers

# Get flags
FLAGS = helpers.get_flags()

# Initialize data loader
data_loader_dict = helpers.setup_batch_loaders(
    data_type=FLAGS.data,
    mix_data_loaders=FLAGS.mix_data,
    max_enc_seq_length=FLAGS.max_enc_seq_length,
    max_dec_seq_length=FLAGS.max_dec_seq_length,
    batch_size=FLAGS.batch_size,
    random_seed=FLAGS.random_seed,
    model=FLAGS.model.lower()
)

# Setup model
model = helpers.setup_model(
    flags=FLAGS,
    prediction_mode=True,
    alphabet=data_loader_dict['alphabet'],
    vocabulary=data_loader_dict['vocabulary'],
    data_name=data_loader_dict['name'],
    num_classes=data_loader_dict['num_classes']
)

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # Initialize model
    model.init(sess)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    for name in [n.name for n in input_graph_def.node]:
        print(name)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def=input_graph_def,
        output_node_names=model.output_node_names
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(model.config.frozen_model_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    print('%d ops in the final graph.' % (len(output_graph_def.node)))
del sess
tf.reset_default_graph()
