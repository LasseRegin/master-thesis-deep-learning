
import sys
import time
import math

import scipy        # Import these before tensorflow to prevent segmentationfault
import numpy as np  # Import these before tensorflow to prevent segmentationfault
import tensorflow as tf

import helpers
import plotting

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
    model=FLAGS.model.lower(),
    mix_ratio=FLAGS.mix_ratio,
    verbose=FLAGS.verbose
)

_k = data_loader_dict['train_loader'].batch_count * 6.2446 + 139.9846
def inverse_sigmoid_decay(global_step):
    """ Returns the probability of using the correct token instead of the
        predicted token from previous time-step.
    """
    return 1.0 - _k / (_k + math.exp(global_step / _k))

# Setup model
model = helpers.setup_model(
    flags=FLAGS,
    prediction_mode=False,
    alphabet=data_loader_dict['alphabet'],
    vocabulary=data_loader_dict['vocabulary'],
    data_name=data_loader_dict['name'],
    num_classes=data_loader_dict['num_classes'],
    ss_decay_func=inverse_sigmoid_decay
)

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # Initialize model
    model.init(sess)

    train_batches_total = val_batches_total = 0
    for epoch in range(0, FLAGS.epochs):
        train_loss_1 = train_loss_2 = val_loss_1 = val_loss_2 = 0.0

        train_loss = 0.0
        train_prob_x = 0.0
        train_accuracy = 0.0
        train_batches = 0

        start = time.time()
        for data_dict in data_loader_dict['train_loader']:
            result = model.train_op(sess, **data_dict)
            train_loss += result['mean_loss']
            train_prob_x += result['mean_prob_x']
            train_accuracy += result.get('accuracy', 0.0)
            train_batches += 1
            train_batches_total += 1

            train_loss_1 += result.get('mean_seq_loss', 0.0)
            train_loss_2 += result.get('mean_class_loss', 0.0)

            if FLAGS.create_summary and (train_batches_total % FLAGS.update_every) == 0:
                model.train_writer.add_summary(
                    summary=result['summary'],
                    global_step=result['global_step']
                )

        val_loss = 0.0
        val_prob_x = 0.0
        val_accuracy = 0.0
        val_batches = 0
        for data_dict in data_loader_dict['val_loader']:
            result = model.val_op(sess, **data_dict)
            val_loss += result['mean_loss']
            val_prob_x += result['mean_prob_x']
            val_accuracy += result.get('accuracy', 0.0)
            val_batches += 1
            val_batches_total += 1

            val_loss_1 += result.get('mean_seq_loss', 0.0)
            val_loss_2 += result.get('mean_class_loss', 0.0)

            if FLAGS.create_summary and (val_batches_total % FLAGS.update_every) == 0:
                model.val_writer.add_summary(
                    summary=result['summary'],
                    global_step=result['global_step']
                )

        model.add_epoch(
            session=sess,
            train_loss=train_loss / train_batches,
            train_prob_x=train_prob_x / train_batches,
            train_accuracy=train_accuracy / train_batches,
            val_loss=val_loss / val_batches,
            val_prob_x=val_prob_x / val_batches,
            val_accuracy=val_accuracy / val_batches,

            # For the mixed loss plot
            train_loss_1=train_loss_1 / train_batches,
            train_loss_2=train_loss_2 / train_batches,
            val_loss_1=val_loss_1 / val_batches,
            val_loss_2=val_loss_2 / val_batches
        )

        if not model.is_improving(patience=FLAGS.patience):
            break

        # Save model
        if epoch > 0 and (epoch % FLAGS.save_epochs) == 0:
            model.save(sess)

        # Push printed messages to stdout
        sys.stdout.flush()
    model.save(sess)

    # Plot results
    loss_plotter = plotting.LossPlotter(
        loss_maintainer=model.loss_maintainer,
        config=model.config
    )
    loss_plotter.save()

    # Plot mixed losses
    if hasattr(model, 'mixed_loss_maintainer'):
        mixed_loss_plotter = plotting.MixedLossPlotter(
            mixed_loss_maintainer=model.mixed_loss_maintainer,
            config=model.config
        )
        mixed_loss_plotter.save()
