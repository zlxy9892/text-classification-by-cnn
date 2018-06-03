# -*- coding: utf-8 -*-

import sys
import os
import time
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.contrib import learn
import gflags
import data_helpers
from text_cnn import TextCNN


### parameters ###
# ===============================================

FLAGS = gflags.FLAGS

# data loading parameters
gflags.DEFINE_string('positive_data_file', './inputs/rt.pos', 'Data source for positive data.')
gflags.DEFINE_string('negative_data_file', './inputs/rt.neg', 'Data source for negative data.')

# evaluate parameters
gflags.DEFINE_integer('batch_size', 64, 'Batch size (default: 64).')
gflags.DEFINE_string('checkpoint_dir', './', 'Checkpoint directory from the training.')
gflags.DEFINE_bool('eval_train', True, 'Evalute on all the training data.')

# device parameters
gflags.DEFINE_bool('allow_soft_placement', True, 'Allow device soft device placement.')
gflags.DEFINE_bool('log_device_placement', False, 'Log placement of ops on devices.')

FLAGS(sys.argv)
# show parameters
print('\nPARAMETERS:')
print('================================')
for attr, value in FLAGS.flag_values_dict().items():
    print('{0}: {1}'.format(attr.upper(), value))
print('================================\n\n')
input('press enter to start...\n\n')


# load the train data or your own data.
if FLAGS.eval_train is True:
    x_text, y_test = data_helpers.load_text_and_label(file_pos_file=FLAGS.positive_data_file, file_neg_file=FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_text = ['it is so bad.', 'nice buying experience.']
    y_test = [0, 1]

# load the vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, '..', 'vocab')
vocab_prosessor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_prosessor.transform(x_text)))

print('Evaluating ...\n')

### evaluating
# ===============================================

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print('Loaded the latest checkpoint: <{}>\n'.format(checkpoint_file))

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("\nTotal number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}\n".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
df_pred = np.column_stack((np.array(x_text), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving prediction csv file to {0}".format(out_path))
pd.DataFrame(df_pred).to_csv(out_path, header=['text', 'prediction'])
