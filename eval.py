# -*- coding: utf-8 -*-

import sys
import os
import time
import datetime
import numpy as np
import tensorflow as tf
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
gflags.DEFINE_string('checkpoint_dir', '', 'Checkpoint directory from the training.')
gflags.DEFINE_bool('eval_train', False, 'Evalute on all the training data.')

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
