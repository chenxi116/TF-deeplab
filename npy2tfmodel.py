# Copyright 2017 Chenxi Liu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# sample usage:
# python npy2tfmodel.py 0 ./model/ResNet101_init.npy ./model/ResNet101_init.tfmodel

import numpy as np
import tensorflow as tf
import deeplab_model
import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import pdb


weights = np.load(sys.argv[2])[()]

model = deeplab_model.DeepLab()

sess = tf.Session()
sess.run(tf.initialize_all_variables())
var_list = tf.all_variables()
count = 0
for item in var_list:
    item_name = item.name[8:-2] # "DeepLab/" at beginning, ":0" at last
    if not item_name in weights.keys():
        continue
    print item_name
    count += 1
    sess.run(tf.assign(item, weights[item_name]))
assert(count == len(weights))

snapshot_saver = tf.train.Saver()
snapshot_saver.save(sess, sys.argv[3])