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

# This code is modified from 
# https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/loadcaffe.py

# sample usage:
# python caffemodel2npy.py deploy.prototxt 
# ../deeplab/ResNet101/init.caffemodel ./model/ResNet101_init.npy

import numpy as np
import pdb
import re


class CaffeLayerProcessor(object):
    def __init__(self, net):
        self.net = net
        self.layer_names = net._layer_names
        self.param_dict = {}
        self.processors = {
            'Convolution': self.proc_conv,
            'InnerProduct': self.proc_fc,
            'BatchNorm': self.proc_bn,
            'Scale': self.proc_scale
        }

    def process(self):
        for idx, layer in enumerate(self.net.layers):
            param = layer.blobs
            name = self.layer_names[idx]
            if 'res05' in name or 'res075' in name:
            	continue
            if layer.type in self.processors:
                name_ = self.rename(name)
                dic = self.processors[layer.type](idx, name_, param)
                self.param_dict.update(dic)
        return self.param_dict

    def rename(self, caffe_layer_name):
        if caffe_layer_name.startswith('scale'):
            caffe_layer_name = 'bn' + caffe_layer_name[5:]

        NAME_MAP = {'bn_conv1': 'group_1/bn_conv1',
            'conv1': 'group_1/conv1',
            'fc1_voc12_c0': 'fc1_voc12/conv0',
            'fc1_voc12_c1': 'fc1_voc12/conv1',
            'fc1_voc12_c2': 'fc1_voc12/conv2',
            'fc1_voc12_c3': 'fc1_voc12/conv3'}
        if caffe_layer_name in NAME_MAP:
            return NAME_MAP[caffe_layer_name]

        s = re.search('([a-z]+)([0-9]+)([a-z]+)_', caffe_layer_name)
        if s is None:
            s = re.search('([a-z]+)([0-9]+)([a-z]+)([0-9]+)_', caffe_layer_name)
            layer_block_part1 = s.group(3)
            layer_block_part2 = s.group(4)
            assert layer_block_part1 in ['a', 'b']
            layer_block = 0 if layer_block_part1 == 'a' else int(layer_block_part2)
        else:
            layer_block = ord(s.group(3)) - ord('a')
        layer_type = s.group(1)
        layer_group = s.group(2)

        layer_branch = int(re.search('_branch([0-9])', caffe_layer_name).group(1))
        assert layer_branch in [1, 2]
        if layer_branch == 2:
            layer_id = re.search('_branch[0-9]([a-z])', caffe_layer_name).group(1)
            layer_id = ord(layer_id) - ord('a') + 1
        else:
            layer_id = 'add'

        TYPE_DICT = {'res':'conv', 'bn':'bn'}

        layer_type = TYPE_DICT[layer_type]
        tf_name = 'group_{}_{}/block_{}/{}'.format(
                int(layer_group), layer_block, layer_id, layer_type)
        print caffe_layer_name, tf_name
        return tf_name

    def proc_conv(self, idx, name, param):
        assert len(param) <= 2
        assert param[0].data.ndim == 4
        # caffe: ch_out, ch_in, h, w
        W = param[0].data.transpose(2,3,1,0)
        if len(param) == 1:
            return {name + '/DW': W}
        else:
            return {name + '/DW': W,
                    name + '/biases': param[1].data}

    def proc_fc(self, idx, name, param):
        # TODO caffe has an 'transpose' option for fc/W
        assert len(param) == 2
        prev_layer_name = self.net.bottom_names[name][0]
        prev_layer_output = self.net.blobs[prev_layer_name].data
        if prev_layer_output.ndim == 4:
            W = param[0].data
            # original: outx(CxHxW)
            W = W.reshape((-1,) + prev_layer_output.shape[1:]).transpose(2,3,1,0)
            # become: (HxWxC)xout
        else:
            W = param[0].data.transpose()
        return {name + '/DW': W.squeeze(),
                name + '/biases': param[1].data.squeeze()}

    def proc_bn(self, idx, name, param):
        # assert param[2].data[0] == 1.0
        return {name + '/mean': param[0].data,
                name + '/variance': param[1].data,
                name + '/factor': param[2].data }

    def proc_scale(self, idx, name, param):
        # bottom_name = self.net.bottom_names[name][0]
        # # find the bn layer before this scaling
        # for i, layer in enumerate(self.net.layers):
        #     if layer.type == 'BatchNorm':
        #         name2 = self.layer_names[i]
        #         bottom_name2 = self.net.bottom_names[name2][0]
        #         if bottom_name2 == bottom_name:
        #             # scaling and BN share the same bottom, should merge
        #             return {name2 + '/beta': param[1].data,
        #                     name2 + '/gamma': param[0].data }
        return {name + '/beta': param[1].data,
                name + '/gamma': param[0].data}
        # assume this scaling layer is part of some BN
        # raise ValueError()


def load_caffe(model_desc, model_file):
    """
    return a dict of params
    """
    import caffe
    caffe.set_mode_cpu()
    net = caffe.Net(model_desc, model_file, caffe.TEST)
    param_dict = CaffeLayerProcessor(net).process()
    return param_dict

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('weights')
    parser.add_argument('output')
    args = parser.parse_args()
    ret = load_caffe(args.model, args.weights)

    # pdb.set_trace()

    import numpy as np
    np.save(args.output, ret)
