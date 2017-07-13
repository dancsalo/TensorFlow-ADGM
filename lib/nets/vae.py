# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from model.config import cfg
import math

bottleneck = resnet_v1.bottleneck

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        # NOTE 'is_training' here does not work because inside resnet it gets reset:
        # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': cfg.RESNET.BN_TRAIN,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=initializers.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


@slim.add_arg_scope
def bottleneck_trans_same(inputs, depth, depth_bottleneck, stride, rate=1,
                     outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN after convolutions.
    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.
    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.
    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_trans', [inputs]) as sc:
        shortcut = slim.conv2d_transpose(inputs, depth, 3, stride=stride,
                                         activation_fn=None, scope='shortcut', padding='SAME')

        residual = slim.conv2d_transpose(inputs, depth_bottleneck, [1, 1], stride=1,
                                         scope='conv1_trans')
        residual = slim.conv2d_transpose(residual, depth_bottleneck, 3, stride=stride, scope='conv2', padding='SAME')
        residual = slim.conv2d_transpose(residual, depth, [1, 1], stride=1,
                                         activation_fn=None, scope='conv3_trans')
        output = tf.nn.relu(shortcut + residual)
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)

@slim.add_arg_scope
def bottleneck_trans_valid(inputs, depth, depth_bottleneck, stride, rate=1,
                     outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN after convolutions.
    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.
    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.
    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_trans', [inputs]) as sc:
        shortcut = slim.conv2d_transpose(inputs, depth, 3, stride=stride,
                                         activation_fn=None, scope='shortcut', padding='VALID')

        residual = slim.conv2d_transpose(inputs, depth_bottleneck, [1, 1], stride=1,
                                         scope='conv1_trans')
        residual = slim.conv2d_transpose(residual, depth_bottleneck, 3, stride=stride, scope='conv2', padding='VALID')
        residual = slim.conv2d_transpose(residual, depth, [1, 1], stride=1,
                                         activation_fn=None, scope='conv3_trans')

        output = tf.nn.relu(shortcut + residual)
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


class Vae:
    def __init__(self):

        self._num_classes = cfg.NUM_CLASSES
        self._batch_size = cfg.TRAIN.BATCH_SIZE
        self._latent_size = 128
        self._hidden_size = 256

        self._x_labeled = tf.placeholder(tf.float32, shape=[self._batch_size, 28, 28, 1])
        self._x_unlabeled = tf.placeholder(tf.float32, shape=[self._batch_size, 28, 28, 1])
        self._x = tf.concat([self._x_labeled, self._x_unlabeled], 0)
        self._y_labeled = tf.placeholder(tf.float32, shape=[self._batch_size, self._num_classes])

        self._losses = {}

        self._initializer = self.define_initializer()
        self._blocks_encoder = [resnet_utils.Block('block4', bottleneck, [(256, 128, 1)] * 3)]
        self._blocks_decoder_valid = [resnet_utils.Block('block5', bottleneck_trans_valid,
                                                         [(128, 128, 1), (256, 128, 2)])]
        self._blocks_decoder_same = [resnet_utils.Block('block5', bottleneck_trans_same,
                                                        [(128, 64, 2), (128, 64, 2)])]
        self._resnet_scope = 'resnet_v1_%d' % 101

        x_unlabeled_tiled = tf.tile(self._x_unlabeled, [self._num_classes, 1, 1, 1])  # (100, 256) --> (2100, 256)
        self.outputs = {'labeled': {'x_in': self._x_labeled}, 'unlabeled': {'x_in': x_unlabeled_tiled}}

    def add_losses(self):
        # Losses
        lb_l = self._calc_lb('labeled')
        self._losses['lb_l'] = tf.reduce_mean(lb_l)
        lb_u = self._calc_lb('unlabeled')
        self._losses['lb_u'] = tf.reduce_mean(lb_u)
        elbo = tf.reduce_mean(tf.concat([lb_l, lb_u], 0))
        self._losses['lb'] = elbo
        return elbo

    def build_network(self):
        # Q Networks
        q_x = self.encoder(self._x)  # Out of here comes (2200, 256)
        q_z_x = self.gaussian_stochastic(q_x, self._latent_size, 'q_z')

        # P Networks
        p_x = self.decoder(q_z_x)
        x_hat = self.bernoulli_stochastic(p_x, 1, 'p_x')

        # Image summaries
        tf.summary.image('x_in_label', self._x_labeled)
        tf.summary.image('xhat', x_hat)

    def create_architecture(self, mode, tag=None):

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        biases_regularizer = weights_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            self.build_network()

        elbo = self.add_losses()
        self._summary_op = tf.summary.merge_all()
        return elbo

    def _calc_lb(self, data_type):

        outputs = self.outputs[data_type]

        log_qz = self.gaussian_log_density(outputs['q_z_sample'], outputs['q_z_mu'], outputs['q_z_sigma2'])
        log_pz = self.standard_gaussian_log_density(outputs['q_z_sample'])
        log_px = self.bernoulli_log_density(outputs['x_in'], outputs['p_x_mu'])

        self._losses['{}_log_qz'.format(data_type)] = tf.reduce_mean(log_qz)
        self._losses['{}_log_pz'.format(data_type)] = tf.reduce_mean(log_pz)
        self._losses['{}_log_px'.format(data_type)] = tf.reduce_mean(log_px)

        lb = tf.reduce_sum(tf.squared_difference(outputs['x_in'], outputs['p_x_mu']), axis=[1,2,3])
        lb -= 0.5 * tf.reduce_sum(1.0 - tf.square(outputs['q_z_mu']) - 0.5 * tf.square(outputs['q_z_sigma2']) +
            tf.log(outputs['q_z_sigma2'] + 1e-12), axis=[1,2,3])

        tf.summary.scalar('{}_log_qz'.format(data_type), tf.reduce_mean(log_qz))
        tf.summary.scalar('{}_log_pz'.format(data_type), tf.reduce_mean(log_pz))
        tf.summary.scalar('{}_log_px'.format(data_type), tf.reduce_mean(log_px))

        return lb

    def split_labeled_unlabeled(self, tensor, key):
        """
        :param tensor: input tensor of both labeled and unlabeled data concatenated together
        :return: list of 2 tensors with labeled and unlabeled portions
        """
        if tensor.get_shape()[0] % (self._num_classes + 1) == 0:
            all = tf.split(tensor, self._num_classes + 1)
            self.outputs['labeled'][key] = all[0]
            self.outputs['unlabeled'][key] = tf.concat(all[1:], 0)
        else:
            self.outputs['labeled'][key], self.outputs['unlabeled'][key] = tf.split(tensor, 2)

    def encoder(self, x):
        with tf.variable_scope('encoder'):
            net = resnet_utils.conv2d_same(x, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            x = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
            x, _ = resnet_v1.resnet_v1(x,
                                        self._blocks_encoder,
                                        global_pool=False,
                                        include_root_block=False,
                                        scope=self._resnet_scope)
            x = tf.reduce_mean(x, axis=[1, 2])
            x_features_labeled, x_features_unlabeled = tf.split(x, 2)

        x_features_tiled = tf.tile(x_features_unlabeled, [self._num_classes, 1])  # (100, 256) --> (2100, 256)
        x_features = tf.concat([x_features_labeled, x_features_tiled], 0) # (2100, 256) --> (2200, 256)
        return x_features

    def decoder(self, p_x):
        with tf.variable_scope('valid'):
            p_x, _ = resnet_v1.resnet_v1(p_x,
                                          self._blocks_decoder_valid,
                                          global_pool=False,
                                          include_root_block=False,
                                          scope=self._resnet_scope)
        with tf.variable_scope('same'):
            p_x, _ = resnet_v1.resnet_v1(p_x,
                                          self._blocks_decoder_same,
                                          global_pool=False,
                                          include_root_block=False,
                                          scope=self._resnet_scope)
        return p_x

    def bernoulli_stochastic(self, input_tensor, num_maps, scope):
        """
        :param inputs_list: list of Tensors to be added and input into the block
        :return: random variable single draw, mean, standard deviation, and intermediate representation
        """
        with tf.variable_scope(scope):
            input_tensor = tf.expand_dims(tf.expand_dims(input_tensor, 1), 1) if len(input_tensor.get_shape()) != 4 \
                else input_tensor
            intermediate = slim.conv2d(input_tensor, self._hidden_size, [1, 1], weights_initializer=self._initializer,
                                       scope='conv1')
            mean = slim.conv2d(intermediate, num_maps, [1, 1], weights_initializer=self._initializer,
                               activation_fn=tf.sigmoid, scope='mean')
            tf.summary.histogram(scope + '_mu', mean)
            rv_single_draw = tf.where(tf.random_uniform(tf.shape(mean)) - mean < 0, tf.ones(tf.shape(mean)),
                                      tf.zeros(tf.shape(mean)))

        self.split_labeled_unlabeled(mean, '{}_mu'.format(scope))
        self.split_labeled_unlabeled(rv_single_draw, '{}_sample'.format(scope))
        return rv_single_draw

    def gaussian_stochastic(self, input_tensor, num_maps, scope):
        """
        :param inputs_list: list of Tensors to be added and input into the block
        :return: random variable single draw, mean, standard deviation, and intermediate representation
        """
        with tf.variable_scope(scope):
            input_tensor = tf.expand_dims(tf.expand_dims(input_tensor, 1), 1) if len(input_tensor.get_shape()) != 4 \
                else input_tensor
            intermediate = slim.conv2d(input_tensor, self._hidden_size, [1, 1], weights_initializer=self._initializer,
                                       scope='conv1')
            mean = slim.conv2d(intermediate, num_maps, [1, 1], weights_initializer=self._initializer,
                               activation_fn=None, scope='mean')
            sigma2 = tf.nn.softplus(
                slim.conv2d(intermediate, num_maps, [1, 1], weights_initializer=self._initializer,
                            activation_fn=None, scope='sigma2'))
            tf.summary.histogram(scope + '_mean', mean)
            tf.summary.histogram(scope + '_sigma2', sigma2)
            rv_single_draw = mean + tf.sqrt(sigma2) * tf.random_normal(tf.shape(mean))

        self.split_labeled_unlabeled(mean, '{}_mu'.format(scope))
        self.split_labeled_unlabeled(sigma2, '{}_sigma2'.format(scope))
        self.split_labeled_unlabeled(rv_single_draw, '{}_sample'.format(scope))
        return rv_single_draw

    def get_summary(self, sess, x_labeled, x_unlabeled, y_labeled):
        feed_dict = {self._x_labeled: x_labeled, self._x_unlabeled: x_unlabeled, self._y_labeled: y_labeled}
        summary = sess.run(self._summary_op, feed_dict=feed_dict)
        return summary

    def train_step(self, sess, x_labeled, x_unlabeled, y_labeled, train_op):
        feed_dict = {self._x_labeled: x_labeled, self._x_unlabeled: x_unlabeled, self._y_labeled: y_labeled}
        data_types = ['labeled', 'unlabeled']
        losses = ['log_qz', 'log_pz', 'log_px']
        all_losses = list()
        for d in data_types:
            for l in losses:
                all_losses.append(self._losses['{}_{}'.format(d, l)])
        all_outputs = all_losses + [self._losses['lb_u'], self._losses['lb_l'], self._losses['lb'], train_op]
        self.l_qz, self.l_pz, self.l_px, self.u_qz, self.u_pz, self.u_px, self.lb_u, self.lb_l, self.loss, _ =\
            sess.run(all_outputs, feed_dict=feed_dict)

    def train_step_with_summary(self, sess, x_labeled, x_unlabeled, y_labeled, train_op):
        feed_dict = {self._x_labeled: x_labeled, self._x_unlabeled: x_unlabeled, self._y_labeled: y_labeled}
        data_types = ['labeled', 'unlabeled']
        losses = ['log_qz', 'log_pz', 'log_px']
        all_losses = list()
        for d in data_types:
            for l in losses:
                all_losses.append(self._losses['{}_{}'.format(d, l)])
        all_outputs = all_losses + [self._losses['lb_u'], self._losses['lb_l'], self._losses['lb'], self._summary_op,
                                    train_op]
        self.l_qz, self.l_pz, self.l_px, self.u_qz, self.u_pz, self.u_px, self.lb_u, self.lb_l, self.loss, summary, _ =\
            sess.run(all_outputs, feed_dict=feed_dict)
        return summary

    def print_losses(self):
        print('total loss: %.6f\n >>> lb_l: %.6f\n >>> lb_u: %.6f\n >>> l_qz: %.6f\n'
              '>>> l_pz: %.6f\n >>> l_px: %.6f\n >>> u_qz: %.6f\n >>> u_pz: %.6f\n >>> u_px: %.6f'
              % (self.loss, self.lb_l, self.lb_u, self.l_qz, self.l_pz, self.l_px, self.u_qz, self.u_pz, self.u_px))

    @staticmethod
    def standard_gaussian_log_density(x):
        c = - 0.5 * math.log(2 * math.pi)
        density = c - tf.square(x) / 2
        return -density

    @staticmethod
    def gaussian_log_density(x, mu, sigma2):
        c = - 0.5 * math.log(2 * math.pi)
        density = c - tf.log(sigma2) / 2 - tf.squared_difference(x, mu) / (2 * sigma2)
        return -density


    @staticmethod
    def bernoulli_log_density(x, mu):
        density = x * tf.log(mu + 1e-12) + (1 - x) * tf.log(1 - mu + 1e-12)
        return -density

    @staticmethod
    def define_initializer():
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        return initializer
