# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.timer import Timer
import pickle
import numpy as np
import os
import glob
import time

from model.config import cfg
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


class SolverWrapper(object):
    """
      A wrapper class for the training process
    """

    def __init__(self, sess, network, dataset, output_dir, tbdir):
        self.sess = sess
        self.net = network()
        self.dataset = dataset()
        self.output_dir = output_dir
        self.tbdir = tbdir
        # Simply put '_val' at the end to save the summaries from the validation set
        self.tbvaldir = tbdir + '_val'
        if not os.path.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)

    def snapshot(self, sess, iter_num):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter_num) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter_num) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def train_model(self, sess, max_iters):
        # Determine different scales for anchors, see paper
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)
            # Build the main computation graph
            loss = self.net.create_architecture('TRAIN', tag='default')

            # Set learning rate and momentum
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(lr, beta1=cfg.TRAIN.BETA1, beta2=cfg.TRAIN.BETA2)

            # Compute the gradients wrt the loss
            gvs = self.optimizer.compute_gradients(loss)

            # Clip and norm gradients
            gvs = [(tf.clip_by_norm(grad, cfg.TRAIN.CLIP_NORM), var) for grad, var in gvs]
            gvs = [(tf.clip_by_value(grad, cfg.TRAIN.CLIP_MIN_VAL, cfg.TRAIN.CLIP_MAX_VAL),
                   var) for grad, var in gvs]

            # Apply graidents
            train_op = self.optimizer.apply_gradients(gvs)

            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tbvaldir)

        # Find previous snapshots if there is any to restore from
        sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redstr = '_iter_{:d}.'.format(cfg.TRAIN.STEPSIZE + 1)
        sfiles = [ss.replace('.meta', '') for ss in sfiles]
        sfiles = [ss for ss in sfiles if redstr not in ss]

        nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        nfiles = [nn for nn in nfiles if redstr not in nn]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        np_paths = nfiles
        ss_paths = sfiles

        last_snapshot_iter = 0
        if lsf == 0:
            variables = tf.global_variables()
            # Initialize all variables first
            sess.run(tf.variables_initializer(variables, name='init'))
        else:
            # Get the most recent snapshot and restore
            ss_paths = [ss_paths[-1]]
            np_paths = [np_paths[-1]]

            print('Restoring model snapshots from {:s}'.format(sfiles[-1]))
            variables = tf.global_variables()
            sess.run(tf.variables_initializer(variables, name='init'))

            var_keep_dic = self.get_variables_in_checkpoint_file(str(sfiles[-1]))
            variables_to_restore = []
            for v in variables:
                if v.name.split(':')[0] in var_keep_dic:
                    print('Variables restored: %s' % v.name)
                    variables_to_restore.append(v)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, str(sfiles[-1]))
            print('Restored.')
            # Needs to restore the other hyperparameters/states for training, (TODO xinlei) I have
            # tried my best to find the random states so that it can be recovered exactly
            # However the Tensorflow state is currently not available
            with open(str(nfiles[-1]), 'rb') as fid:
                st0 = pickle.load(fid)
                last_snapshot_iter = pickle.load(fid)

                np.random.set_state(st0)

                # Set the learning rate, only reduce once
                if last_snapshot_iter > cfg.TRAIN.STEPSIZE:
                    sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
                else:
                    sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))

        timer = Timer()
        iters = last_snapshot_iter + 1
        last_summary_time = time.time()
        while iters < int(max_iters) + 1:
            # Learning rate
            if iters == cfg.TRAIN.STEPSIZE + 1:
                # Add snapshot here before reducing the learning rate
                self.snapshot(sess, iters)
                sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))

            timer.tic()
            # Get training data, one batch at a time
            x_labeled, x_unlabeled, y_labeled = self.dataset.forward()

            now = time.time()
            if now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                # Compute the graph with summary
                l_qa, l_qz, l_pa, l_pz, l_px, l_py, u_qa, u_qz, u_pa, u_pz, u_px, u_py, lb_u, lb_l, loss, lb_l, lb_u,\
                summary = self.net.train_step_with_summary(sess, x_labeled, x_unlabeled, y_labeled, train_op)
                self.writer.add_summary(summary, float(iters))
                last_summary_time = now
            else:
                # Compute the graph without summary
                l_qa, l_qz, l_pa, l_pz, l_px, l_py, u_qa, u_qz, u_pa, u_pz, u_px, u_py, lb_u, lb_l, loss, lb_l, lb_u =\
                    self.net.train_step(sess, x_labeled, x_unlabeled, y_labeled, train_op)
            timer.toc()

            # Display training information
            if iters % (cfg.TRAIN.DISPLAY) == 0:
                print('iter: %d / %d, total loss: %.6f\n >>> lb_l: %.6f\n >>> lb_u: %.6f\n >>> l_qa: %.6f\n >>> l_qz: %.6f\n >>> l_pa: %.6f\n >>> l_pz: %.6f\n >>> l_px: %.6f\n >>> l_py: %.6f\n >>> u_qa: %.6f\n >>> u_qz: %.6f\n >>> l_pa: %.6f\n >>> u_pz: %.6f\n >>> u_px: %.6f\n >>> u_py: %.6f\n >>> lr: %f' %
                      (iters, int(max_iters),  loss, lb_l, lb_u, l_qa, l_qz, l_pa, l_pz, l_px, l_py, u_qa, u_qz,
                       u_pa, u_pz, u_px, u_py, lr.eval()))
                print('speed: {:.3f}s / iter'.format(timer.average_time))
            if iters % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iters
                snapshot_path, np_path = self.snapshot(sess, iters)
                np_paths.append(np_path)
                ss_paths.append(snapshot_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
                    for c in range(to_remove):
                        nfile = np_paths[0]
                        os.remove(str(nfile))
                        np_paths.remove(nfile)

                if len(ss_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
                    for c in range(to_remove):
                        sfile = ss_paths[0]
                        # To make the code compatible to earlier versions of Tensorflow,
                        # where the naming tradition for checkpoints are different
                        try:
                            if os.path.exists(str(sfile)):
                                os.remove(str(sfile))
                            else:
                                os.remove(str(sfile + '.data-00000-of-00001'))
                                os.remove(str(sfile + '.index'))
                        except FileNotFoundError:
                            print('Could not find file to remove. Moving on ..')
                        sfile_meta = sfile + '.meta'
                        os.remove(str(sfile_meta))
                        ss_paths.remove(sfile)

            iters += 1

        if last_snapshot_iter != iters - 1:
            self.snapshot(sess, iters - 1)

        self.writer.close()
        self.valwriter.close()


def train_net(network, dataset, output_dir, tb_dir, max_iters=40000):
    """Train a Fast R-CNN network."""
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        # from tensorflow.python import debug as tf_debug

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sw = SolverWrapper(sess, network, dataset, output_dir, tb_dir)
        print('Solving...')
        sw.train_model(sess, max_iters)
        print('done solving')
