import os
import datetime
import time
import sys
import numpy as np
import tensorflow as tf
import threading
import time
from visualization import (
    LearningViewer,
    GameOfLifeViewer,
    ParityViewer
)
from mlp.layer_helpers import (
    fully_connected_layer,
    reshape_layer
)
from simple_io import folder_create
import random
random.seed(1221)
from learned_module_wrapper import (
    gol_fun,
    gol_fun_counter
)
import copy


class gol_evol(threading.Thread):

    img_height = 3
    img_width = 3
    img_channels = 1

    def __init__(self, steps=5000, exp_dir='out', checkpoint_dir='checkpoints', data=None,
                 input_dim=9, hl_unit_no=100, out_dim=9, name_scope=''):
        super().__init__()

        print("Initializing gol Thread...")
        self.viewer = LearningViewer(
            pos_arg='right',
            name='gol',
            scholar=self
        )
        self.name_scope = name_scope
        self.step = 0
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hl_unit_no = hl_unit_no
        self.__define_network()

        self.file_list = []

        self.gol_data = data

        # create session and run training loop
        self.sess = tf.Session()#(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.train_step_no = 0
        # self.saver.restore(self.sess, tf.train.latest_checkpoint('out/readout/checkpoints/'))

        self.batch_prev_step = 0
        self.retrain_prev_step = 0
        self.epoch = 0

        # -----------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------

        # add summary operations
        tf.summary.scalar('error', self.error)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary_op = tf.summary.merge_all()

        # create objects for writing summaries and checkpoints during training
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_dir = exp_dir
        self.checkpoint_dir = os.path.join(self.exp_dir, checkpoint_dir)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.train_writer = tf.summary.FileWriter(os.path.join(self.exp_dir, 'train-summaries'))
        self.valid_writer = tf.summary.FileWriter(os.path.join(self.exp_dir, 'valid-summaries'))
        self.saver = tf.train.Saver(max_to_keep=3)

        self.num_epochs = steps

        # create arrays to store run train / valid set stats
        self.train_accuracy = np.zeros(self.num_epochs)
        self.train_error = np.zeros(self.num_epochs)
        self.valid_accuracy = np.zeros(self.num_epochs)
        self.valid_error = np.zeros(self.num_epochs)

        print("gol Thread Initialized.")

    def __define_network(self):
        #-----------------------------------------------------------------------------------------------------
        #---------------------- define model graph------------------------------------------------------------
        num_classes = self.out_dim  # combinatorics
        tf.reset_default_graph()
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.inputs = tf.placeholder(tf.float32, [None,
                                                  self.input_dim], 'inputs')
        self.targets = tf.placeholder(tf.float32, [None, num_classes], 'targets')

        # ----------------------------- NETWORK DEFINITION --------------------------------------------------
        # put conv layers on cpu because too big for gtx1050
        # with tf.device('/cpu:0'):
        nonlinearity = tf.nn.tanh
        with tf.name_scope(self.name_scope + 'fully_connected_layer1'):
            fully_connected_layer1 = fully_connected_layer(self.inputs, self.input_dim, self.hl_unit_no, nonlinearity=nonlinearity)
        with tf.name_scope(self.name_scope + 'fully_connected_layer2'):
            fully_connected_layer2 = fully_connected_layer(fully_connected_layer1, self.hl_unit_no, self.hl_unit_no, nonlinearity=nonlinearity)
        with tf.name_scope(self.name_scope + 'output-layer'):
            outputs = fully_connected_layer(fully_connected_layer2, self.hl_unit_no, num_classes, nonlinearity=tf.identity)

        # ------------ define error computation -------------
        with tf.name_scope(self.name_scope +'predictions'):
            # self.predictions = np.softmax(outputs, 1)
            self.predictions = tf.sigmoid(outputs)

        with tf.name_scope(self.name_scope +'error'):
            # vars = tf.trainable_variables()
            self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=self.targets)) #+
                                   # tf.add_n([tf.nn.l2_loss(v) for v in vars
                                   #           if 'bias' not in v.name]) * 0.001)
        with tf.name_scope(self.name_scope +'accuracy'):
            # self.accuracy = tf.reduce_mean(tf.cast(
            #         tf.equal(tf.argmax(outputs, 1), tf.argmax(self.targets, 1)),
            #         tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.greater(self.predictions, 0.5),
                         tf.equal(self.targets, 1.0)), 'float'))
        # --- define training rule ---
        with tf.name_scope(self.name_scope +'train'):
            self.train_step = tf.train.RMSPropOptimizer(learning_rate=0.0005).minimize(self.error)

        self.saver = tf.train.Saver(max_to_keep=3)

    def run(self):
        sol_count = 0
        while self.epoch < self.num_epochs:
            # -- trigger learning when found data
            # un-pack
            for input_batch, target_batch in self.gol_data:
                # -- do train step with current batch
                _, summary, batch_error, batch_acc, batch_predictions = self.sess.run(
                    [self.train_step, self.summary_op, self.error, self.accuracy, self.predictions],
                    feed_dict={self.inputs: input_batch, self.targets: target_batch,
                               self.batch_size: target_batch.shape[0]})

                # add symmary and accumulate stats
                self.train_writer.add_summary(summary, self.train_step_no)
                self.train_error[self.epoch] = batch_error
                self.train_accuracy[self.epoch] = batch_acc
                self.train_step_no += 1
                print('Epoch {0:02d}: err(train)={1:.6f} acc(train)={2:.6f}'
                      .format(self.epoch + 1, self.train_error[self.epoch], self.train_accuracy[self.epoch]))
                self.viewer.show_frames(
                    train_error=self.train_error[:self.epoch],
                    train_accuracy=self.train_accuracy[:self.epoch]
                )
                self.epoch += 1
                if batch_acc == 1.:
                    sol_count += 1

            if self.epoch % 100 == 0:
                # checkpoint model variables
                self.save()

            if sol_count > 20:
                break

        print("Ending Run!")
        self.release()


    def save(self):
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), self.train_step_no)
        sys.stdout.flush()

    def release(self):
        # save frozen model
        output_graph = self.checkpoint_dir + "/frozen_model.pb"

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,  # The session is used to retrieve the weights
            self.sess._graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
            ['inputs', self.name_scope + 'predictions/Sigmoid']  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print("Closing and Releasing")

        # close writer and session objects
        self.train_writer.close()
        self.valid_writer.close()
        self.sess.close()

        # save run stats to a .npz file
        np.savez_compressed(
            os.path.join(self.exp_dir, 'run.npz'),
            train_error=self.train_error,
            train_accuracy=self.train_accuracy,
            valid_error=self.valid_error,
            valid_accuracy=self.valid_accuracy
        )
        # print("Done, Run ended.")
        # self.join()


class gol_read_out(threading.Thread):

    img_height = 3
    img_width = 3
    img_channels = 1

    def __init__(self, steps=5000, exp_dir='out', checkpoint_dir='checkpoints', data=None, hl_unit_no=400):
        super().__init__()

        print("Initializing gol Thread...")
        self.viewer = LearningViewer(
            pos_arg='right',
            name='gol',
            scholar=self
        )
        self.step = 0
        self.hl_unit_no = hl_unit_no
        self.__define_network()

        self.file_list = []

        self.gol_data = data

        # create session and run training loop
        self.sess = tf.Session()#(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.train_step_no = 0
        self.saver.restore(self.sess, tf.train.latest_checkpoint('out/readout/checkpoints/'))

        self.batch_prev_step = 0
        self.retrain_prev_step = 0
        self.epoch = 0

        # -----------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------

        # add summary operations
        tf.summary.scalar('error', self.error)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary_op = tf.summary.merge_all()

        # create objects for writing summaries and checkpoints during training
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_dir = exp_dir
        self.checkpoint_dir = os.path.join(self.exp_dir, checkpoint_dir)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.train_writer = tf.summary.FileWriter(os.path.join(self.exp_dir, 'train-summaries'))
        self.valid_writer = tf.summary.FileWriter(os.path.join(self.exp_dir, 'valid-summaries'))
        self.saver = tf.train.Saver(max_to_keep=3)

        self.num_epochs = steps

        # create arrays to store run train / valid set stats
        self.train_accuracy = np.zeros(self.num_epochs)
        self.train_error = np.zeros(self.num_epochs)
        self.valid_accuracy = np.zeros(self.num_epochs)
        self.valid_error = np.zeros(self.num_epochs)

        print("gol Thread Initialized.")

    def __define_network(self):
        #-----------------------------------------------------------------------------------------------------
        #---------------------- define model graph------------------------------------------------------------
        num_classes = 9  # combinatorics
        tf.reset_default_graph()
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.inputs = tf.placeholder(tf.float32, [None,
                                                  19
                                                  ], 'inputs')
        self.targets = tf.placeholder(tf.float32, [None, 9], 'targets')

        # ----------------------------- NETWORK DEFINITION --------------------------------------------------
        # put conv layers on cpu because too big for gtx1050
        # with tf.device('/cpu:0'):
        # nonlinearity =
        with tf.name_scope('output-layer'):
            read_out_layer = fully_connected_layer(self.inputs, 19, num_classes)
        with tf.name_scope('output-layer'):
            outputs = fully_connected_layer(read_out_layer, num_classes, num_classes, nonlinearity=tf.identity)

        # ------------ define error computation -------------
        with tf.name_scope('predictions'):
            # self.predictions = np.softmax(outputs, 1)
            self.predictions = tf.sigmoid(outputs) #, name='readout_output')

        with tf.name_scope('error'):
            # vars = tf.trainable_variables()
            self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=self.targets)) #+
                                   # tf.add_n([tf.nn.l2_loss(v) for v in vars
                                   #           if 'bias' not in v.name]) * 0.001)
        with tf.name_scope('accuracy'):
            # self.accuracy = tf.reduce_mean(tf.cast(
            #         tf.equal(tf.argmax(outputs, 1), tf.argmax(self.targets, 1)),
            #         tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.greater(self.predictions, 0.5),
                         tf.equal(self.targets, 1.0)), 'float'))
        # --- define training rule ---
        with tf.name_scope('train'):
            self.train_step = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(self.error)

        self.saver = tf.train.Saver(max_to_keep=3)


    def run(self):
        sol_count = 0

        print('Loading learned model for forward function...')
        self.forward_fun = gol_fun('out/forward/checkpoints/frozen_model.pb', output_tensor_names=['prefix/predictions/Sigmoid:0'])
        print('DONE...')

        print('Loading learned model for identity function...')
        self.identity_fun = gol_fun('out/identity/checkpoints/frozen_model.pb', output_tensor_names=['prefix/predictions/Sigmoid:0'])
        print('DONE!')

        print('Loading learned model for count function...')
        self.counter_fun = gol_fun_counter('out/counter/checkpoints/frozen_model.pb',
                                   output_tensor_names=['prefix/predictions/count:0',
                                                        'prefix/predictions/decision:0'])
        print('DONE!')

        while self.epoch < self.num_epochs:
            # -- trigger learning when found data
            # un-pack
            for input_batch, target_batch in self.gol_data:
                # -- do train step with current batch
                #todo: change input batch into output batch

                forward_mid_batch, _ = self.forward_fun.predict(input_batch)  # eventually shold be (20x20xbatch_size)
                identity_mid_batch, _ = self.identity_fun.predict(input_batch)  # eventually shold be (20x20xbatch_size)
                _, decisions_mid_batch_pos = self.counter_fun.predict(
                    np.array(random.sample(range(10000), 512)).reshape(-1, 1)
                )
                counts_mid_batch_neg, decisions_mid_batch_neg = self.counter_fun.predict(
                    np.array(random.sample(range(-1, -10000, -1), 512)).reshape(-1, 1)
                )
                forward_mid_batch = np.append(forward_mid_batch, forward_mid_batch, axis=1)
                identity_mid_batch = np.append(identity_mid_batch, identity_mid_batch, axis=1)
                # counts_mid_batch = np.append(counts_mid_batch_pos, counts_mid_batch_neg, axis=0)
                decisions_mid_batch = np.append(decisions_mid_batch_pos, decisions_mid_batch_neg, axis=0)

                new_input_batch = np.concatenate(
                    (forward_mid_batch[0, :], identity_mid_batch[0, :], decisions_mid_batch),
                    axis=1
                )
                new_tagets = np.append(target_batch, input_batch, axis=0)

                _, summary, batch_error, batch_acc, batch_predictions = self.sess.run(
                    [self.train_step, self.summary_op, self.error, self.accuracy, self.predictions],
                    feed_dict={self.inputs: new_input_batch, self.targets: new_tagets,
                               self.batch_size: target_batch.shape[0]})

                # add symmary and accumulate stats
                self.train_writer.add_summary(summary, self.train_step_no)
                self.train_error[self.epoch] = batch_error
                self.train_accuracy[self.epoch] = batch_acc
                self.train_step_no += 1
                print('Epoch {0:02d}: err(train)={1:.6f} acc(train)={2:.6f}'
                      .format(self.epoch + 1, self.train_error[self.epoch], self.train_accuracy[self.epoch]))
                self.viewer.show_frames(
                    train_error=self.train_error[:self.epoch],
                    train_accuracy=self.train_accuracy[:self.epoch]
                )
                self.epoch += 1

                if batch_acc == 1.:  # or (self.gol_data.function is not 'forward' and batch_acc > .950520):
                    sol_count += 1
                else:
                    sol_count = 0

            if self.epoch % 100 == 0:
                # checkpoint model variables
                self.save()

            if sol_count > 20:
            # if self.epoch > 2:
                break

        print("Ending Run!")
        self.release()


    def save(self):
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), self.train_step_no)
        sys.stdout.flush()

    def release(self):
        # save frozen model
        output_graph = self.checkpoint_dir + "/frozen_model.pb"

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,  # The session is used to retrieve the weights
            self.sess._graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
            ['inputs', 'predictions/Sigmoid']  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print("Closing and Releasing")

        # close writer and session objects
        self.train_writer.close()
        self.valid_writer.close()
        self.sess.close()

        # save run stats to a .npz file
        np.savez_compressed(
            os.path.join(self.exp_dir, 'run.npz'),
            train_error=self.train_error,
            train_accuracy=self.train_accuracy,
            valid_error=self.valid_error,
            valid_accuracy=self.valid_accuracy
        )
        # print("Done, Run ended.")
        # self.join()


class gol_counter(threading.Thread):

    img_height = 1
    img_width = 1
    img_channels = 1

    def __init__(self, steps=5000, exp_dir='out', checkpoint_dir='checkpoints', data=None, name_scope='counter'):
        super().__init__()

        print("Initializing gol Thread...")
        self.viewer = LearningViewer(
            pos_arg='right',
            name='gol',
            scholar=self
        )
        self.name_scope = name_scope
        self.step = 0
        self.__define_network()

        self.file_list = []

        self.gol_data = data

        # create session and run training loop
        self.sess = tf.Session()#(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.train_step_no = 0

        self.batch_prev_step = 0
        self.retrain_prev_step = 0
        self.epoch = 0

        # -----------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------

        # add summary operations
        tf.summary.scalar('error', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary_op = tf.summary.merge_all()

        # create objects for writing summaries and checkpoints during training
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_dir = exp_dir
        self.checkpoint_dir = os.path.join(self.exp_dir, checkpoint_dir)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.train_writer = tf.summary.FileWriter(os.path.join(self.exp_dir, 'train-summaries'))
        self.train_writer.add_graph(self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(self.exp_dir, 'valid-summaries'))
        self.saver = tf.train.Saver(max_to_keep=3)

        self.num_epochs = steps

        # create arrays to store run train / valid set stats
        self.train_accuracy = np.zeros(self.num_epochs)
        self.train_error = np.zeros(self.num_epochs)
        self.valid_accuracy = np.zeros(self.num_epochs)
        self.valid_error = np.zeros(self.num_epochs)

        print("gol Thread Initialized.")

    def __define_network(self):
        #-----------------------------------------------------------------------------------------------------
        #---------------------- define model graph------------------------------------------------------------
        num_outputs = 1
        tf.reset_default_graph()
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.inputs = tf.placeholder(tf.float32, [None, 1], 'inputs')
        self.targets = tf.placeholder(tf.float32, [None, 1], 'targets')

        # ----------------------------- NETWORK DEFINITION --------------------------------------------------
        # put conv layers on cpu because too big for gtx1050
        # with tf.device('/cpu:0'):
        # nonlinearity = tf.nn.relu
        # with tf.name_scope(self.name_scope +'fully_connected_layer1'):
        #     fully_connected_layer1 = fully_connected_layer(self.inputs, 1, 1, nonlinearity=tf.identity)
        with tf.name_scope(self.name_scope +'middle-layer'):
            weights_mid = tf.Variable(np.array([1.]).astype(np.float32).reshape(-1, 1))
            biases_mid = tf.Variable(np.array([-1.]).astype(np.float32).reshape(-1, 1), 'biases')
            output_middle = tf.matmul(self.inputs, weights_mid) + biases_mid
        with tf.name_scope(self.name_scope +'output-layer-count'):
            weight_out1 = tf.Variable(np.array([1.]).astype(np.float32).reshape(-1, 1))
            output_count = tf.matmul(output_middle, weight_out1)

        with tf.name_scope(self.name_scope +'output-layer-decision'):
            weight_out2 = tf.Variable(np.array([1.]).astype(np.float32).reshape(-1, 1), 'weights' )
            bias2 = tf.Variable(np.array([0.]).astype(np.float32).reshape(-1, 1), 'biases')
            output_decision = tf.nn.sigmoid(tf.matmul(output_middle, weight_out2) + bias2)

        # ------------ define error computation -------------
        with tf.name_scope(self.name_scope +'predictions'):
            # self.predictions = np.softmax(outputs, 1)
            self.prediction_count = tf.identity(output_count, name='count')
            self.prediction_decision = tf.identity(output_decision, name='decision')
            # self.predictions = tf.add(tf.multiply(self.inputs, W), b)

        with tf.name_scope(self.name_scope +'error'):
            vars = tf.trainable_variables()
            self.loss = tf.reduce_mean(tf.pow(output_count - self.targets, 2))
            # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_count, labels=self.targets)) #+
            #                            tf.add_n([tf.nn.l2_loss(v) for v in vars
            #                                  if 'bias' not in v.name]) * 0.0001)
        with tf.name_scope(self.name_scope +'accuracy'):
            # self.accuracy = tf.reduce_mean(tf.cast(
            #         tf.equal(tf.argmax(outputs, 1), tf.argmax(self.targets, 1)),
            #         tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.to_int32(output_count), tf.to_int32(self.targets)), 'float'))

            # self.accuracy = tf.reduce_mean(tf.cast(
            #     tf.equal(tf.greater(self.predictions[1], 0.5),
            #              tf.greater(self.targets, 0.)), 'float'))

        # --- define training rule ---
        with tf.name_scope(self.name_scope +'train'):
            self.train_step = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(self.loss)
            # self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=3)

    def run(self):
        sol_count = 0
        while self.epoch < self.num_epochs:
            # -- trigger learning when found data
            # un-pack
            batch_acc = 0
            for input_batch, target_batch in self.gol_data:
                # -- do train step with current batch
                _, summary, batch_error, batch_acc, batch_pred1, batch_pred2= self.sess.run(
                    [self.train_step, self.summary_op, self.loss, self.accuracy, self.prediction_count, self.prediction_decision],
                    feed_dict={self.inputs: input_batch, self.targets: target_batch[:, 0].reshape(-1, 1),
                               self.batch_size: target_batch.shape[0]})

                # add symmary and accumulate stats
                self.train_writer.add_summary(summary, self.train_step_no)
                self.train_error[self.epoch] = batch_error
                self.train_accuracy[self.epoch] = batch_acc
                self.train_step_no += 1
            print('Epoch {0:02d}: err(train)={1:.6f} acc(train)={2:.6f}'
                  .format(self.epoch + 1, self.train_error[self.epoch], self.train_accuracy[self.epoch]))
            self.viewer.show_frames(
                train_error=self.train_error[:self.epoch],
                train_accuracy=self.train_accuracy[:self.epoch]
            )
            self.epoch += 1

            if batch_acc == 1.:  # or (self.gol_data.function is not 'forward' and batch_acc > .950520):
                sol_count += 1

            if self.epoch % 10 == 0:
                # checkpoint model variables
                self.save()

            if sol_count > 20:
            # if self.epoch > 2:
                break

        print("Ending Run!")
        self.release()


    def save(self):
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), self.train_step_no)
        sys.stdout.flush()

    def release(self):
        # save frozen model
        output_graph = self.checkpoint_dir + "/frozen_model.pb"

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,  # The session is used to retrieve the weights
            self.sess._graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
            ['inputs', 'predictions/count', 'predictions/decision']  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print("Closing and Releasing")

        # close writer and session objects
        self.train_writer.close()
        self.valid_writer.close()
        self.sess.close()

        # save run stats to a .npz file
        np.savez_compressed(
            os.path.join(self.exp_dir, 'run.npz'),
            train_error=self.train_error,
            train_accuracy=self.train_accuracy,
            valid_error=self.valid_error,
            valid_accuracy=self.valid_accuracy
        )
        # print("Done, Run ended.")
        # self.join()


# class representing the game of life game
class GameOfLife(object):
    step = 0
    board = None
    side_pad = 2

    def __init__(self, board=None, size=(100, 100), teaching_live=False, batch_size=200, live_data_path='backward/live_data/'):

        print('initializing game board...')
        if board is None:
            self.board_size = size
            board = np.zeros(size)
            self.padded_board = cropped_padder(board, size[0] + 2*self.side_pad, size[1] + 2*self.side_pad)
        else:
            self.board_size = board.shape
        # the padded board is necessary for the network convolutions
        self.padded_board = cropped_padder(board,
                                           self.board_size[0] + 2*self.side_pad,
                                           self.board_size[1] + 2*self.side_pad)
        print('DONE..')

        # initialize buffer for backward learning (currently not fully supported)
        self.teaching_live = teaching_live
        if self.teaching_live:
            self.batch_size = batch_size
            self.live_data_path = live_data_path
            folder_create(self.live_data_path)
            self._initialize_teaching()

        # viewer to view board live
        self.board_viewer = GameOfLifeViewer(window_name='Game Of Life Board', board_size=self.board_size)

        print('Loading learned model for forward function...')
        self.forward_fun = gol_fun('out/forward/checkpoints/frozen_model.pb', output_tensor_names=['prefix/predictions/Sigmoid:0'])
        print('DONE...')

        print('Loading learned model for identity function...')
        self.identity_fun = gol_fun('out/identity/checkpoints/frozen_model.pb', output_tensor_names=['prefix/predictions/Sigmoid:0'])
        print('DONE!')

        print('Loading learned model for count function...')
        self.counter_fun = gol_fun_counter('out/counter/checkpoints/frozen_model.pb',
                                   output_tensor_names=['prefix/predictions/count:0',
                                                        'prefix/predictions/decision:0'])
        print('DONE!')

        print('Loading learned model for readout function...')
        self.readout_fun = gol_fun('out/readout/checkpoints/frozen_model.pb', output_tensor_names=['prefix/predictions/Sigmoid:0'])
        print('DONE!')

    def _initialize_teaching(self):
            self.states = np.zeros((self.batch_size,) + self.board_size)
            self.next_states = np.zeros((self.batch_size,) + self.board_size)

    # function to step forward in time through the use of the forward learned network
    def step_forward(self):
        self.next_padded_board = np.zeros(self.padded_board.shape)
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                board_patch = self.padded_board[i+self.side_pad-1:i+self.side_pad+2,
                                                j+self.side_pad-1:j+self.side_pad+2]
                # prefict the next state of the network
                predictions, _ = self.forward_fun.predict(board_patch.reshape(1, -1))
                self.next_padded_board[i+self.side_pad, j+self.side_pad] = predictions[0][0, 4]

        if self.teaching_live:
            self.update_states()

        self.padded_board = copy.deepcopy(self.next_padded_board)
        self.step += 1

        return self.padded_board

    # identity network to stop the forward flow of the network
    def stop(self):
        self.next_padded_board = np.zeros(self.padded_board.shape)
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                board_patch = self.padded_board[i+self.side_pad-1:i+self.side_pad+2,
                                                j+self.side_pad-1:j+self.side_pad+2]
                predictions, _ = self.identity_fun.predict(board_patch.reshape(1, -1))
                self.next_padded_board[i+self.side_pad, j+self.side_pad] = predictions[0][0, 4]

        if self.teaching_live:
            self.update_states()

        self.padded_board = copy.deepcopy(self.next_padded_board)
        self.step += 1

        return self.padded_board

    # for loop implemented as a recursion by iterating output-input through the learned architecture -
    # it forwards the state of the board of "iteration" time steps
    def net_step(self, iterations):
        self.next_padded_board = np.zeros(self.padded_board.shape)
        current_iter = iterations
        # convolution - can be easily implemented in tensors as well
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                board_patch = self.padded_board[i+self.side_pad-1:i+self.side_pad+2,
                                                j+self.side_pad-1:j+self.side_pad+2]
                ## f(x)
                fx, _ = self.forward_fun.predict(board_patch.reshape(1, -1))
                ## identity
                Ix, _ = self.identity_fun.predict(board_patch.reshape(1, -1))
                ## controller
                current_iter, current_decision = self.counter_fun.predict(np.array([[iterations]]))
                ## readout
                predictions, _ = self.readout_fun.predict(np.concatenate((fx[0, :], Ix[0, :], current_decision), axis=1))
                self.next_padded_board[i + self.side_pad, j + self.side_pad] = predictions[0][0, 4]

        self.padded_board = copy.deepcopy(self.next_padded_board)
        text = text="counter: {}, control: {}".format(current_iter[0][0], int(current_decision[0][0]>.5))
        self.visualize(text=text)

        # recursive call simulating tensor connection output to input
        self.net_step(current_iter[0][0])

    def count(self, current_count):
        return self.counter_fun.predict(np.array(current_count).reshape(-1, 1))

    def visualize(self, text):
        self.board_viewer.show_frame(self.count, self.padded_board[self.side_pad:-self.side_pad,
                                                            self.side_pad:-self.side_pad], text=text)

    def update_states(self):
        self.states[self.step, :] = self.padded_board[self.side_pad:-self.side_pad,
                                                      self.side_pad:-self.side_pad]
        self.next_states[self.step, :] = self.next_padded_board[self.side_pad:-self.side_pad,
                                                                self.side_pad:-self.side_pad]

    def dump_files(self):
        if self.teaching_live and self.step % self.batch_size == 0:
            np.savez_compressed('{}{}{}'.format(self.live_data_path, 'batch_', self.step),
                                inputs=self.states,
                                targets=self.next_states)


class ParityBitChecker:

    def __init__ (self):
        # viewer to view parity live
        self.parity_viewer = ParityViewer(window_name='Parity Viewer')
        print('Loading learned model for parity checking function...')
        self.parity_fun = gol_fun('out/parity/checkpoints/frozen_model.pb', output_tensor_names=['prefix/predictions/Sigmoid:0'])
        print('DONE!')

        print('Loading learned model for count function...')
        self.counter_fun = gol_fun_counter('out/counter/checkpoints/frozen_model.pb',
                                   output_tensor_names=['prefix/predictions/count:0',
                                                        'prefix/predictions/decision:0'])
        print('DONE!')

        self.current_string = None
        self.current_results = None

    def visualize(self, step):
        self.parity_viewer.show_frame(step, bytes_string=self.current_string, results=self.current_results)

    def check_parity(self, byte_string, bytes_number=1):

        if self.current_string is None or self.current_results is None:
            self.current_string = byte_string
            self.current_results = np.zeros(byte_string.shape[0])

        current_iter, current_decision = self.counter_fun.predict(np.array([[bytes_number]]))
        parity_pass, probabilities = self.parity_fun.predict([byte_string[int(current_iter[0][0]), :]])
        self.current_results[int(current_iter[0][0])] = parity_pass[0][0][0]
        if current_decision[0][0] >= .5:
            self.visualize(current_iter[0][0])
            print("counter: {}, control: {}, decision: {}".format(current_iter[0][0], int(current_decision[0][0] > .5), parity_pass[0][0][0]))
            self.check_parity(byte_string, current_iter[0][0])

        time.sleep(10)
        self.current_results = None
        self.current_string = None
        return

def cropped_padder(cropped_image, width=300, height=300):
    padded_img = np.zeros((height, width)).astype(np.int16)

    pad_width = width - cropped_image.shape[1]
    pad_height = height - cropped_image.shape[0]

    lower_width = int(np.floor(pad_width/2))
    lower_height = int(np.floor(pad_height/2))
    upper_width = lower_width + cropped_image.shape[1]
    upper_height = lower_height + cropped_image.shape[0]

    padded_img[lower_height:upper_height, lower_width:upper_width] = cropped_image.copy()

    return padded_img
