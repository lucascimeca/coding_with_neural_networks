# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of data points.
"""

import os
import numpy as np
from mlp import DEFAULT_SEED
from itertools import product
import copy

class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch

    # Python 3.x compatibility
    def __next__(self):
        return self.next()


class GameOfLifeDataProvider(DataProvider):
    """Data provider for CIFAR-10 object images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, input_num=None, convolution=False, function='forward'):
        """Create a new GOL-10 data provider object.

        Args:
            which_set: One of 'train' or 'valid'. Determines which
                portion of the CIFAR-10 data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid'], (
            'Expected which_set to be either train or valid. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 2**9
        self.function = function
        # load data from compressed numpy file
        print("prev dataset")
        inputs = np.array(list(product([0.,1.], repeat=9)))


        if function is 'identity':
            targets = copy.deepcopy(inputs)
        elif function is 'forward':
            targets = self.step_forward(inputs)
        else:
            # currently implemented in the wrong way!!
            # todo
            print("generating dataset!")
            targets = np.array(list(product([0., 1.], repeat=25)))
            inputs = self.step_forward(inputs)

            tmp = copy.deepcopy(inputs)
            # targets = tmp
            # prune in-out maps to achieve pseudo inverse
            uniqueInputs, indicesList = np.unique(copy.deepcopy(targets), return_index=True, axis=0)
            targets = np.zeros(inputs.shape)
            for i in range(inputs.shape[0]):
                targets[i, :] = tmp[indicesList[i]]

        # label map gives strings corresponding to integer label targets
        self.label_map = None
        if input_num:
            inputs = inputs[:input_num]
            targets = targets[:input_num]
        inputs = inputs if not convolution else inputs.reshape(inputs.shape[0], 32, 32, 3)
        # pass the loaded data to the parent class __init__
        if batch_size == 0:
            batch_size = inputs.shape[0]
        super(GameOfLifeDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def gof_next_fun(self, input):
        output = copy.deepcopy(input)
        determinant_vas = input[:4] + input[5:]
        if sum(determinant_vas) < 2:
            output[4] = 0.
        elif sum(determinant_vas) == 3:
            output[4] = 1.
        elif sum(determinant_vas) > 3:
            output[4] = 0.
        else:
            output[4] = output[4]
        return output

    def gof_prev_fun(self, input):
        determinant_vas = input[:4] + input[5:]
        if input[4] == 1.:
            if sum(determinant_vas) < 2:
                pass
            elif sum(determinant_vas) == 3:
                pass
            else:
                pass
        else:
            if sum(determinant_vas) < 2:
                pass
            elif sum(determinant_vas) == 3:
                pass
            else:
                pass

    def step_forward(self, inputs):
        if inputs.shape[1] == 9:
            outputs = np.zeros(inputs.shape)
            for i in range(inputs.shape[0]):
                outputs[i, :] = self.gof_next_fun(inputs[i, :])
        else:
            side_pad = 1
            outputs = copy.deepcopy(inputs)
            dim = np.sqrt(inputs.shape[1]).astype(np.int32)
            for elem_idx in inputs.shape[0]:
                board = inputs[elem_idx, :].reshape(dim, -1)
                next_board = copy.deepcopy(board)
                for i in range(dim-2):
                    for j in range(dim-2):
                        board_patch = board[i+side_pad-1:i+side_pad+2,
                                            j+side_pad-1:j+side_pad+2]
                        predictions = self.gof_next_fun(board_patch.reshape(1, -1))
                        next_board[i+side_pad, j+side_pad] = predictions[0, 4]
                outputs[elem_idx, :] = next_board.reshape(1, -1)
        return outputs


class ParityCheckDataProvider(DataProvider):
    """Data provider for CIFAR-10 object images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, input_num=None, convolution=False, protocol='even'):
        """Create a new GOL-10 data provider object.

        Args:
            which_set: One of 'train' or 'valid'. Determines which
                portion of the CIFAR-10 data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid'], (
            'Expected which_set to be either train or valid. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 2**9
        # load data from compressed numpy file
        self.protocol = protocol
        inputs = np.array(list(product([0., 1.], repeat=8)))
        targets = np.zeros((inputs.shape[0], 1))
        if self.protocol == 'even':
            for i in range(inputs.shape[0]):
                targets[i, :] = int(np.sum(inputs[i, :]) % 2 == 0)
        else:
            for i in range(inputs.shape[0]):
                targets[i, :] = int(np.sum(inputs[i, :]) % 2 != 0)


        # label map gives strings corresponding to integer label targets
        self.label_map = None
        if input_num:
            inputs = inputs[:input_num]
            targets = targets[:input_num]
        inputs = inputs if not convolution else inputs.reshape(inputs.shape[0], 32, 32, 3)
        # pass the loaded data to the parent class __init__
        if batch_size == 0:
            batch_size = inputs.shape[0]
        super(ParityCheckDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)


class CounterDataProvider(DataProvider):
    """Data provider for CIFAR-10 object images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, input_num=None, convolution=False, function='forward'):
        """Create a new GOL-10 data provider object.

        Args:
            which_set: One of 'train' or 'valid'. Determines which
                portion of the CIFAR-10 data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid'], (
            'Expected which_set to be either train or valid. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 2**9
        self.function = function
        # load data from compressed numpy file
        print("prev dataset")
        inputs = np.array(list(range(-100000, 1000000))).reshape(-1, 1)
        targets = np.zeros((inputs.shape[0], 2))
        targets[:, 0] = (inputs-1).reshape(-1)
        targets[:, 1] = (inputs > 0.).astype(np.int64).reshape(-1)

        # label map gives strings corresponding to integer label targets
        self.label_map = None
        if input_num:
            inputs = inputs[:input_num]
            targets = targets[:input_num]
        inputs = inputs if not convolution else inputs.reshape(inputs.shape[0], 32, 32, 3)
        # pass the loaded data to the parent class __init__
        if batch_size == 0:
            batch_size = inputs.shape[0]
        super(CounterDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

