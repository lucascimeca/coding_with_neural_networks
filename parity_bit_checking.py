# run Game of Life live, with learned forward and backward functions
from itertools import count
import numpy as np
import time
import random
from itertools import product
from model_definitions import ParityBitChecker

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

_bytes_string = np.array(list(product([0., 1.], repeat=8)))
bytes_string_subset = _bytes_string[random.sample(list(range(_bytes_string.shape[0])), 15), :]


def run_parity_bit_checking(bytes_string=None):
    bit_checker = ParityBitChecker()
    bit_checker.check_parity(bytes_string, bytes_number=bytes_string.shape[0])


if __name__ == "__main__":
    run_parity_bit_checking(bytes_string=bytes_string_subset)
