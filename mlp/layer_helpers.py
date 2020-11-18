import tensorflow as tf
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM


# ---------------- define helper functions -------------------------------------------------------------------
def fully_connected_layer(l_inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim) ** 0.5),
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(l_inputs, weights) + biases)
    return outputs


def conv_layer(l_inputs, input_channel_dim, output_channel_dim, kernel_size=5, bias_init=0.0, name=""):
    kernel = tf.Variable(
        tf.truncated_normal(
            shape=[kernel_size,
                   kernel_size,
                   input_channel_dim,
                   output_channel_dim],
            stddev=1e-3,
            name='weights'))

    conv = tf.nn.conv2d(
        l_inputs,
        kernel,
        [1, 1, 1, 1],
        padding='SAME'
    )
    biases = tf.Variable(tf.zeros([output_channel_dim]), 'biases')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name="conv" + name)
    return conv1


def pool_layer(l_inputs, kernel_size=[1, 3, 3, 1], strides_dim=[1, 2, 2, 1], name=""):
    return tf.nn.max_pool(l_inputs, ksize=kernel_size,
                          strides=strides_dim,
                          padding='SAME', name='pool' + name)


def norm_layer(l_inputs, depth_radius=4, bias=1, alpha=0.001 / 9.0, beta=0.75, name=""):
    return tf.nn.lrn(l_inputs, depth_radius=depth_radius,
                     bias=bias, alpha=alpha, beta=beta,
                     name='norm' + name)


def reshape_layer(l_inputs, output_dim, batch_size, bias_init=0.1, name=""):
    reshape = tf.reshape(l_inputs, tf.stack([batch_size, -1]))
    dim = l_inputs.get_shape()[1].value*l_inputs.get_shape()[2].value*l_inputs.get_shape()[3].value
    weights = tf.Variable(tf.truncated_normal(
                                [dim, output_dim],
                                stddev=1e-3),
                          name='weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    return tf.nn.relu(tf.matmul(reshape, weights) + biases, name="reshape"+name)


def inception2d(l_inputs, input_channel_dim, output_channel_dim, kernel_sizes=[5, 10, 20, 25], name=""):

    # 5x5
    one_filter = tf.Variable(tf.truncated_normal([kernel_sizes[0], kernel_sizes[0], input_channel_dim, output_channel_dim], stddev=5e-2, name='weights'))
    one_by_one = tf.nn.conv2d(l_inputs, one_filter, strides=[1, 1, 1, 1], padding='SAME')

    # 10x10
    two_filter = tf.Variable(tf.truncated_normal([kernel_sizes[1], kernel_sizes[1], input_channel_dim, output_channel_dim], stddev=1e-2, name='weights'))
    two_by_three = tf.nn.conv2d(l_inputs, two_filter, strides=[1, 1, 1, 1], padding='SAME')

    # 25x25
    three_filter = tf.Variable(tf.truncated_normal([kernel_sizes[2], kernel_sizes[2], input_channel_dim, output_channel_dim], stddev=2e-3, name='weights'))
    three_by_five = tf.nn.conv2d(l_inputs, three_filter, strides=[1, 1, 1, 1], padding='SAME')

    # 25x25
    four_filter = tf.Variable(tf.truncated_normal([kernel_sizes[3], kernel_sizes[3], input_channel_dim, output_channel_dim], stddev=2e-3, name='weights'))
    four_by_five = tf.nn.conv2d(l_inputs, four_filter, strides=[1, 1, 1, 1], padding='SAME')

    # bias dimension = 3*filter_count and then the extra in_channels for the avg pooling
    # biases = tf.Variable(tf.truncated_normal([3*output_channel_dim], 0, 1e-3)),

    biases = tf.Variable(tf.zeros([4*output_channel_dim]), 'biases')
    incept_l = tf.concat([one_by_one, two_by_three, three_by_five, four_by_five], axis=3)  # Concat in the 4th dim to stack
    pre_activation = tf.nn.bias_add(incept_l, biases)
    return tf.nn.relu(pre_activation, name="inception_" + name)
