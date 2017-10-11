import chess
import chess.pgn
import math

import numpy as np
import tensorflow as tf

def get_boards_from_game(game):
    boards = []
    node = game
    while not node.is_end():
        next_node = node.variations[0]
        node = next_node
        boards.append(node.board())
    return boards

def board_to_arr(board):
    color_to_num = {chess.WHITE: 1, chess.BLACK: -1}
    arr = np.zeros((8,8,6))

    for color in [True, False]:
        for piece in range(6):
            sqs = board.pieces(piece + 1, color)
            for sq in sqs:
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                arr[file, rank, piece] = color_to_num[color]
    return arr

def quantize_arr_vec(arr):
    out = np.zeros_like(arr)
    # arr is N x 8 x 8 x 6
    arr = np.copy(arr)
    arr_reshaped = np.reshape(arr, (-1, arr.shape[-1])) # now D x 6

    abs_maxes = np.argmax(np.abs(arr_reshaped), axis=1)

    abs_mask = np.zeros_like(arr_reshaped)
    abs_mask[np.arange(abs_mask.shape[0]), abs_maxes] = 1

    # first do maxes
    max_mask = np.logical_and(abs_mask, arr_reshaped > .5)
    max_mask = np.reshape(max_mask, arr.shape)
    out[max_mask] = 1

    # now do the mins
    min_mask = np.logical_and(abs_mask, arr_reshaped < -.5)
    min_mask = np.reshape(min_mask, arr.shape)
    out[min_mask] = -1

    return out

def arr_to_board(arr):
    b = chess.Board()

    quantized = quantize_arr_vec(arr)

    for i in range(8):
        for j in range(8):
            b.remove_piece_at(chess.square(i,j))
            piece = np.argmax(np.abs(quantized[i,j,:]))
            if quantized[i,j,piece] == 1:
                b.set_piece_at(chess.square(i,j), chess.Piece(piece + 1, True))
            if quantized[i,j,piece] == -1:
                b.set_piece_at(chess.square(i,j), chess.Piece(piece + 1, False))

    return b

def create_cae(x, filter_sizes):
    num_filters = 20

    prev_layer = x
    for f in filter_sizes:
        filters = tf.Variable(tf.random_uniform([3, 3, int(prev_layer.get_shape()[3]), f]))
        b = tf.Variable(tf.zeros([f]))
        conv_out = tf.nn.tanh(tf.nn.conv2d(prev_layer, filters, [1,1,1,1], "SAME") + b)

        pool_out = tf.nn.max_pool(conv_out, [1,2,2,1], [1,2,2,1], 'SAME')
        prev_layer = pool_out

    encoded = pool_out

    for f in reversed([6] + filter_sizes[1:]):
        filters = tf.Variable(tf.random_uniform([3, 3, int(prev_layer.get_shape()[3]), f]))
        b = tf.Variable(tf.zeros([f]))
        conv_out = tf.nn.tanh(tf.nn.conv2d(prev_layer, filters, [1,1,1,1], "SAME") + b)

        pool_out = UnPooling2x2ZeroFilled(conv_out)
        prev_layer = pool_out

    decoded = pool_out

    return {
        'encoded': encoded,
        'decoded': decoded,
        'cost' : tf.sqrt(tf.reduce_mean(tf.square(x - decoded)))
    }

# https://github.com/ppwwyyxx/tensorpack/blob/5c9174db6c0710e04665eb8918a2bf3ffa0d043b/tensorpack/models/pool.py#L77
def UnPooling2x2ZeroFilled(x):
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
    return ret

def create(x, layer_sizes):
    # Build the encoding layers
    next_layer_input = x

    encoding_matrices = []
    for dim in layer_sizes:
        input_dim = int(next_layer_input.get_shape()[1])

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))

        # Initialize b to zero
        b = tf.Variable(tf.zeros([dim]))

        # We are going to use tied-weights so store the W matrix for later reference.
        encoding_matrices.append(W)

        output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)

        # the input into the next layer is the output of this layer
        next_layer_input = output

    # The fully encoded x value is now stored in the next_layer_input
    encoded_x = next_layer_input

    # build the reconstruction layers by reversing the reductions
    layer_sizes.reverse()
    encoding_matrices.reverse()


    for i, dim in enumerate(layer_sizes[1:] + [ int(x.get_shape()[1])]) :
        # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
        W = tf.transpose(encoding_matrices[i])
        b = tf.Variable(tf.zeros([dim]))
        output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)
        next_layer_input = output

    # the fully encoded and reconstructed value of x is here:
    reconstructed_x = next_layer_input

    return {
    'encoded': encoded_x,
    'decoded': reconstructed_x,
    'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
    }
