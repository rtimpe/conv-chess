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

def load_boards_from_pgnf(fname, num_games=100):
    with open(fname) as pgnf:
        seen_boards = {}
        for i in range(num_games):
            game = chess.pgn.read_game(pgnf)
            boards = get_boards_from_game(game)
            for board in boards:
                arr = board_to_arr(board)
                board_fen = board.board_fen()
                # since we're reading board positions from actual games, there
                # will be many duplicate positions.  ignore them to keep the
                # training data unbiased
                if board_fen not in seen_boards:
                    seen_boards[board_fen] = arr
        X = np.array(list(seen_boards.values()))
    return X

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
    all_variables = []
    for f in filter_sizes:
        filters = tf.Variable(tf.random_uniform([3, 3, int(prev_layer.get_shape()[3]), f]))
        b = tf.Variable(tf.zeros([f]))
        all_variables.append(filters)
        all_variables.append(b)

        conv_out = tf.nn.tanh(tf.nn.conv2d(prev_layer, filters, [1,1,1,1], "SAME") + b)

        pool_out = tf.nn.max_pool(conv_out, [1,2,2,1], [1,2,2,1], 'SAME')
        prev_layer = pool_out

    encoded = pool_out

    for f in reversed([6] + filter_sizes[1:]):
        filters = tf.Variable(tf.random_uniform([3, 3, int(prev_layer.get_shape()[3]), f]))
        b = tf.Variable(tf.zeros([f]))
        all_variables.append(filters)
        all_variables.append(b)

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

def create_fully_connected(x, layer_sizes, sess):
    # Build the encoding layers
    next_layer_input = x

    all_variables = []
    encoding_matrices = []
    for dim in layer_sizes:
        input_dim = int(next_layer_input.get_shape()[1])

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))

        # Initialize b to zero
        b = tf.Variable(tf.zeros([dim]))

        # We are going to use tied-weights so store the W matrix for later reference.
        encoding_matrices.append(W)

        # also store variables for initialization
        all_variables.append(W)
        all_variables.append(b)

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

        all_variables.append(b)

    # the fully encoded and reconstructed value of x is here:
    reconstructed_x = next_layer_input

    # initialize variables here.  not sure if this is the best place to do it?
    # but good enough for now
    init = tf.variables_initializer(all_variables)
    sess.run(init)

    return {
        'encoded': encoded_x,
        'decoded': reconstructed_x,
        'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
    }


# https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables#35618160
def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def train(autoencoder, X, flatten, input_ph, sess, num_iters=20000, batch_size=100):
    train_step = tf.train.AdamOptimizer(0.001).minimize(autoencoder['cost'])

    # initialize remaining variables (should just be variables from the optimizer)
    initialize_uninitialized(sess)
    # print(sess.run(tf.report_uninitialized_variables()))
    # sess.run(tf.initialize_variables([tf.get_variable(name) for name in sess.run(tf.report_uninitialized_variables())]))

    for i in range(num_iters):
        inds = np.random.choice(X.shape[0], batch_size)
        X_batch = X[inds]
        if flatten:
           X_batch = np.reshape(X_batch, (batch_size, -1))
        sess.run(train_step, feed_dict={input_ph: X_batch})
        if i % 1000 == 0:
            print(i, " cost", sess.run(autoencoder['cost'], feed_dict={input_ph: X_batch}))
