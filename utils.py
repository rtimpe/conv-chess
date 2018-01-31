import chess
import chess.pgn
import math
import pystockfish

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

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
                board_fen = board.board_fen()
                # since we're reading board positions from actual games, there
                # will be many duplicate positions.  ignore them to keep the
                # training data unbiased
                if board_fen not in seen_boards:
                    seen_boards[board_fen] = board.fen()
        # X = np.array(list(seen_boards.values()))
    return list(seen_boards.values())

def flip_boards(boards):
    # boards should be N x 8 x 8 x 6
    return -1 * np.flip(np.flip(boards, axis=1), axis=2)

def flip_board(board):
    return -1 * np.flip(np.flip(board, axis=0), axis=1)

def fens_to_arrs(fens):
    arrs = []
    for fen in fens:
        board = chess.Board(fen=fen)
        arrs.append(board_to_arr(board))

    return np.array(arrs)

def convert_and_normalize(fens_and_scores):
    boards = []
    scores = []
    for fen, score in fens_and_scores:
        board = chess.Board(fen=fen)
        if board.turn == chess.WHITE:
            boards.append(board_to_arr(board))
            scores.append(score)
        else:
            boards.append(flip_board(board_to_arr(board)))
            scores.append(-1 * score)

    return np.array(boards), np.array(scores)

def score_fens(fens):
    eng = pystockfish.Engine(depth=10)
    data = []
    for fen in fens:
        # pystockfish is a bit buggy.  for now, just ignore boards it can't parse
        # FIXME at some point
        try:
            eng.setfenposition(fen)
            result = eng.bestmove()
            score = result['info']['score']['value']
            data.append((fen, score))
        except:
            pass
    return data

def board_to_arr(board):
    color_to_num = {chess.WHITE: 1, chess.BLACK: -1}
    arr = np.zeros((8, 8, 6))

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
            b.remove_piece_at(chess.square(i, j))
            piece = np.argmax(np.abs(quantized[i, j, :]))
            if quantized[i, j, piece] == 1:
                b.set_piece_at(chess.square(i, j), chess.Piece(piece + 1, True))
            if quantized[i, j, piece] == -1:
                b.set_piece_at(chess.square(i, j), chess.Piece(piece + 1, False))

    return b

def conv_layer(i, in_size, out_size, input_layer):
    with tf.variable_scope('conv_layer_' + str(i)):
        # input_layer = tf.layers.batch_normalization(input_layer, training=True)

        initializer = tf.contrib.layers.xavier_initializer()
        filters = tf.Variable(initializer([3, 3, int(in_size), out_size]), name='filters', dtype=tf.float32)
        # filters = tf.get_variable('filters', shape=[3, 3, in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())#tf.Variable(tf.random_uniform([3, 3, int(prev_layer.get_shape()[3]), layer]))
        tf.summary.histogram('filters', filters)
        b = tf.Variable(tf.zeros([out_size]))
        tf.summary.histogram('biases_', b)

        conv_out = tf.nn.relu(tf.nn.conv2d(input_layer, filters, [1, 1, 1, 1], "SAME") + b)

        conv_out = tf.layers.batch_normalization(conv_out, training=True)

        tf.summary.histogram('activations', conv_out)

        # mean, var = tf.nn.moments(conv_out, [0])
        # tf.summary.scalar('activation_mean', mean)
        # tf.summary.scalar('activation_var', var)

    return [filters, b], conv_out


def make_resnet(x, y, sess):
    resnet_out, _ = nets.resnet_v2.resnet_v2_50(x)
    initializer = tf.contrib.layers.xavier_initializer()
    W = tf.Variable(initializer([2048, 1]), name='fc_weights1')
    b = tf.Variable(tf.zeros([1]))
    score = tf.matmul(tf.squeeze(resnet_out), W) + b

    tf.summary.histogram('score', score)

    sess.run(tf.global_variables_initializer())
    return {
        'score': score,
        'cost': tf.reduce_mean(tf.square(score - y))
    }

def create_cnn(x, y, architecture, fc_dim, sess):
    prev_layer = x
    params = []
    for (i, layer) in enumerate(architecture):
        # layer is int (filter size) or 'pool' (max pool layer)
        if layer == 'pool':
            pool_out = tf.nn.max_pool(prev_layer, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='layer_' + str(i))
            prev_layer = pool_out
        else:
            # filters = tf.Variable(tf.random_uniform([3, 3, int(prev_layer.get_shape()[3]), layer]))
            # tf.summary.histogram('filters_' + str(i), filters)
            # b = tf.Variable(tf.zeros([layer]))
            # tf.summary.histogram('biases_' + str(i), b)
            # params.append(filters)
            # params.append(b)

            # conv_out = tf.nn.relu(tf.nn.conv2d(prev_layer, filters, [1, 1, 1, 1], "SAME") + b, name='layer_' + str(i))

            p, conv_out = conv_layer(i, prev_layer.get_shape()[3], layer, prev_layer)
            # params.append(p)
            params += p

            prev_layer = conv_out


    # add a two connected layers
    shape = prev_layer.get_shape()
    fc_input_dim = int(shape[1] * shape[2] * shape[3])
    fc_input = tf.reshape(prev_layer, [-1, fc_input_dim])
    initializer = tf.contrib.layers.xavier_initializer()
    W = tf.Variable(initializer([fc_input_dim, fc_dim]), name='fc_weights1')
    b = tf.Variable(tf.zeros([fc_dim]))
    params.append(W)
    params.append(b)

    fc_output = tf.nn.relu(tf.matmul(fc_input, W) + b)
    tf.summary.histogram('fc_activations', fc_output)

    W = tf.Variable(initializer([fc_dim, 1]), name='fc_weights2')
    b = tf.Variable(tf.zeros([1]))
    params.append(W)
    params.append(b)

    tf.summary.histogram('fc_weights2', W)

    score = tf.matmul(fc_output, W) + b
    tf.summary.histogram('scores', score)

    init = tf.variables_initializer(params)
    sess.run(init)

    cost = tf.reduce_mean(tf.square(score - y))
    tf.summary.scalar('cost', cost)

    return {
        'score': score,
        'cost': cost
    }

# http://people.idsia.ch/~ciresan/data/icann2011.pdf
def create_cae(x, architecture, fc_dim):
    num_filters = 20

    prev_layer = x
    all_filters = []
    all_biases = []
    for layer in architecture:
        # layer is int (filter size) or 'pool' (max pool layer)
        if layer == 'pool':
            pool_out = tf.nn.max_pool(prev_layer, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            prev_layer = pool_out
        else:
            filters = tf.Variable(tf.random_uniform([3, 3, int(prev_layer.get_shape()[3]), layer]))
            b = tf.Variable(tf.zeros([layer]))
            all_filters.append(filters)
            all_biases.append(b)

            conv_out = tf.nn.tanh(tf.nn.conv2d(prev_layer, filters, [1, 1, 1, 1], "SAME") + b)

            prev_layer = conv_out


    # add a single fully connected layer
    shape = prev_layer.get_shape()
    fc_input_dim = int(shape[1] * shape[2] * shape[3])
    fc_input = tf.reshape(prev_layer, [-1, fc_input_dim])
    W = tf.Variable(tf.random_uniform([fc_input_dim, fc_dim], -1.0 / math.sqrt(fc_input_dim), 1.0 / math.sqrt(fc_input_dim)))
    b1 = tf.Variable(tf.zeros([fc_dim]))

    fc_output = tf.nn.tanh(tf.matmul(fc_input, W) + b1)

    encoded = fc_output

    # now transpose weights of fc layer to start decoding
    b2 = tf.Variable(tf.zeros([fc_input_dim]))
    fc_t_output = tf.nn.tanh(tf.matmul(encoded, tf.transpose(W)) + b2)

    prev_layer = tf.reshape(fc_t_output, [-1, int(shape[1]), int(shape[2]), int(shape[3])])

    i = len(all_filters) - 1
    for layer in reversed(architecture):
        # layer is int (filter size) or 'pool' (max pool layer)
        if layer == 'pool':
            pool_out = UnPooling2x2ZeroFilled(prev_layer)
            prev_layer = pool_out
        else:
            filters = tf.transpose(all_filters[i], perm=[1, 0, 3, 2])
            b = tf.Variable(tf.zeros([filters.get_shape()[3]]))
            all_biases.append(b)

            conv_out = tf.nn.tanh(tf.nn.conv2d(prev_layer, filters, [1, 1, 1, 1], "SAME") + b)

            prev_layer = conv_out
            i -= 1

    # add a special conv layer to get output to 8 x 8 x 6
    filters = tf.Variable(tf.random_uniform([3, 3, int(prev_layer.get_shape()[3]), 6]))
    b = tf.Variable(tf.zeros([6]))
    all_filters.append(filters)
    all_biases.append(b)

    conv_out = tf.nn.tanh(tf.nn.conv2d(prev_layer, filters, [1, 1, 1, 1], "SAME") + b)

    decoded = conv_out

    # initialize variables here.  not sure if this is the best place to do it?
    # but good enough for now
    init = tf.variables_initializer(all_filters + all_biases + [W] + [b1] + [b2])

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

        output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)

        # the input into the next layer is the output of this layer
        next_layer_input = output

    # The fully encoded x value is now stored in the next_layer_input
    encoded_x = next_layer_input

    # build the reconstruction layers by reversing the reductions
    layer_sizes.reverse()
    encoding_matrices.reverse()


    for i, dim in enumerate(layer_sizes[1:] + [int(x.get_shape()[1])]):
        # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
        W = tf.transpose(encoding_matrices[i])
        b = tf.Variable(tf.zeros([dim]))
        output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)
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
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def train(autoencoder, X, flatten, input_ph, sess, num_iters=20000, batch_size=100, lr=.001):
    train_step = tf.train.AdamOptimizer(lr).minimize(autoencoder['cost'])

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

def train_labeled(cnn, X, y, input_ph, label_ph, sess, X_eval=None, y_eval=None, num_iters=20000, batch_size=100, lr=.0001):
    train_step = tf.train.AdamOptimizer(lr).minimize(cnn['cost'])

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('summaries', sess.graph)

    # initialize remaining variables (should just be variables from the optimizer)
    print(sess.run(tf.report_uninitialized_variables()))
    initialize_uninitialized(sess)

    for i in range(num_iters):
        inds = np.random.choice(X.shape[0], batch_size)
        X_batch = X[inds]
        y_batch = y[inds, np.newaxis]

        sess.run(train_step, feed_dict={input_ph: X_batch, label_ph: y_batch})

        # if i % 500 == 0:
        #     print(sorted(inds))

        if i % 1000 == 0:
            summary, cost = sess.run([merged, cnn['cost']], feed_dict=
                                     {input_ph: X_batch, label_ph: y_batch})
            print(i, " cost", cost)
            writer.add_summary(summary, i)
            if X_eval is not None:
                cost = sess.run(cnn['cost'], feed_dict={input_ph: X_eval, label_ph: y_eval[:, np.newaxis]})
                print("eval cost", cost)
