# !/usr/bin/env python
# Created by "Thieu" at 23:59, 14/12/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

# https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/

# 1. Fitness function
# 2. Lower bound and upper bound of variables
# 3. Min or max
# 4. Number of dimensions (number of variables)


# Assumption that we are trying to optimize the multi-layer perceptron with 3 layer 1 input, 1 hidden, 1 output.
# 1. Batch-size training
# 2. Epoch training
# 3. Optimizer
# 4. Learning rate
# 5. network weight initialization
# 6. activation functions
# 7. number of hidden units

# Rules:
# 1. Batch-size: [ 2, 4, 8 ]
# 2. Epoch : [700, 800, .... 2000]
# 3. Optimizer: ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# 4. Learning rate: [0.01 -> 0.5]       real number
# 5. network weight initialization: ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# 6. activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# 7. hidden units: [5, 100] --> integer number


# solution = [ x1, x2, x3, x4, x5, x6, x7, x8,  ]

# 1st solution: hidden layer = 2  ==> [ x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 ]
# x9: the number of hidden units of layer 1
# x10: the number of hidden units of layer 2
# 2nd solution: hidden layer = 4 ==> [ x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 ]
# But in Metaheuristic Process --> You can not expand the solution.
# The number of dimensions is fixed before and after the MH process.


# 1 way: solution = [ x1, x2, x3, x4, x5, x6, x7, x8,  ]
# x8: should be the number of hidden layer
# x7: should be the number of hidden node in each layer --> all hidden layer has the same number of hidden node.


# 2 way: I limit the number of hidden layers to 5. Number of hidden layers belongs [1, 5]

# solution = [ x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13 ]
# x8: number of hidden layers
# x9: number of hidden units in 1st the hidden layer
# x10: ..... 2nd hidden layer
# x11: .... 3rd hidden layer
# x12: .... 4th hidden layer
# x13: .... 5th hidden layer

# if a solution with x8 = 2 hidden layers ==>
# x9, x10 --> then ignore other values: x11, x12, x13

# solution 1 = [x1, x2, x3, x4, x5, x6, x7, x8, 100, 50, 10, 10, 10]
# solution 2 = [x1, x2, x3, x4, x5, x6, x7, x8, 100, 50, 10, 10, 20]
# solution 3 = [x1, x2, x3, x4, x5, x6, x7, x8, 100, 50, 10, 10, 30]

# 8. Number of hidden layers with number of hidden nodes in each layers.


# univariate mlp example
from numpy import array, reshape
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from mealpy.swarm_based import GWO
from keras import optimizers


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    ## https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# LABEL ENCODER
OPT_ENCODER = LabelEncoder()
OPT_ENCODER.fit(['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])  # domain range ==> 7 values
# SGD <= 0
# RMSprop <= 1
print(OPT_ENCODER.inverse_transform([1]))

WOI_ENCODER = LabelEncoder()
WOI_ENCODER.fit(['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])

ACT_ENCODER = LabelEncoder()
ACT_ENCODER.fit(['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])


def decode_solution(solution):
    batch_size = 2 ** int(solution[0])
    epoch = 100 * int(solution[1])
    opt_integer = int(solution[2])
    opt = OPT_ENCODER.inverse_transform([opt_integer])[0]
    learning_rate = solution[3]
    network_weight_initial_integer = int(solution[4])
    network_weight_initial = WOI_ENCODER.inverse_transform([network_weight_initial_integer])[0]
    act_integer = int(solution[5])
    activation = ACT_ENCODER.inverse_transform([act_integer])[0]
    n_hidden_units = int(solution[6])
    return [batch_size, epoch, opt, learning_rate, network_weight_initial, activation, n_hidden_units]


def fitness_function(solution):
    # batch_size = 2**int(solution[0])
    # # 1 -> 1.99 ==> 1
    # # 2 -> 2.99 ==> 2
    # # 3 -> 3.99 ==> 3
    #
    # epoch = 100 * int(solution[1])
    # # 100 * 7 = 700
    # # 100 * 20 = 2000
    #
    # opt_integer = int(solution[2])
    # opt = OPT_ENCODER.inverse_transform([opt_integer])[0]
    # # 0 - 0.99 ==> 0 index ==> should be SGD (for example)
    # # 1 - 1.99 ==> 1 index ==> should be RMSProp
    #
    # learning_rate = solution[3]
    #
    # network_weight_initial_integer = int(solution[4])
    # network_weight_initial = WOI_ENCODER.inverse_transform([network_weight_initial_integer])[0]
    #
    # act_integer = int(solution[5])
    # activation = ACT_ENCODER.inverse_transform([act_integer])[0]
    #
    # n_hidden_units = int(solution[6])

    batch_size, epoch, opt, learning_rate, network_weight_initial, activation, n_hidden_units = decode_solution(solution)

    # define model
    model = Sequential()
    model.add(Dense(n_hidden_units, activation=activation, input_dim=n_steps, kernel_initializer=network_weight_initial))
    model.add(Dense(1))

    # Compile model
    optimizer = getattr(optimizers, opt)(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # fit model
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=0)

    # We take the loss value of validation set as a fitness value for selecting the best model
    # demonstrate prediction

    yhat = model(X_test)
    fitness = mean_squared_error(y_test, yhat)
    return fitness


LB = [1, 7, 0, 0.01, 0, 0, 5]
UB = [3.99, 20.99, 6.99, 0.5, 7.99, 7.99, 50]

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
scaler = MinMaxScaler()
scaled_seq = scaler.fit_transform(reshape(raw_seq, (-1, 1))).flatten()

# choose a number of time steps
n_steps = 3
# split into samples            60% - training
X_train, y_train = split_sequence(scaled_seq[0:12], n_steps)
X_test, y_test = split_sequence(scaled_seq[12:20], n_steps)

problem = {
    "fit_func": fitness_function,
    "lb": LB,
    "ub": UB,
    "minmax": "min",
    "verbose": True,
}

from mealpy.evolutionary_based import FPA

model = FPA.BaseFPA(problem, epoch=5, pop_size=20)

# model = GWO.BaseGWO(problem, epoch=5, pop_size=20)
model.solve()

print(f"Best solution: {model.solution[0]}")
batch_size, epoch, opt, learning_rate, network_weight_initial, activation, n_hidden_units = decode_solution(model.solution[0])

print(f"Batch-size: {batch_size}, Epoch: {epoch}, Opt: {opt}, Learning-rate: {learning_rate}")
print(f"NWI: {network_weight_initial}, Activation: {activation}, n-hidden: {n_hidden_units}")




