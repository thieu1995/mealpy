#!/usr/bin/env python
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
from sklearn.preprocessing import LabelEncoder
from examples.applications.keras.timeseries_util import generate_data, decode_solution, generate_loss_value
from mealpy.evolutionary_based import FPA
from mealpy.swarm_based import GWO


def fitness_function(solution, data):
    structure = decode_solution(solution, data)
    fitness = generate_loss_value(structure, data)
    return fitness


if __name__ == "__main__":
    # LABEL ENCODER
    OPT_ENCODER = LabelEncoder()
    OPT_ENCODER.fit(['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])  # domain range ==> 7 values

    WOI_ENCODER = LabelEncoder()
    WOI_ENCODER.fit(['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])

    ACT_ENCODER = LabelEncoder()
    ACT_ENCODER.fit(['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])

    DATA = generate_data()
    DATA["OPT_ENCODER"] = OPT_ENCODER
    DATA["WOI_ENCODER"] = WOI_ENCODER
    DATA["ACT_ENCODER"] = ACT_ENCODER

    LB = [1, 5, 0, 0.01, 0, 0, 5]
    UB = [3.99, 20.99, 6.99, 0.5, 7.99, 7.99, 50]

    problem = {
        "fit_func": fitness_function,
        "lb": LB,
        "ub": UB,
        "minmax": "min",
        "log_to": None,
        "save_population": False,
        "data": DATA,
    }
    model = FPA.OriginalFPA(epoch=5, pop_size=20)
    # model = GWO.OriginalGWO(epoch=5, pop_size=20)
    model.solve(problem)

    print(f"Best solution: {model.solution[0]}")
    sol = decode_solution(model.solution[0], DATA)

    print(f"Batch-size: {sol['batch_size']}, Epoch: {sol['epoch']}, Opt: {sol['opt']}, "
          f"Learning-rate: {sol['learning_rate']}, NWI: {sol['network_weight_initial']}, "
          f"Activation: {sol['activation']}, n-hidden: {sol['n_hidden_units']}")
