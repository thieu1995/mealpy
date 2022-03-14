# !/usr/bin/env python
# Created by "Thieu" at 18:49, 13/12/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

## Time-series Prediction problem using Metaheuristic Algorithm to train Neural Network (Replace the Gradient Descent Optimizer)

# 1. Fitness function
# 2. Lower bound and upper bound of variables
# 3. Number of dimension (number of variables)
# 4. min, max

## https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/


# univariate mlp example
from numpy import array, size, reshape
from keras.models import Sequential
from keras.layers import Dense
from mealpy.swarm_based import GWO
from mealpy.evolutionary_based import FPA
from permetrics.regression import Metrics


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
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


class HybridMlp:

    def __init__(self, dataset, n_hidden_nodes, epoch, pop_size):
        self.X_train, self.X_test, self.Y_train, self.Y_test = dataset[0], dataset[1], dataset[2], dataset[3]
        self.n_hidden_nodes = n_hidden_nodes
        self.epoch = epoch
        self.pop_size = pop_size
        self.model, self.problem, self.optimizer, self.solution, self.best_fit = None, None, None, None, None
        self.n_dims, self.n_inputs = None, None

    def create_network(self):
        # define model
        model = Sequential()
        model.add(Dense(self.n_hidden_nodes, activation='relu', input_dim=n_steps))
        model.add(Dense(1))
        # model.compile(optimizer='adam', loss='mse')
        self.model = model

    def create_problem(self):
        self.n_inputs = self.X_train.shape[1]
        self.n_dims = (self.n_inputs * self.n_hidden_nodes) + self.n_hidden_nodes + (self.n_hidden_nodes * 1) + 1
        self.problem = {
            "fit_func": self.fitness_function,
            "lb": [-1, ] * self.n_dims,
            "ub": [1, ] * self.n_dims,
            "minmax": "min",
            "obj_weights": [0.3, 0.2, 0.5]  # [mae, mse, rmse]
        }

    def prediction(self, solution, data):
        self.decode_solution(solution)
        return self.model.predict(data)

    def training(self):
        self.create_network()
        self.create_problem()
        # self.optimizer = GWO.BaseGWO(self.problem, self.epoch, self.pop_size)
        self.optimizer = FPA.BaseFPA(self.problem, self.epoch, self.pop_size)
        self.solution, self.best_fit = self.optimizer.solve("thread")

        # 3 input nodes, 5 hidden node (1 single hidden layer), 1 output node
        # solution = [w11, w21, w31, w12, w22, w32, ....,  w15, w25, w35, b1, b2, b3, b4, b5, wh11, wh21, wh31, wh41, wh51, bo]
        # number of weights = number of dimensions = 3 * 5 + 5 + 5 * 1 + 1 = 26

    def decode_solution(self, solution=None):
        ## solution: vector
        ### Transfer solution back into weights of neural network
        weight_sizes = [(w.shape, size(w)) for w in self.model.get_weights()]
        weights = []
        cut_point = 0
        for ws in weight_sizes:
            temp = reshape(solution[cut_point: cut_point + ws[1]], ws[0])
            weights.append(temp)
            cut_point += ws[1]
        self.model.set_weights(weights)

    def fitness_function(self, solution):
        ## Training score and Testing score for fitness function
        ## with the weight: [0.3, 0.7]
        self.decode_solution(solution)
        predictions = self.model.predict(self.X_train)
        obj_metric = Metrics(self.Y_train.flatten(), predictions.flatten())
        # mse = obj_metric.get_metric_by_name("MSE")
        # rmse = obj_metric.get_metric_by_name("RMSE")
        # mae = obj_metric.get_metric_by_name("MAE")
        results_dict = obj_metric.get_metrics_by_list_names(["RMSE", "MAE", "MSE"])
        mae, mse, rmse = results_dict["MAE"], results_dict["MSE"], results_dict["RMSE"]
        return [mae, mse, rmse]


# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
# choose a number of time steps
n_steps = 3
# split into samples
X_train, Y_train = split_sequence(raw_seq[0:12], n_steps)
X_test, Y_test = split_sequence(raw_seq[12:20], n_steps)

## Initialization parameters
dataset = [X_train, X_test, Y_train, Y_test]
n_hidden_nodes = 5
epoch = 100
pop_size = 50

## Create hybrid model
model = HybridMlp(dataset, n_hidden_nodes, epoch, pop_size)
model.training()

## Access to the best model
# model.solution

## Predict the up coming time-series points
x_input = array([210, 220, 230])
x_input = x_input.reshape((1, n_steps))
yhat = model.prediction(model.solution, x_input)
print(yhat)

