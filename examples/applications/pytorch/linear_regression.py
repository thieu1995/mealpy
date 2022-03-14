# !/usr/bin/env python
# Created by "Thieu" at 13:47, 23/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/

import torch
from torch.autograd import Variable
from sklearn import preprocessing
from mealpy.bio_based import SMA

X_TRAIN = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]))
Y_TRAIN = Variable(torch.Tensor([[2.1], [4.2], [6.3], [8.4], [10.5], [12.6]]))

X_TEST = Variable(torch.Tensor([[8.0], [9.0], [10.0]]))
Y_TEST = Variable(torch.Tensor([[16.8], [18.9], [21.0]]))

# Handle categorical variable first
OPT_ENCODER = preprocessing.LabelEncoder()
OPT_ENCODER.fit(["SGD", "Adam", "RMSprop", "Rprop", "Adamax", "Adagrad"])

# Define LR model
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        return self.linear(x)


def fitness_function(solution):
    def training(epoch):
        for iter in range(epoch):
            # Forward pass: Compute predicted y by passing x to the model
            pred_y = our_model(X_TRAIN)
            # Compute and print loss
            loss = criterion(pred_y, Y_TRAIN)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {}, loss {}'.format(epoch, loss.item()))
            return loss.item()
    def testing():
        pred_y = our_model(X_TEST)
        return criterion(pred_y, Y_TEST).item()

    ## De-coded solution into real parameters
    ## 1st dimension is: optimizer
    ## 2nd dimension is: learning-rate
    ## 3rt dimension is: epoch
    opt = OPT_ENCODER.inverse_transform([ int(solution[0]) ])[0]
    lr = solution[1]
    epoch = int(solution[2]) * 50

    # Create model object
    our_model = LinearRegressionModel()
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = getattr(torch.optim, opt)(our_model.parameters(), lr=lr)
    train_loss = training(epoch)
    test_loss = testing()
    return [train_loss, test_loss]      # We will use weighting method to get the best model based on both train and test


# So now we need to define lower/upper bound and metaheursitic algorithm
LB = [1, 0, 10]  # [lowerbound for optimizer, lowerbound for learning-rate, lowerbound for epoch]
UB = [6.99, 1, 40]  # [upperbound for optimizer, upperbound for learning-rate, upperbound for poch]

MAX_GEN = 50
POP_SIZE = 20

problem = {
    "fit_func": fitness_function,
    "lb": LB,
    "ub": UB,
    "minmax": "min",
    "obj_weights": [0.3, 0.7]        # training weight 0.3 and testing weight 0.7
}
model = SMA.BaseSMA(problem, epoch=MAX_GEN, pop_size=POP_SIZE, pr=0.03)
model.solve()
print(f"Best fitness: {model.solution[1]}")

# This will print out the best value (optimized value) of opt, learning-rate and epoch. Just need a decode function
opt_optimized = OPT_ENCODER.inverse_transform([ int(model.solution[0][0]) ])[0]
lr_optimized = model.solution[0][1]
epoch_optimized = int(model.solution[0][2]) * 50
print(f"Best optimizer = {opt_optimized}, lr = {lr_optimized}, epoch = {epoch_optimized}")



