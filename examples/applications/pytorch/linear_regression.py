#!/usr/bin/env python
# Created by "Thieu" at 13:47, 23/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/
import numpy as np
import torch
from torch.autograd import Variable
from mealpy import StringVar, FloatVar, IntegerVar, BBO, Problem


# Define LR model
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        return self.linear(x)


class MyProblem(Problem):
    def __init__(self, bounds, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x: np.ndarray):
        # Decode solution from real value to real-world solution
        x = self.decode_solution(x)
        opt, learning_rate, epoch = x["optimizer"], x["learning-rate"], x["epoch"]

        # Create model object
        ml_model = LinearRegressionModel()
        criterion = torch.nn.MSELoss(reduction="sum")
        optimizer = getattr(torch.optim, opt)(ml_model.parameters(), lr=learning_rate)

        ## Training
        for iter in range(epoch):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = ml_model(self.data["X_train"])
            # Compute and print loss
            loss = criterion(y_pred, self.data["y_train"])
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {}, loss {}'.format(epoch, loss.item()))
        train_loss = loss.item()

        ## Testing
        y_pred_test = ml_model(self.data["X_test"])
        test_loss = criterion(y_pred_test, self.data["y_test"]).item()

        ## Return metrics as fitness values
        # We will use weighting method to get the best model based on both train and test
        return [train_loss, test_loss]


MAX_GEN = 10
POP_SIZE = 10

dataset = {
    "X_train": Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])),
    "y_train": Variable(torch.Tensor([[2.1], [4.2], [6.3], [8.4], [10.5], [12.6]])),
    "X_test": Variable(torch.Tensor([[8.0], [9.0], [10.0]])),
    "y_test": Variable(torch.Tensor([[16.8], [18.9], [21.0]]))
}

bounds = [
    StringVar(valid_sets=["SGD", "Adam", "RMSprop", "Rprop", "Adamax", "Adagrad"], name="optimizer"),
    FloatVar(lb=0., ub=1.0, name="learning-rate"),
    IntegerVar(lb=10, ub=40, name="epoch")
]

## training weight 0.3 and testing weight 0.7
problem = MyProblem(bounds=bounds, minmax="min", data=dataset, name="ml-problem", obj_weights=(0.5, 0.5))
model = BBO.OriginalBBO(epoch=MAX_GEN, pop_size=POP_SIZE, pr=0.03)
best_agent = model.solve(problem)

print(f"Best agent: {best_agent}")
print(f"Best solution: {best_agent.solution}")
print(f"Best accuracy: {best_agent.target.fitness}")
print(f"Best parameters: {model.problem.decode_solution(best_agent.solution)}")
