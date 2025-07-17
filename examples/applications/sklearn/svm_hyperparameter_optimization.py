#!/usr/bin/env python
# Created by "Thieu" at 15:19, 17/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

"""
### Example: Optimize Machine Learning Model Hyperparameters

In this example, we'll use the SMA optimizer to find the optimal hyperparameters for an SVC (Support Vector Classifier) model.
This demonstrates how to integrate MEALPY with a machine learning workflow.
"""


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, metrics

from mealpy import FloatVar, StringVar, IntegerVar, BoolVar, CategoricalVar, SMA, Problem


# Load the data set; In this example, the breast cancer dataset is loaded.
X, y = datasets.load_breast_cancer(return_X_y=True)

# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

data = {
    "X_train": X_train_std,
    "X_test": X_test_std,
    "y_train": y_train,
    "y_test": y_test
}


class SvmOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        super().__init__(bounds, minmax, **kwargs)
        self.data = data

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        C_paras, kernel_paras = x_decoded["C_paras"], x_decoded["kernel_paras"]
        degree, gamma, probability = x_decoded["degree_paras"], x_decoded["gamma_paras"], x_decoded["probability_paras"]

        svc = SVC(C=C_paras, kernel=kernel_paras, degree=degree,
                  gamma=gamma, probability=probability, random_state=1)
        # Fit the model
        svc.fit(self.data["X_train"], self.data["y_train"])
        # Make the predictions
        y_predict = svc.predict(self.data["X_test"])
        # Measure the performance
        return metrics.accuracy_score(self.data["y_test"], y_predict)

my_bounds = [
    FloatVar(lb=0.01, ub=1000., name="C_paras"),
    StringVar(valid_sets=('linear', 'poly', 'rbf', 'sigmoid'), name="kernel_paras"),
    IntegerVar(lb=1, ub=5, name="degree_paras"),
    CategoricalVar(valid_sets=('scale', 'auto', 0.01, 0.05, 0.1, 0.5, 1.0), name="gamma_paras"),
    BoolVar(n_vars=1, name="probability_paras"),
]
problem = SvmOptimizedProblem(bounds=my_bounds, minmax="max", data=data)
model = SMA.OriginalSMA(epoch=100, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")
print(f"Best solution: {model.g_best.solution}")
print(f"Best accuracy: {model.g_best.target.fitness}")
print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")
