# !/usr/bin/env python
# Created by "Thieu" at 17:00, 27/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

## Link: https://vitalflux.com/classification-model-svm-classifier-python-example/

# Sklearn modules & classes
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import datasets, metrics
from mealpy.bio_based import SMA

# Load the data set; In this example, the breast cancer dataset is loaded.
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# LABEL ENCODER
KERNEL_ENCODER = LabelEncoder()
KERNEL_ENCODER.fit(['linear', 'poly', 'rbf', 'sigmoid'])
# print(KERNEL_ENCODER.inverse_transform( [1, 3]))


def fitness_function(solution):
    # if kernel belongs to 0 - 0.99 ==> 0       ==> linear
    #                       2 - 2.99 ==> 2
    #                       3 - 3.99 ==> 3      ==> sigmoid

    kernel_encoded = int(solution[0])
    c = solution[1]
    kernel_decoded = KERNEL_ENCODER.inverse_transform([kernel_encoded])[0]

    svc = SVC(C=c, random_state=1, kernel=kernel_decoded)
    # Fit the model
    svc.fit(X_train_std, y_train)
    # Make the predictions
    y_predict = svc.predict(X_test_std)
    # Measure the performance
    return metrics.accuracy_score(y_test, y_predict)

problem = {
    "fit_func": fitness_function,
    "lb": [0, 0.1],
    "ub": [3.99, 1000],
    "minmax": "max",
    "verbose": True,
}

model = SMA.BaseSMA(problem, epoch=50, pop_size=50)
model.solve()
print(f"Best solution: {model.solution[0]}")
print(f"Best kernel: {KERNEL_ENCODER.inverse_transform([int(model.solution[0][0])])[0]}, Best c: {model.solution[0][1]}")

print(f"Best accuracy: {model.solution[1]}")


# Brute-force method
# Instantiate the Support Vector Classifier (SVC)
# C in R+
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# C = [1, 10, 100, 1000]

# for c in C:
#     for kernel in kernels:
#         svc = SVC(C=c, random_state=1, kernel=kernel)
#
#         # Fit the model
#         svc.fit(X_train_std, y_train)
#
#         # Make the predictions
#         y_predict = svc.predict(X_test_std)
#
#         # Measure the performance
#         print("Accuracy score %.3f" % metrics.accuracy_score(y_test, y_predict))

