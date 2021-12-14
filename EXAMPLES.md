### Tutorial Videos

* Part 1: [Link](https://www.youtube.com/watch?v=wh-C-57D_EM)
* Part 2: [Link](https://www.youtube.com/watch?v=TAUlSykOjeI)
* Please read the description in the video for timestamp notes

* Or watch the full video with timestamp notes below: [Link](https://www.youtube.com/watch?v=HWc-yNcyPLw)

```python 
0:00 - Intro
0:19 - Download and install Miniconda on Windows 11
1:22 - Create a new environment using Miniconda
2:32 - Install Mealpy
5:08 - Pycharm and set environment on it
9:22 - Introducing the structure of Mealpy library
10:16 - The Optimizer class
10:50 - The Problem class
11:44 - The Termination class
15:10 - The History class (How to draw figures)
16:37 - How to import the mealpy library (Optimizer class)
18:32 - Define a problem dictionary (problem instance of Problem class)
19:32 - Define objective-function 
21:18 - Problem definition (Find minimum of Fx function)
23:10 - How to call an optimizer to solve optimization problem 
25:38 - The Problem class
26:23 - Sequential, Thread and Process training mode setting
28:23 - Explaining the current best and global best (training output)
29:18 - How to get final fitness and final position (solution)
30:38 - The structure of the "solution" attribute in Optimizer class
33:48 - Other ways to pass Lowerbound and Upperbound in problem dictionary
36:05 - How to import and define the Termination object
43:08 - Time-bound termination object
45:16 - Early Stopping termination object
47:18 - How to use Sequential/MultiThreading/MultiProcessing training mode
51:58 - Fix error with MultiProcessing training mode 
55:54 - How to deal with Multi-objective Optimization Problem
1:05:09 - How to deal with Constrained Optimization Problem
1:11:46 - How to draw some important figures using History object
1:23:15 - How to use Mealpy to optimize hyper-parameters of a model
1:26:15 - Using Mealpy to optimization hyper-parameters of a traditional SVM classification
1:30:18 - Brute force method for tunning hyper-parameters
1:36:18 - GridSearchCV method for tunning hyper-parameters
1:39:28 - Metaheuristic Algorithm method for tunning hyper-parameters
```




### Example

* Please don't misunderstand between parameters (hyper-parameters) and variables.
* Assumption that you have to find minimum of function F(x) = x1^3 + x2^2 + x3^4 with
  (-1 <= x1 <= 4), (5 <= x2 <= 10) and (-7 <= x2 <= -4). Then

    * Your solution is x = [x1, x2, x3], x1, x2, x3 here are the variables.
    * The number of dimension (problem size) = 3 (variables)
    * Your fitness value is fx = F(x)
    * lower bound and upper bound: lb = [-1, 5, -7] and ub = [4, 10, -4]
    * parameters (hyper-parameters) is depended on each algorithm.
    * objective function here is F(x) for minimize problem.

* **And PLEASE read some examples inside folder "examples" before email asking me how to call the optimizer. Lots of
  simple and complicated examples there. Take your time to learn how to use it.**

```python 
# Define an objective function, for example above:
def Fx(solution):
  fx = solution[0] ** 3 + solution[1] ** 2 + solution[2] ** 4
  return fx 
```

## Version Mealpy >= 2.0.0

* The batch-size idea is removed due to the new feature which is parallel training. 

```python 

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.bio_based import SMA
from mealpy.problem import Problem
from mealpy.utils.termination import Termination

# Setting parameters

# A - Different way to provide lower bound and upper bound. Here are some examples:

## A1. When you have different lower bound and upper bound for each parameters
problem_dict1 = {
    "obj_func": F5,
    "lb": [-3, -5, 1, -10, ],
    "ub": [5, 10, 100, 30, ],
    "minmax": "min",
    "verbose": True,
}
problem_obj1 = Problem(problem_dict1)

## A2. When you have same lower bound and upper bound for each parameters, then you can use:
##      + int or float: then you need to specify your problem size / number of dimensions (n_dims)
problem_dict2 = {
    "obj_func": F5,
    "lb": -10,
    "ub": 30,
    "minmax": "min",
    "verbose": True,
    "n_dims": 30,  # Remember the keyword "n_dims"
}
problem_obj2 = Problem(problem_dict2)

##      + array: 2 ways
problem_dict3 = {
    "obj_func": F5,
    "lb": [-5],
    "ub": [10],
    "minmax": "min",
    "verbose": True,
    "n_dims": 30,  # Remember the keyword "n_dims"
}
problem_obj3 = Problem(problem_dict3)

n_dims = 100
problem_dict4 = {
    "obj_func": F5,
    "lb": [-5] * n_dims,
    "ub": [10] * n_dims,
    "minmax": "min",
    "verbose": True,
}

## Run the algorithm

### Your parameter problem can be an instane of Problem class or just dict like above
model1 = SMA.BaseSMA(problem_obj1, epoch=100, pop_size=50, pr=0.03)
model1.solve()

model2 = SMA.BaseSMA(problem_dict4, epoch=100, pop_size=50, pr=0.03)
model2.solve()

# B - Test with different Stopping Condition (Termination) by creating an Termination object

## There are 4 termination cases:
### 1. FE (Number of Function Evaluation)
### 2. MG (Maximum Generations / Epochs): This is default in all algorithms
### 3. ES (Early Stopping): Same idea in training neural network (If the global best solution not better an epsilon
###     after K epoch then stop the program
### 4. TB (Time Bound): You just want your algorithm run in K seconds. Especially when comparing different algorithms.

termination_dict1 = {
    "mode": "FE",
    "quantity": 100000  # 100000 number of function evaluation
}
termination_dict2 = {  # When creating this object, it will override the default epoch you define in your model
    "mode": "MG",
    "quantity": 1000  # 1000 epochs
}
termination_dict3 = {
    "mode": "ES",
    "quantity": 30  # after 30 epochs, if the global best doesn't improve then we stop the program
}
termination_dict4 = {
    "mode": "ES",
    "quantity": 60  # 60 seconds = 1 minute to run this algorithm only
}
termination_obj1 = Termination(termination_dict1)
termination_obj2 = Termination(termination_dict2)
termination_obj3 = Termination(termination_dict3)
termination_obj4 = Termination(termination_dict4)

### Pass your termination object into your model as a addtional parameter with the keyword "termination"
model3 = SMA.BaseSMA(problem_dict1, epoch=100, pop_size=50, pr=0.03, termination=termination_obj1)
model3.solve()
### Remember you can't pass termination dict, it only accept the Termination object


# C - Test with different training mode (sequential, threading parallelization, processing parallelization)

## + sequential: Default for all algorithm (single core)
## + thread: create multiple threading depend on your chip
## + process: create multiple cores to run your algorithm.
## Note: For windows, your program need the if __nam__ == "__main__" condition to avoid creating infinite processors

model5 = SMA.BaseSMA(problem_dict1, epoch=100, pop_size=50, pr=0.03)
model5.solve(mode='sequential')  # Default

model6 = SMA.BaseSMA(problem_dict1, epoch=100, pop_size=50, pr=0.03)
model6.solve(mode='thread')

if __name__ == "__main__":
    model7 = SMA.BaseSMA(problem_dict1, epoch=100, pop_size=50, pr=0.03)
    model7.solve(mode='process')

# D - Drawing all available figures

## There are 8 different figures for each algorithm.
## D.1: Based on fitness value:
##      1. Global best fitness chart
##      2. Local best fitness chart
## D.2: Based on objective value:
##      3. Global objective chart
##      4. Local objective chart
## D.3: Based on runtime value (runtime for each epoch)
##      5. Runtime chart
## D.4: Based on exploration verse exploration value
##      6. Exploration vs Exploitation chart
## D.5: Based on diversity of population
##      7. Diversity chart
## D.6: Based on trajectory value (1D, 2D only)
##      8. Trajectory chart

model8 = SMA.BaseSMA(problem_dict1, epoch=100, pop_size=50, pr=0.03)
model8.solve()

## You can access them all via object "history" like this:
model8.history.save_global_objectives_chart(filename="hello/goc")
model8.history.save_local_objectives_chart(filename="hello/loc")
model8.history.save_global_best_fitness_chart(filename="hello/gbfc")
model8.history.save_local_best_fitness_chart(filename="hello/lbfc")
model8.history.save_runtime_chart(filename="hello/rtc")
model8.history.save_exploration_exploitation_chart(filename="hello/eec")
model8.history.save_diversity_chart(filename="hello/dc")
model8.history.save_trajectory_chart(list_agent_idx=[3, 5], list_dimensions=[3], filename="hello/tc")


# E - Handling Multi-Objective function and Constraint Method

## To handling Multi-Objective, mealpy is using weighting method which converting multiple objectives to a single target (fitness value)

## Define your objective function, your constraint
def obj_function(solution):
    t1 = solution[0] ** 2
    t2 = ((2 * solution[1]) / 5) ** 2
    t3 = 0
    for i in range(3, len(solution)):
        t3 += (1 + solution[i] ** 2) ** 0.5
    return [t1, t2, t3]


## Define your objective weights. For example:
###  f1: 50% important
###  f2: 20% important
###  f3: 30% important
### Then weight = [0.5, 0.2, 0.3] ==> Fitness value = 0.5*f1 + 0.2*f2 + 0.3*f3
### Default weight = [1, 1, 1]

problem_dict9 = {
    "obj_func": obj_function,
    "lb": [-3, -5, 1, -10, ],
    "ub": [5, 10, 100, 30, ],
    "minmax": "min",
    "verbose": True,
    "obj_weight": [0.5, 0.2, 0.3]  # Remember the keyword "obj_weight"
}
problem_obj9 = Problem(problem_dict9)
model9 = SMA.BaseSMA(problem_obj9, epoch=100, pop_size=50, pr=0.03)
model9.solve()

## To access the results, you can get the results by solve() method
position, fitness_value = model9.solve()

## To get all fitness value and all objective values, get it via "solution" attribute
## A agent / solution format [position, [fitness, [obj1, obj2, ..., obj_n]]]
position = model9.solution[0]
fitness_value = model9.solution[1][0]
objective_values = model9.solution[1][1]


# F - Test with different variants of this algorithm

model10 = SMA.OriginalSMA(problem_obj9, epoch=100, pop_size=50, pr=0.03)
model10.solve()

```




## Version 1.1.0 <= Mealpy <= 1.2.2

* The batch-size idea is not existed in Meta-heuristics field. I just take an inspiration from training batch-size of
  neural network field and combine it with metaheuristics. Therefore, some algorithms will have it, some won't. Don't
  worry, if you don't want to use it, just call the algorithm like usual, you don't need to specify any additional
  parameters. But if you want to use it, check the example above, you need to specify some additional hyper-parameters.

* The batch-size idea exited in this version only.

```python

# This is basic example how you can call an optimizer, and its variants. For the version ( MEALPY >= 1.1.0)

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.swarm_based.PSO import BasePSO, PPSO, P_PSO, HPSO_TVAC

# Setting parameters
obj_func = F5  # This objective function come from "opfunu" library. You can design your own objective function like above
verbose = False  # Print out the training results
epoch = 500  # Number of iterations / generations / epochs
pop_size = 50  # Populations size (Number of individuals / Number of solutions)

# A - Different way to provide lower bound and upper bound. Here are some examples:

## 1. When you have different lower bound and upper bound for each variables
lb1 = [-3, -5, 1]
ub1 = [5, 10, 100]

md1 = BasePSO(obj_func, lb1, ub1, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[1])

## 2. When you have same lower bound and upper bound for each variables, then you can use:
##      + int or float: then you need to specify your problem size (number of dimensions)
problemSize = 10
lb2 = -5
ub2 = 10
md2 = BasePSO(obj_func, lb2, ub2, verbose, epoch, pop_size,
              problem_size=problemSize)  # Remember the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md2.train()
print(md2.solution[1])

##      + array: 2 ways
lb3 = [-5]
ub3 = [10]
md3 = BasePSO(obj_func, lb3, ub3, verbose, epoch, pop_size,
              problem_size=problemSize)  # Remember the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md3.train()
print(md3.solution[1])

lb4 = [-5] * problemSize
ub4 = [10] * problemSize
md4 = BasePSO(obj_func, lb4, ub4, verbose, epoch, pop_size)  # No need the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md4.train()
print(md4.solution[1])

# B - Test with algorithm has batch size idea

## 1. Not using batch size idea

md5 = BasePSO(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md5.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

## 2. Using batch size idea
batchIdea = True
batchSize = 5

md6 = BasePSO(obj_func, lb4, ub4, verbose, epoch, pop_size, batch_idea=batchIdea,
              batch_size=batchSize)  # Remember the keywords
best_pos1, best_fit1, list_loss1 = md6.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

# C - Test with different variants of this algorithm

md1 = PPSO(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = PSO_W(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = HPSO_TVA(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)
```

## Mealpy <= 1.0.5 

```python 
# Simple example: this is for previous version ( version <= 1.0.5)

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.evolutionary_based.GA import BaseGA

## Setting parameters
obj_func = F1
# lb = [-15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3]
# ub = [15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3]
lb = [-100]
ub = [100]
problem_size = 100
batch_size = 25
verbose = True
epoch = 1000
pop_size = 50
pc = 0.95
pm = 0.025

md1 = BaseGA(obj_func, lb, ub, problem_size, batch_size, verbose, epoch, pop_size, 0.85, 0.05)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

# Or run the simple:
python examples/run_simple.py
```
