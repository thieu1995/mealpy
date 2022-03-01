===============
Video Tutorials
===============

--------------------
Mealpy Tutorial Full
--------------------

* Part 1: |link_p1|, Part 2: |link_p2|

.. |link_p1| raw:: html

   <a href="https://www.youtube.com/watch?v=wh-C-57D_EM" target="_blank">Youtube P1</a>

.. |link_p2| raw:: html

   <a href="https://www.youtube.com/watch?v=TAUlSykOjeI" target="_blank">Youtube P1</a>

* Please read the description in the video for timestamp notes

* Or watch the |Full Video| with timestamp notes below:

.. image:: https://img.youtube.com/vi/HWc-yNcyPLw/0.jpg
   :target: https://www.youtube.com/watch?v=HWc-yNcyPLw

.. |Full video| raw:: html

   <a href="https://www.youtube.com/watch?v=HWc-yNcyPLw" target="_blank">Full Video</a>


::

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


-----------------------
Mealpy + Neural Network
-----------------------


Gradient Descent Replacement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Metaheuristic Algorithm in general can replace the Gradient Descent optimization to train the neural network. |youtube_1|

.. image:: https://img.youtube.com/vi/auq7Na1Meus/0.jpg
   :target: https://www.youtube.com/watch?v=auq7Na1Meus

.. |youtube_1| raw:: html

   <a href="https://www.youtube.com/watch?v=auq7Na1Meus" target="_blank">Youtube Link</a>


- For Time-Series Problem:

   * Traditional Multilayer Perceptron (MLP): |link_code_1|
   * Hybrid MLP Model (Mealpy + MLP): |link_code_2|

.. |link_code_1| raw:: html

   <a href="https://github.com/thieu1995/mealpy/tree/master/examples/applications/keras/traditional-mlp-time-series.py" target="_blank">Link Code</a>

.. |link_code_2| raw:: html

   <a href="https://github.com/thieu1995/mealpy/tree/master/examples/applications/keras/mha-hybrid-mlp-time-series.py" target="_blank">Link Code</a>


- For Classification Problem:

   * Traditional Multilayer Perceptron (MLP): |link_code_3|
   * Hybrid MLP Model (Mealpy + MLP): |link_code_4|

.. |link_code_3| raw:: html

   <a href="https://github.com/thieu1995/mealpy/blob/master/examples/applications/keras/traditional-mlp-classification.py" target="_blank">Link Code</a>

.. |link_code_4| raw:: html

   <a href="https://github.com/thieu1995/mealpy/blob/master/examples/applications/keras/mha-hybrid-mlp-classification.py" target="_blank">Link Code</a>



Optimize ANN Hyper-parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Metaheuristic Algorithm also can optimize Hyper-parameter of Neural Network. |youtube_3|

.. image:: https://img.youtube.com/vi/Fl3h9t087Pk/0.jpg
   :target: https://www.youtube.com/watch?v=Fl3h9t087Pk

.. |youtube_3| raw:: html

   <a href="https://www.youtube.com/watch?v=Fl3h9t087Pk" target="_blank">Youtube Link</a>

- |link_code_5| for Classification problem.

.. |link_code_5| raw:: html

   <a href="https://github.com/thieu1995/mealpy/blob/master/examples/applications/keras/mha-hyper-parameter-mlp-time-series.py" target="_blank">Link Code</a>




-------------------------
Other Mealpy Applications
-------------------------

* Solving Knapsack Problem (Discrete problems): |link_code_6|

* Optimize SVM (SVC) model: |link_code_7|

* Optimize Linear Regression Model: |link_code_8|


.. |link_code_6| raw:: html

   <a href="https://github.com/thieu1995/mealpy/blob/master/examples/applications/discrete-problems/knapsack-problem.py" target="_blank">Link Code</a>

.. |link_code_7| raw:: html

   <a href="https://github.com/thieu1995/mealpy/blob/master/examples/applications/sklearn/svm_classification.py" target="_blank">Link Code</a>

.. |link_code_8| raw:: html

   <a href="https://github.com/thieu1995/mealpy/blob/master/examples/applications/pytorch/linear_regression.py" target="_blank">Link Code</a>