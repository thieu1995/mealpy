Stopping Condition (Termination)
================================

.. toctree::
   :maxdepth: 3


In meta-heuristic algorithms, the optimization process involves iteratively generating and evolving a population of candidate solutions (individuals) to the problem.
Each generation consists of evaluating the fitness of each individual, selecting the best individuals for reproduction, and applying specific operators to generate a new population.

By setting a maximum number of generations as a stopping condition, the algorithm will terminate after a certain number of iterations, even if a
satisfactory solution has not been found. This is highly useful to prevent the algorithm from running indefinitely, especially if there are no
clear convergence criteria or if the fitness landscape is complex and difficult to navigate.

.. important::
    **Resource Management & Convergence**

    The choice of the maximum number of generations must be based on the specific problem being solved and the available computational resources. A number that is too small may prevent the algorithm from converging to a satisfactory solution, while a number that is too large may result in unnecessary computational expense.

.. note::
    **Default Behavior**

    By default, when creating an optimizer in MEALPY, the default stopping condition is based purely on **epochs** (generations/iterations).

.. attention::
    **Advanced Termination Strategies**

    You can explore different stopping conditions by creating a ``Termination`` dictionary. Furthermore, you can combine **multiple stopping criteria together** to improve and strictly control your model's execution time!

Termination Types
-----------------

There are 4 termination types available in the ``Termination`` class:

* **MG (Maximum Generations):** Also known as Epochs or Iterations.
* **FE (Function Evaluations):** The maximum number of times the objective function is evaluated.
* **TB (Time Bound):** Forces the algorithm to run for a fixed amount of time (e.g., *K* seconds). This is especially useful when benchmarking or comparing different algorithms.
* **ES (Early Stopping):** Inspired by neural network training, this stops the program if the global best solution has not improved by an :math:`\epsilon` margin after *K* consecutive epochs.

Configuration Parameters
------------------------

To configure these conditions, provide the following parameters to the ``Termination`` class (set any parameter to ``None`` if you do not want to use it):

* ``max_epoch`` *(int)*: Indicates the maximum number of generations for the **MG** type.
* ``max_fe`` *(int)*: Indicates the maximum number of function evaluations for the **FE** type.
* ``max_time`` *(float)*: Indicates the maximum amount of time in seconds for the **TB** type.
* ``max_early_stop`` *(int)*: Indicates the maximum number of epochs for the **ES** type.
* ``epsilon`` *(float)*: *(Optional)* The tolerance value used for the **ES** termination type (default value: ``1e-10``).
* ``termination`` *(dict)*: *(Optional)* A dictionary encompassing your chosen termination criteria.


**1. MG (Maximum Generations / Epochs): This is default in all algorithms**

.. code-block:: python

   term_dict = {  # When creating this object, it will override the default epoch you define in your model
      "max_epoch": 1000  # 1000 epochs
   }

**2. FE (Number of Function Evaluation)**

.. code-block:: python

   term_dict = {
      "max_fe": 100000    # 100000 number of function evaluation
   }

**3. TB (Time Bound): If you want your algorithm to run for a fixed amount of time (e.g., K seconds), especially when comparing different algorithms.**

.. code-block:: python

   term_dict = {
      "max_time": 60  # 60 seconds to run this algorithm only
   }

**4. ES (Early Stopping): Similar to the idea in training neural networks (stop the program if the global best solution has not improved by epsilon after K epochs).**

.. code-block:: python

   term_dict = {
      "max_early_stop": 30  # after 30 epochs, if the global best doesn't improve then we stop the program
   }

**Setting multiple stopping criteria together. The first one that occurs will be used.**

.. code-block:: python

   # Use max epochs and max function evaluations together
   term_dict = {
      "max_epoch": 1000,
      "max_fe": 60000
   }

   # Use max function evaluations and time bound together
   term_dict = {
      "max_fe": 60000,
      "max_time": 40
   }

   # Use max function evaluations and early stopping together
   term_dict = {
      "max_fe": 55000,
      "max_early_stop": 15
   }

   # Use max epochs, max FE and early stopping together
   term_dict = {
      "max_epoch": 1200,
      "max_fe": 55000,
      "max_early_stop": 25
   }

   # Use all available stopping conditions together
   term_dict = {
      "max_epoch": 1100,
      "max_fe": 80000,
      "max_time": 10.5,
      "max_early_stop": 25
   }


**After import and create a termination object, and an optimizer object, you can pass termination object to solve() function**

.. code-block:: python

   model3 = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03)
   model3.solve(problem_dict1, termination=term_dict)
