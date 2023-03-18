Stopping Condition (Termination)
================================

In meta-heuristic algorithms, the optimization process involves iteratively generating and evolving a population of candidate solutions (individuals) to the
problem. Each generation consists of evaluating the fitness of each individual, selecting the best individuals for reproduction, and applying some specific
operators to generate a new population. By setting a maximum number of generations as a stopping condition, the algorithm will terminate after a certain
number of iterations, even if a satisfactory solution has not been found. This can be useful to prevent the algorithm from running indefinitely, especially
if there is no clear convergence criteria or if the fitness landscape is complex and difficult to navigate.

However, it is important to note that the choice of the maximum number of generations should be based on the specific problem being solved, as well as the
computational resources available. A too small number of generations may not allow the algorithm to converge to a satisfactory solution, while a too large
number may result in unnecessary computational expense.


By default, when creating an optimizer, the default stopping condition (termination) is based on epochs (generations, iterations).

However, there are different stopping conditions that you can try by creating a Termination dictionary. You can also use multiple stopping criteria together
to improve your model. There are 4 termination types in the class Termination:

+ MG: Maximum Generations / Epochs / Iterations
+ FE: Maximum Number of Function Evaluations
+ TB: Time Bound - If you want your algorithm to run for a fixed amount of time (e.g., K seconds), especially when comparing different algorithms.
+ ES: Early Stopping -  Similar to the idea in training neural networks (stop the program if the global best solution has not improved by epsilon after K epochs).

+ Parameters for Termination class, set it to None if you don't want to use it
    + max_epoch (int): Indicates the maximum number of generations for the MG type.
    + max_fe (int): Indicates the maximum number of function evaluations for the FE type.
    + max_time (float): Indicates the maximum amount of time for the TB type.
    + max_early_stop (int): Indicates the maximum number of epochs for the ES type.
        + epsilon (float): (Optional) This is used for the ES termination type (default value: 1e-10).
    + termination (dict): (Optional) A dictionary of termination criteria.


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

   model3 = SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)
   model3.solve(problem_dict1, termination=term_dict)

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

