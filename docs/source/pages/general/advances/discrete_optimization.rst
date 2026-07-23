Discrete Optimization
=====================

.. toctree::
   :maxdepth: 3

For this type of problem, we recommend creating a custom child class of the ``Problem`` class and overriding the necessary functions. At a minimum, the following three functions are typically considered for overriding:

* ``obj_func(self, solution)``: The fitness function to evaluate the solution.
* ``generate_position(self, lb, ub)``: A function that generates a valid initial solution.
* ``amend_position(self, solution, lb, ub)``: A function that brings the solution back within valid boundaries.

.. important::
    **Decision Variables: No Need to Override Core Functions!**

    Before you dive into overriding ``generate_position`` and ``amend_position`` manually, let's look at the tools MEALPY provides. MEALPY has natively implemented almost all types of **decision variables** you might need.

    By simply selecting the appropriate Variable class and passing it to the ``bounds`` parameter, MEALPY handles the initialization (``generate_position``) and boundary constraints (``amend_position``) automatically under the hood. Most of the time, **you only need to override the ``obj_func``!**

To assist you in choosing the right tools, refer to the table below. It outlines different types of decision variables available in MEALPY, along with their syntax and common problem applications.

.. list-table:: Available Decision Variables in MEALPY
   :widths: 20 55 25
   :header-rows: 1

   * - Class
     - Syntax
     - Problem Types
   * - **FloatVar**
     - ``FloatVar(lb=(-10., )*7, ub=(10., )*7, name="delta")``
     - Continuous Problem
   * - **IntegerVar**
     - ``IntegerVar(lb=(-10., )*7, ub=(10., )*7, name="delta")``
     - LP, IP, NLP, QP, MIP
   * - **StringVar**
     - ``StringVar(valid_sets=(("auto", "backward", "forward"), ("leaf", "branch", "root")), name="delta")``
     - ML, AI-optimize
   * - **BinaryVar**
     - ``BinaryVar(n_vars=11, name="delta")``
     - Networks
   * - **BoolVar**
     - ``BoolVar(n_vars=11, name="delta")``
     - ML, AI-optimize
   * - **PermutationVar**
     - ``PermutationVar(valid_set=(-10, -4, 10, 6, -2), name="delta")``
     - Combinatorial Optimization
   * - **CategoricalVar**
     - ``CategoricalVar(valid_sets=(("auto", 2, 3, "backward", True), (0, "tournament", "round-robin")), name="delta")``
     - MIP, MILP
   * - **SequenceVar**
     - ``SequenceVar(valid_sets=((1, ), {2, 3}, [3, 5, 1]), return_type=list, name='delta')``
     - Hyper-parameter tuning
   * - **TransferBoolVar**
     - ``TransferBoolVar(n_vars=11, name="delta", tf_func="sstf_02")``
     - ML, AI-optimize, Feature Selection
   * - **TransferBinaryVar**
     - ``TransferBinaryVar(n_vars=11, name="delta", tf_func="vstf_04")``
     - Networks, Feature Selection

Example: Travelling Salesman Problem (TSP)
------------------------------------------

Let's say we want to solve the Travelling Salesman Problem (TSP). Because this is a permutation problem, we can simply utilize MEALPY's built-in ``PermutationVar``. This specific variable type automatically handles the generation and boundary-repair logic for us, meaning we only have to write the objective function.

.. code-block:: python

    import numpy as np
    from mealpy import Problem, PermutationVar, PSO

    class DOP(Problem):
        def __init__(self, bounds, minmax, CITY_POSITIONS=None, **kwargs):
            super().__init__(bounds, minmax, **kwargs)
            self.CITY_POSITIONS = CITY_POSITIONS

        def obj_func(self, solution):
            ## Objective for this problem is the sum of distance between all cities that the salesman has passed.
            ## This can be changed depending on your specific requirements.
            
            # Decode the solution based on the PermutationVar definition
            x = self.decode_solution(solution)["per"]
            
            # Get the coordinates of the cities in the order of the permutation
            city_coord = self.CITY_POSITIONS[x]
            line_x = city_coord[:, 0]
            line_y = city_coord[:, 1]
            
            # Calculate the Euclidean distance between consecutive cities
            total_distance = np.sum(np.sqrt(np.square(np.diff(line_x)) + np.square(np.diff(line_y))))
            return total_distance

    ## Define coordinates for 13 random cities for this example
    random_cities = np.random.rand(13, 2) * 100 

    ## Create an instance of the DOP class
    ## For the Travelling Salesman Problem, the solution should be a permutation of city indices
    problem_cop = DOP(
        bounds=PermutationVar(valid_set=list(range(13)), name="per"),
        minmax="min", 
        CITY_POSITIONS=random_cities,
        log_to="file", 
        log_file="dop-results.txt"
    )

    ## Define the model and solve the problem
    model = PSO.OriginalPSO(epoch=1000, pop_size=50)
    model.solve(problem=problem_cop)
