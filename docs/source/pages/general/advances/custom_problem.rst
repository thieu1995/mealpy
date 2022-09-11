Custom Problem
==============

For a complex problem, we recommend user to define and custom child class of Problem class instead of defining the problem dictionary.

For example, training neural network will required the dataset passing to the fitness function. Defining a child class can pass any additional data that you
need.

.. code-block:: python

	from mealpy.swarm_based import PSO
	from mealpy.utils.problem import Problem

	class NeuralNetwork(Problem):
	    def __init__(self, lb, ub, minmax, name="NeuralNetwork", dataset=None, additional=None, **kwargs):
	        super().__init__(lb, ub, minmax, **kwargs)
	        self.name = name
			self.dataset = dataset
			self.additional = additional

	    def fit_func(self, solution):
			network = NET(self.dataset, self.additional)
			fitness = network.loss
			return fitness

	## Create an instance of MOP class
	problem_cop = COP(lb=[-3, -5, 1, -10, ], ub=[5, 10, 100, 30, ], name="Network",
					dataset=dataset, additional=additional, minmax="min")

	## Define the model and solve the problem
	model = PSO.OriginalPSO(epoch=1000, pop_size=50)
	model.solve(problem=problem_cop)


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4



