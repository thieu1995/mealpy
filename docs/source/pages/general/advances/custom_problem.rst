Custom Problem
==============


For complex problems, we recommend that the user define a custom child class of the `Problem` class instead of defining the problem dictionary.
For instance, when training a neural network, the dataset needs to be passed to the fitness function. Defining a child class allows for passing any
additional data that may be needed.


.. code-block:: python

	from mealpy import PSO, FloatVar, Problem

	class NeuralNetwork(Problem):
	    def __init__(self, bounds=None, minmax, name="NeuralNetwork", dataset=None, additional=None, **kwargs):
	        self.name = name
			self.dataset = dataset
			self.additional = additional
	        super().__init__(bounds, minmax, **kwargs)

	    def obj_func(self, solution):
			network = NET(self.dataset, self.additional)
			obj = network.loss
			return obj

	## Create an instance of MOP class
	problem_cop = COP(bounds=FloatVar(lb=[-3, -5, 1, -10, ], ub=[5, 10, 100, 30, ]), name="Network",
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



