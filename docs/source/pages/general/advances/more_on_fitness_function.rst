More on Fitness Function
========================

Usually, when defining a fitness function we only need 1 parameter which is the solution.

.. code-block:: python

	def fitness_function(solution):
		fitness = np.sum(solution**2)
	    return fitness

But what if we need to pass additional data to the fitness function to calculate the fitness value?
as in the case of calculating the loss of neural network as fitness value, we need to pass the dataset into this function.

In previous version, we deal with this problem by writing all the code in the same file python and use DATASET as global variable.

But from version 2.4.2, you can define your data (whatever it is) as an input parameter to fitness function, such as.

.. code-block:: python

   from mealpy.swarm_based import PSO

   def fitness_function(solution, data):
       dataset = data['dataset']
       additional_infor = data['additional-information']
       network = NET(dataset, additional_infor)
       fitness = network.loss
       return fitness

   DATA = {
       "dataset": dataset,
       "additional-information": temp,
   }

   problem = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "data": DATA,     # Remember this keyword 'data'
   }

   model = PSO.BasePSO(problem, epoch=10, pop_size=50)
   model.solve()

**Notes**: As you can see, any data or information should store in same dictionary (Recommended) and then pass it to Problem object.
And then you can get it by key in fitness function.


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4



