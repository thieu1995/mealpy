Custom Problem
==============

.. toctree::
   :maxdepth: 3


While problem dictionaries are quick to set up, we highly recommend defining a custom child class of the ``Problem`` class for complex optimization tasks. 

.. important::
    **The Power of Encapsulation**

    Object-oriented design is crucial when your objective function depends on external variables or heavy data structures. For instance, when optimizing a Neural Network, you must pass datasets, hyperparameters, or training states to the fitness function. A custom class allows you to neatly encapsulate all this additional data via the ``__init__`` method, avoiding messy global variables.

Here is a practical example of how to implement a custom problem class for Neural Network weight optimization:

.. code-block:: python

    from mealpy import PSO, FloatVar, Problem

    class NeuralNetworkProblem(Problem):
        def __init__(self, bounds=None, minmax="min", dataset=None, additional=None, **kwargs):
            # 1. Initialize the parent Problem class
            super().__init__(bounds, minmax, **kwargs)
            # 2. Store your custom external data
            self.dataset = dataset
            self.additional = additional

        def obj_func(self, solution):
            # 3. Decode the solution and use your external data
            # weights = self.decode_solution(solution)["weights"]
            
            # (Conceptual Code) Initialize network and apply the optimizer's weights
            # network = NET(dataset=self.dataset, config=self.additional)
            # network.set_weights(weights)
            # loss = network.calculate_loss()
            
            loss = 0.5  # Dummy loss for demonstration
            return loss

    ## Mock data for demonstration
    my_dataset = {"X_train": [...], "y_train": [...]}
    my_config = {"learning_rate": 0.01, "batch_size": 32}

    ## Create an instance of your custom class
    problem_nn = NeuralNetworkProblem(
        bounds=FloatVar(lb=[-3, -5, 1, -10], ub=[5, 10, 100, 30], name="weights"), 
        minmax="min", 
        dataset=my_dataset, 
        additional=my_config,
        name="NN_Optimization"  # Extra kwargs are passed smoothly to the parent class
    )

    ## Define the model and solve the problem
    model = PSO.OriginalPSO(epoch=1000, pop_size=50)
    model.solve(problem=problem_nn)
