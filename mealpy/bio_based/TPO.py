import numpy as np


class UpdatedTPO:
    """
        The updated Tree Physiology Optimization (TPO) published by
        A. Hanif Halim and I. Ismail on November 9, 2017.

    Notes
    _____
    The `alpha`, `beta` and `theta` should fine-tune to get faster
    convergence toward the global optimum. A good approximate range for
    `alpha` is [0.3, 3], for `beta` [20, 70] and for `theta` [0.7, 0.99].

    Examples
    --------
    >>> def obj_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": obj_function,
    >>>     "n_dims": 5,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> alpha = 0.4
    >>> beta = 50
    >>> theta = 0.95
    >>> num_branches = 50
    >>> num_leaves = 40
    >>> epoch = 50
    >>> model1 = UpdatedTPO(problem_dict1, num_branches, num_leaves, epoch, alpha, beta, theta)
    >>> solution = model1.solve()
    >>> print(solution)

    References
    __________
    [1] Halim, A. Hanif and Ismail, I. "Tree Physiology Optimization in Benchmark
    Function and Traveling Salesman Problem" Journal of Intelligent Systems, vol. 28,
    no. 5, 2019, pp. 849-871.
    """

    def __init__(self, problem, num_branches, num_leaves, epoch, alpha, beta, theta, **kwargs):
        """
        Initialize the algorithm components using a uniform distribution.

        Parameters
        ----------
        problem : dict
            Problem that conforms with the format of the Problem class
        num_branches : int
            Number of branches to have in the shoot system.
        num_leaves : int
            Number of leaves to have on each branch.
        epoch : int
            The total number iterations to make.
        alpha : float
            Absorption factor used for the root elongation.
        beta : int or float
            Factor by which shoots are extended as a response to the
            nutrients coming from the root.
        theta: float
            The rate of absorption in the carbon production in shoots
            and nutrient generation in roots.

        """
        self.num_branches = num_branches
        self.num_leaves = num_leaves
        self.num_iterations = epoch
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.dimension = problem["n_dims"]
        self.func = problem["fit_func"]
        self.lb = np.array(problem["lb"]).reshape(-1, 1, 1)
        self.ub = np.array(problem["ub"]).reshape(-1, 1, 1)
        self.shoots = np.random.uniform(0, 5, (self.dimension, num_branches, num_leaves))
        self.roots = np.random.uniform(0, 5, (self.dimension, num_branches, num_leaves))
        self.nutrient_value = np.random.uniform(0, 5, (self.dimension, num_branches, num_leaves))

    def trim_values(self):
        """
        Trim the values of shoots to make sure they stay within the
        specified upper and lower bounds.

        """
        self.shoots = np.maximum(self.shoots, self.lb)
        self.shoots = np.minimum(self.shoots, self.ub)

    def solve(self):
        """
        Minimize the objective function specified in the problem.

        Returns
        -------
        numpy.array
            Best solutions of the problem found after specified epochs.

        """
        func_value = np.empty((self.num_branches, self.num_leaves), dtype=np.float32)
        for i in range(self.num_branches):
            for j in range(self.num_leaves):
                func_value[i, j] = self.func(self.shoots[:, i, j])

        rows = np.arange(start=0, stop=self.num_branches)
        branch_best_idx_old = np.argmin(func_value, axis=1)
        branch_best_value_old = func_value[rows, branch_best_idx_old]
        branch_best_shoot_old = self.shoots[:, rows, branch_best_idx_old]
        global_best_idx = np.argmin(branch_best_value_old)
        global_best_shoots = branch_best_shoot_old[:, global_best_idx]
        self.shoots = global_best_shoots.reshape(-1, 1, 1) + self.beta * self.nutrient_value
        self.trim_values()
        current_theta = self.theta
        for _ in range(self.num_iterations):
            for i in range(self.num_branches):
                for j in range(self.num_leaves):
                    func_value[i, j] = self.func(self.shoots[:, i, j])

            branch_best_idx_new = np.argmin(func_value, axis=1)
            branch_best_value_new = func_value[rows, branch_best_idx_new]
            branch_best_shoot_new = self.shoots[:, rows, branch_best_idx_new]
            better_branches = branch_best_value_new < branch_best_value_old
            branch_best_value_old[better_branches] = branch_best_value_new[better_branches]
            branch_best_shoot_old[:, better_branches] = branch_best_shoot_new[:, better_branches]
            branch_best_idx_old[better_branches] = branch_best_idx_new[better_branches]
            global_best_idx = np.argmin(branch_best_value_old)
            global_best_shoots = branch_best_shoot_old[:, global_best_idx]
            carbon_gain = current_theta * (
                    branch_best_shoot_old.reshape(self.dimension, self.num_branches, 1) - self.shoots)
            roots_old = np.copy(self.roots)
            self.roots += self.alpha * carbon_gain * np.random.uniform(
                low=-0.5, high=0.5, size=(self.dimension, self.num_branches, self.num_leaves))
            nutrient_value = current_theta * (self.roots - roots_old)
            self.shoots = global_best_shoots.reshape(self.dimension, 1, 1) + self.beta * nutrient_value
            self.trim_values()
            current_theta *= self.theta

        return global_best_shoots
