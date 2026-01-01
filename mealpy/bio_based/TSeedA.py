#!/usr/bin/env python
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalTSeedA(Optimizer):
    """
    Tree-Seed Algorithm (TSA) for continuous optimization.

    Paper:
        Kiran, M. S. (2015). TSA: Tree-seed algorithm for continuous optimization.
        Expert Systems with Applications, 42(19), 6686-6698. DOI: 10.1016/j.eswa.2015.04.055

    Parameters:
        epoch (int): maximum number of iterations, default = 10000
        pop_size (int): population size, default = 100
        st (float): search tendency in (0, 1), default = 0.1

    Equations:
        For each seed S_ij:
            if rand_j < st:  S_ij = T_ij + a_ij * (B_j - T_rj)   (Eq. 3)
            else:            S_ij = T_ij + a_ij * (T_ij - T_rj)   (Eq. 4)
        where T is the current tree, B is the global best tree so far, and r != i.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TSeedA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = TSeedA.OriginalTSeedA(epoch=1000, pop_size=50, st=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kiran, M. S. (2015). TSA: Tree-seed algorithm for continuous optimization.
    Expert Systems with Applications, 42(19), 6686-6698. DOI: 10.1016/j.eswa.2015.04.055

    BibTeX
    ~~~~~~
    @article{Kiran2015TSA,
        title={TSA: Tree-seed algorithm for continuous optimization},
        author={Kiran, Mustafa Servet},
        journal={Expert Systems with Applications},
        volume={42},
        number={19},
        pages={6686--6698},
        year={2015},
        doi={10.1016/j.eswa.2015.04.055}
    }
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, st: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: population size, default = 100
            st: search tendency in (0, 1), default = 0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.st = self.validator.check_float("st", st, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "st"])
        self.sort_flag = False

    def _get_random_tree_index(self, current_idx: int) -> int:
        r_idx = int(self.generator.integers(0, self.pop_size - 1))
        if r_idx >= current_idx:
            r_idx += 1
        return r_idx

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        """
        n_dims = self.problem.n_dims
        lb = self.problem.lb
        ub = self.problem.ub
        
        # Calculate number of seeds range
        low = max(1, int(np.ceil(self.pop_size * 0.1)))
        high = max(1, int(np.ceil(self.pop_size * 0.25)))
        if high < low:
            high = low

        for i in range(self.pop_size):
            # Determine number of seeds for this tree (Eq 1)
            # ns = int(np.fix(low + (high - low) * self.generator.random())) + 1
            n_seeds = int(np.fix(low + (high - low) * self.generator.random())) + 1
            if n_seeds > high:
                n_seeds = high
            
            # Find best solution in current population to use in Eq 3
            best_idx = np.argmin([agent.target.fitness for agent in self.pop])
            best_params = self.pop[best_idx].solution.copy()
            
            seeds_pop = []
            seeds_fitness = []
            
            for j in range(n_seeds):
                # Select a neighbor (komsu) different from i
                neighbor_idx = int(np.fix(self.generator.random() * self.pop_size))
                while i == neighbor_idx:
                    neighbor_idx = int(np.fix(self.generator.random() * self.pop_size))
                
                # Create a new seed from the current tree
                # Note: In original Matlab "seeds(j,:) = trees(j,:)" is used initially, 
                # but we use trees(i,:) as the base because we are generating seeds for tree i.
                # However, following the exact logic of the Matlab code where it iterates j for seeds:
                seed = self.pop[i].solution.copy() 

                # Evolve seed params (Eq 3 or Eq 4)
                for d in range(n_dims):
                    if self.generator.random() < self.st:
                        # Eq 3: Use global best
                        alpha = (self.generator.random() - 0.5) * 2
                        seed[d] = self.pop[i].solution[d] + alpha * (best_params[d] - self.pop[neighbor_idx].solution[d])
                    else:
                        # Eq 4: Use neighbor
                        alpha = (self.generator.random() - 0.5) * 2
                        seed[d] = self.pop[i].solution[d] + alpha * (self.pop[i].solution[d] - self.pop[neighbor_idx].solution[d])
                    
                    # Boundary check
                    if seed[d] > ub[d]:
                        seed[d] = ub[d]
                    if seed[d] < lb[d]:
                        seed[d] = lb[d]
                
                # Create agent for seed
                agent = self.generate_empty_agent(seed)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(seed)
                seeds_pop.append(agent)
                seeds_fitness.append(agent.target.fitness if agent.target else float('inf'))
            
            # Update part
            if self.mode in self.AVAILABLE_MODES:
                seeds_pop = self.update_target_for_population(seeds_pop)
                seeds_fitness = [agent.target.fitness for agent in seeds_pop]
            
            # Find the best seed (mintohum)
            best_seed_idx = int(np.argmin(seeds_fitness))
            best_seed_fitness = seeds_fitness[best_seed_idx]
            
            # If best seed is better than current tree, replace tree
            if best_seed_fitness < self.pop[i].target.fitness:
                self.pop[i].update(solution=seeds_pop[best_seed_idx].solution.copy(),
                                   target=seeds_pop[best_seed_idx].target.copy())

