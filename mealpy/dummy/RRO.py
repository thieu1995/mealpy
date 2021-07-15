#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:21, 05/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, rand, choice
from numpy import zeros, power
from copy import deepcopy
from mealpy.optimizer import Root


class OriginalRRO(Root):
    """
        The original version of: Raven Roosting Optimization (RRO)
        Link:
            https://doi.org/10.1007/s00500-014-1520-5
        Questions:
            1. How to set the value of R? I guess R = (UB - LB) / 2
            2. How to handle the case raven fly outside of radius? I guess redo the fly from previous position.
            3. How to select Perception Follower? I guess randomly selected
            4. The Pseudo-code is wrong, 100%. After each iteration, create N random locations. For real?
            5. The sentence "There is a Prob_stop chance the raven will stop". What is Prob_stop. Not mention?
            6. The whole paper contains only simple equation: x(t) = x(t-1) + d. Really?
        Conclusion:
            The algorithm can't even converge for a simple problem (sphere function).
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 r_perception=3.6, r_leader=3.6, n_steps=20, weak_percent=0.4, prob_stop=0.2, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r_perception = r_perception    # Default: [1.8, 3.6], Factor that control the radius of perception
        self.r_leader = r_leader            # Default: [1.8, 3.6], Factor that control the radius of leader
        self.n_steps = n_steps              # Default: [5, 10, 20], Number of moving steps of each raven towards the best solution
        self.weak_percent = weak_percent    # Default: [0,2, 0.4, 0.6] Percentage of population will be moved towards by global best solution
        self.prob_stop = prob_stop          # The probability of stopping the fly

    def train(self):
        r_percept = ((self.ub - self.lb)/ 2) / (self.r_perception * power(self.pop_size, 1.0 / self.problem_size))
        r_leader = ((self.ub - self.lb) / 2) / (self.r_leader * power(self.pop_size, 1.0 / self.problem_size))
        n_ravens = int(self.weak_percent * self.pop_size)
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            # Select the random raven to fly to global best by r_leader
            idx_list = choice(range(0, self.pop_size), n_ravens, replace=False)

            for i in range(self.pop_size):

                if i in idx_list:       # Fly to global best
                    step = 0
                    while step < self.n_steps:
                        d_random = uniform(zeros(self.problem_size), r_leader, self.problem_size)
                        pos_new = pop[i][self.ID_POS] + d_random
                        pos_new = self.amend_position_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        if fit_new < pop_local[i][self.ID_FIT]:
                            pop_local[i] = [pos_new, fit_new]
                            if rand() < self.prob_stop:         # If probability stop fly occur, then stop it, or else move on
                                break
                        step += 1
                        pop[i] = [pos_new, fit_new]
                else:                   # Fly to personal best
                    step = 0
                    while step < self.n_steps:
                        d_random = uniform(zeros(self.problem_size), r_percept, self.problem_size)
                        pos_new = pop[i][self.ID_POS] + d_random
                        pos_new = self.amend_position_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        if fit_new < pop_local[i][self.ID_FIT]:
                            pop_local[i] = [pos_new, fit_new]
                            if rand() < self.prob_stop:
                                break
                        step += 1
                        pop[i] = [pos_new, fit_new]

            g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class IRRO(Root):
    """
        The original version of: Improved Raven Roosting Optimization (IRRO)
        Link:
            https://doi.org/10.1016/j.swevo.2017.11.006
        Questions:
            0. HOW? REALLY? How can this paper is accepted at the most strictly journal like this. I DON'T GET IT?
                This is not science, this is like people to say "pseudo-science or fake science".
            1. Like the above code, RRO is fake algorithm, why would someone try to improve it?
            2. And of course, because it is fake algorithm, so with a simple equation you can improve it.
            3. What is contribution of this paper to get accepted in this journal?
            4. Where is the Algorithm. 2 (OMG, the reviewers don't they see that missing?)
        Conclusion:
            How much money you have to pay to get accepted in this journal? Iran author?
            Please send me your code, if I'm wrong, I will publicly apology.
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 r_perception=3.6, r_leader=3.6, n_steps=20, weak_percent=0.4, food_max=1, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r_perception = r_perception    # Default: [1.8, 3.6], Factor that control the radius of perception
        self.r_leader = r_leader            # Default: [1.8, 3.6], Factor that control the radius of leader
        self.n_steps = n_steps              # Default: [5, 10, 20], Number of moving steps of each raven towards the best solution
        self.weak_percent = weak_percent    # Default: [0,2, 0.4, 0.6] Percentage of population will be moved towards by global best solution
        self.food_max = food_max

    def train(self):
        r_percept = ((self.ub - self.lb) / 2) / (self.r_perception * power(self.pop_size, 1.0 / self.problem_size))
        r_leader = ((self.ub - self.lb) / 2) / (self.r_leader * power(self.pop_size, 1.0 / self.problem_size))
        n_ravens = self.pop_size - int(self.weak_percent * self.pop_size)       # Number of greedy ravens
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        pop_local = deepcopy(pop)

        for epoch in range(self.epoch):

            # Calculate the food <-- The probability stopping of the fly
            food_st = self.food_max * (self.epoch - epoch) / (self.epoch)

            for i in range(self.pop_size):

                if i < n_ravens:  # Fly to global best
                    step = 0
                    while step < self.n_steps:
                        d_random = uniform(zeros(self.problem_size), r_leader, self.problem_size)
                        pos_new = pop[i][self.ID_POS] + d_random
                        pos_new = self.amend_position_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        if fit_new < pop_local[i][self.ID_FIT]:
                            pop_local[i] = [pos_new, fit_new]
                            if rand() < food_st:
                                break
                        step += 1
                        pop[i] = [pos_new, fit_new]
                else:  # Fly to personal best
                    step = 0
                    while step < self.n_steps:
                        d_random = uniform(zeros(self.problem_size), r_percept, self.problem_size)
                        pos_new = pop[i][self.ID_POS] + d_random
                        pos_new = self.amend_position_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        if fit_new < pop_local[i][self.ID_FIT]:
                            pop_local[i] = [pos_new, fit_new]
                            if rand() < food_st:
                                break
                        step += 1
                        pop[i] = [pos_new, fit_new]

            pop_idx = list(range(0, self.pop_size))
            pop_fit = [item[self.ID_FIT] for item in pop_local]
            zipped_pop = zip(pop_fit, pop_idx)
            zipped_pop = sorted(zipped_pop)
            pop1, pop_local1 = deepcopy(pop), deepcopy(pop_local)
            for i, (fit, idx) in enumerate(zipped_pop):
                pop[i] = pop1[idx]
                pop_local[i] = pop_local1[idx]
            if pop_local[0][self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(pop_local[0])

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class BaseRRO(Root):
    """
        My developed version: Raven Roosting Optimization (RRO)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 n_steps=10, weak_percent=0.4, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.n_steps = n_steps  # Default: [5, 10, 20], Number of moving steps of each raven towards the best solution
        self.weak_percent = weak_percent  # Default: [0,2, 0.4, 0.6] Percentage of population will be moved towards by global best solution

    def train(self):
        n_ravens = self.pop_size - int(self.weak_percent * self.pop_size)  # Number of greedy ravens
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        pop_local = deepcopy(pop)

        for epoch in range(self.epoch):

            for i in range(self.pop_size):

                if i < n_ravens:  # Fly to global best
                    step = 0
                    while step < self.n_steps:
                        pos_new = g_best[self.ID_POS] + uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                        pos_new = self.amend_position_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        if fit_new < pop_local[i][self.ID_FIT]:
                            pop_local[i] = [pos_new, fit_new]
                            break
                        step += 1
                        pop[i] = [pos_new, fit_new]
                else:  # Fly to personal best
                    step = 0
                    while step < self.n_steps:
                        pos_new = pop_local[i][self.ID_POS] + uniform() * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS])
                        pos_new = self.amend_position_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        if fit_new < pop_local[i][self.ID_FIT]:
                            pop_local[i] = [pos_new, fit_new]
                            break
                        step += 1
                        pop[i] = [pos_new, fit_new]

            pop_idx = list(range(0, self.pop_size))
            pop_fit = [item[self.ID_FIT] for item in pop_local]
            zipped_pop = zip(pop_fit, pop_idx)
            zipped_pop = sorted(zipped_pop)
            pop1, pop_local1 = deepcopy(pop), deepcopy(pop_local)
            for i, (fit, idx) in enumerate(zipped_pop):
                pop[i] = pop1[idx]
                pop_local[i] = pop_local1[idx]
            if pop_local[0][self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(pop_local[0])

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
