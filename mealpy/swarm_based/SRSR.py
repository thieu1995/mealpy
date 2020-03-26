#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import zeros, ones, abs, ceil, int, clip, floor, round, argmax, argwhere, reshape, sign, concatenate, power
from numpy.random import uniform, randint, permutation, normal
from copy import deepcopy
from mealpy.root import Root


class BaseSRSR(Root):
    """
    Swarm Robotics Search And Rescue: A Novel Artificial Intelligence-inspired Optimization Approach
    """
    ID_POS = 0
    ID_FIT = 1
    ID_MU = 2
    ID_SIGMA = 3
    ID_POS_NEW = 4
    ID_FIT_NEW = 5
    ID_FIT_MOVE = 6

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

    def _create_solution__(self, minmax=0):
        """  This algorithm has different encoding mechanism, so we need to override this method
        """
        x = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fit = self._fitness_model__(solution=x, minmax=minmax)
        mu = 0
        sigma = 0
        x_new = deepcopy(x)
        fit_new = deepcopy(fit)
        fit_move = fit - fit_new
        return [x, fit, mu, sigma, x_new, fit_new, fit_move]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Control Parameters Of Algorithm
        # ==============================================================================================
        #  [c1] movement_factor : Determines Movement Pace Of Robots During Exploration Policy
        #  [c2] sigma_factor    : Determines Level Of Divergence Of Slave Robots From Master Robot
        #  [c3] sigma_limit     : Limits Value Of Sigma Factor
        #  [c4] mu_factor       : Controls Mean Value For Master And Slave Robots
        #       Control Parameters C1, C2 And C3 Are Automatically Tuned While C4 Should Be Set By User
        # ==============================================================================================

        mu_factor = 2 / 3        # [0.1-0.9] Controls Dominance Of Master Robot, Preferably 2/3
        sigma_temp = zeros(self.pop_size)            # Initializing Temporary Stacks
        SIF = None
        movement_factor = ones(self.problem_size) * (self.domain_range[1] - self.domain_range[0])

        for epoch in range(0, self.epoch):
            # ========================================================================================= %%
            #            PHASE 1 (ACCUMULATION): CALCULATING Mu AND SIGMA values FOR SOLUTIONS            %
            # ===========================================================================================%%

            # ------ CALCULATING MU AND SIGMA FOR MASTER ROBOT ----------
            pop[0][self.ID_SIGMA] = uniform()
            if epoch % 2 == 1:
                pop[0][self.ID_MU] = (1 - pop[0][self.ID_SIGMA]) * pop[0][self.ID_POS]
            else:
                pop[0][self.ID_MU] = (1 + (1 - mu_factor) * pop[0][self.ID_SIGMA]) * pop[0][self.ID_POS]

            for i in range(1, self.pop_size):
                # ---------- CALCULATING MU AND SIGMA FOR SLAVE ROBOTS ---------
                pop[i][self.ID_MU] = mu_factor * pop[0][self.ID_POS] + (1-mu_factor) * pop[i][self.ID_POS]
                if epoch == 0:
                    SIF = 6
                sigma_temp[i] = SIF * uniform()
                pop[i][self.ID_SIGMA] = sigma_temp[i] * abs( pop[0][self.ID_POS] - pop[i][self.ID_POS] ) + \
                    uniform() ** 2 * ((pop[0][self.ID_POS] - pop[i][self.ID_POS]) < 0.05)

                # ----- Generating New Positions Using New Obtained Mu And Sigma Values --------------
                temp = normal(pop[i][self.ID_MU], pop[i][self.ID_SIGMA], self.problem_size)
                pop[i][self.ID_POS_NEW] = clip(temp, self.domain_range[0], self.domain_range[1])
                fit = self._fitness_model__(pop[i][self.ID_POS_NEW])
                pop[i][self.ID_FIT_NEW] = fit

                # --------- Calculate Degree Of Cost Movement Of Robots During Movement --------------
                pop[i][self.ID_FIT_MOVE] = pop[i][self.ID_FIT] - pop[i][self.ID_FIT_NEW]

                # ---------- Progress Assessment: Replacing More Quality Solutions With Previous Ones ------

                # Replace Solution If It Reached To A More Quality Position
                respec_id = int(floor(self.pop_size / (1 + round(uniform())))) - 1
                if pop[respec_id][self.ID_FIT] > fit:
                    pop[i][self.ID_POS] = pop[i][self.ID_POS_NEW]
                    pop[i][self.ID_FIT] = pop[i][self.ID_FIT_NEW]
                else:
                    # Replace Solution Whether It Reached A Better Position Or Not
                    if pop[i][self.ID_FIT] > pop[i][self.ID_FIT_NEW]:
                        pop[i][self.ID_POS] = pop[i][self.ID_POS_NEW]
                        pop[i][self.ID_FIT] = pop[i][self.ID_FIT_NEW]


            # --------- Determining Sigma Improvement Factor (Sif) Based On Vvss Movement -------------------
            ## Get best improved fitness
            fit_id = argmax([item[self.ID_FIT_MOVE] for item in pop])
            sigma_factor = 1 + uniform() * max(self.domain_range[1] - self.domain_range[0])
            SIF = sigma_factor * sigma_temp[fit_id]
            # Controlling Parameter Of Algorithm
            if SIF > max(self.domain_range[1]):
                SIF = max(self.domain_range[1]) * uniform()

            # ========================================================================================= %%
            #            Phase 2 (Exploration): Moving Slave Robots Toward Master Robot                   %
            # ===========================================================================================%%
            for i in range(0, self.pop_size):
                gb = uniform(-1, 1, self.problem_size)
                gb[gb >= 0] = 1
                gb[gb < 0] = -1
                temp = pop[i][self.ID_POS] * uniform() + gb * (pop[0][self.ID_POS] - pop[i][self.ID_POS]) +\
                       movement_factor * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                temp = clip(temp, self.domain_range[0], self.domain_range[1])
                fit = self._fitness_model__(temp)

                # ---------- Progress Assessment: Replacing More Quality Solutions With Previous Ones ------

                # Replace Solution If It Reached To A More Quality Position
                respec_id = int(floor(self.pop_size / (1 + round(uniform())))) - 1
                if pop[respec_id][self.ID_FIT] > fit:
                    pop[i][self.ID_POS] = pop[i][self.ID_POS_NEW]
                    pop[i][self.ID_FIT] = pop[i][self.ID_FIT_NEW]
                else:
                    # Replace Solution Whether It Reached A Better Position Or Not
                    if pop[i][self.ID_FIT] > pop[i][self.ID_FIT_NEW]:
                        pop[i][self.ID_POS] = pop[i][self.ID_POS_NEW]
                        pop[i][self.ID_FIT] = pop[i][self.ID_FIT_NEW]

            # ========================================================================================= %%
            #        PHASE 3 (LOCAL SEARCH): CREATING SOME WORKER ROBOTS ASSIGNED TO SEARCH               %
            #                      LOCATIONS AROUND POSITION OF MASTER ROBOT                              %
            # ===========================================================================================%%

            if epoch > 0:
                # --- EXTRACTING "INTEGER PART" AND "FRACTIONAL PART"  OF THE ELEMENTS OF MASTER RPBOT POSITION------
                master_robot = {"original": deepcopy(reshape(pop[0][self.ID_POS], (self.problem_size, 1))),
                                "sign": deepcopy(reshape(sign(pop[0][self.ID_POS]), (self.problem_size, 1))),
                                "abs": deepcopy(reshape(abs(pop[0][self.ID_POS]), (self.problem_size, 1))),
                                "int": deepcopy(reshape(floor(abs(pop[0][self.ID_POS])), (self.problem_size, 1))),                                 # INTEGER PART
                                "frac": deepcopy(reshape(abs(pop[0][self.ID_POS]) - floor(abs(pop[0][self.ID_POS])), (self.problem_size, 1)))}  # FRACTIONAL PART

                # ------- Applying Nth-root And Nth-exponent Operators To Create Position Of New Worker Robots -------
                worker_robot1 = (master_robot["int"] + power(master_robot["frac"], 1 / (1 + randint(1, 4)))) * master_robot["sign"]
                id_changed1 = argwhere(round(uniform(self.domain_range[0], self.domain_range[1], self.problem_size)))
                id_changed1 = reshape(id_changed1, (len(id_changed1)))
                worker_robot1 = reshape(worker_robot1, (self.problem_size, 1))
                worker_robot1[id_changed1] = master_robot["original"][id_changed1]

                worker_robot2 = (master_robot["int"] + power(master_robot["frac"], (1 + randint(1, 4)))) * master_robot["sign"]
                id_changed2 = argwhere(round(uniform(self.domain_range[0], self.domain_range[1], self.problem_size)))
                id_changed2 = reshape(id_changed2, (len(id_changed2)))
                worker_robot2 = reshape(worker_robot2, (self.problem_size, 1))
                worker_robot2[id_changed2] = master_robot["original"][id_changed2]

                # -------- Applying A Combined Ga-like Operator To Create Position Of New Worker Robot -------------
                random_per_mutation = permutation(self.problem_size)
                sec1 = random_per_mutation[0 : int(self.problem_size / 2)]
                sec2 = random_per_mutation[int(self.problem_size/2) : ]
                worker_robot3 = zeros((self.problem_size, 1))
                worker_robot3[sec1] = (master_robot["int"][sec1] + power(master_robot["frac"][sec1], 1 / (1+randint(1, 4)))) * master_robot["sign"][sec1]
                worker_robot3[sec2] = (master_robot["int"][sec2] + master_robot["frac"][sec2] ** (1 + randint(1, 4))) * master_robot["sign"][sec2]
                id_changed3 = argwhere( round(uniform(self.domain_range[0], self.domain_range[1], self.problem_size)))
                id_changed3 = reshape(id_changed3, (len(id_changed3)))
                worker_robot3[id_changed3] = master_robot["original"][id_changed3]

                #------- Applying Round Operators To Create Position Of New Worker Robot -------------------
                worker_robot4 = ceil(master_robot["abs"]) * master_robot["sign"]
                id_changed4 = argwhere(round(uniform(self.domain_range[0], self.domain_range[1], self.problem_size)))
                id_changed4 = reshape(id_changed4, (len(id_changed4)))
                worker_robot4[id_changed4] = master_robot["original"][id_changed4]

                worker_robot5 = floor(master_robot["abs"]) * master_robot["sign"]
                id_changed5 = argwhere(round(uniform(self.domain_range[0], self.domain_range[1], self.problem_size)))
                id_changed5 = reshape(id_changed5, (len(id_changed5)))
                worker_robot5[id_changed5] = master_robot["original"][id_changed5]

                # --------- Proogress Assessment: Replacing More Quality Solutions With Previous Ones ---------------
                workers = concatenate((worker_robot1.T, worker_robot2.T, worker_robot3.T, worker_robot4.T, worker_robot5.T), axis=0)
                for i in range(0, 5):
                    workers[i] = clip(workers[i], self.domain_range[0], self.domain_range[1])
                    fit = self._fitness_model__(workers[i])
                    if fit < pop[1][self.ID_FIT]:
                        pop[-(i+1)][self.ID_POS] = deepcopy(workers[i])
                        pop[-(i+1)][self.ID_FIT] = deepcopy(fit)

            # Sort pop and update the global best solution
            g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            movement_factor = abs(g_best[self.ID_POS]) - pop[-1][self.ID_POS]
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
