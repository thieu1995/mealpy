#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:51, 13/10/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.utils.visualize import export_convergence_chart, export_explore_exploit_chart, \
    export_diversity_chart, export_objectives_chart, export_trajectory_chart


class History:
    """A History class is responsible for saving each iteration's output.

    Note that you can use dump() and parse() for whatever your needs. Our default
    is only for agents, best agent and best agent's index.

    """

    def __init__(self):
        # Stores only the best agent
        self.list_global_best = []          # List of global best solution found so far in all previous generations
        self.list_current_best = []         # List of current best solution in each previous generations
        self.list_epoch_time = []           # List of runtime for each generation
        self.list_global_best_fit = []      # List of global best fitness found so far in all previous generations
        self.list_current_best_fit = []     # List of current best fitness in each previous generations
        self.list_population = []           # List of population in each generations
        self.list_diversity = None          # List of diversity of swarm in all generations
        self.list_exploitation = None       # List of exploitation percentages for all generations
        self.list_exploration = None        # List of exploration percentages for all generations
        self.epoch = None

    def save_initial_best(self, best_agent):
        self.list_global_best = [best_agent]
        self.list_current_best = self.list_global_best.copy()

    def get_global_repeated_times(self, id_fitness, id_target, epsilon):
        count = 0
        for i in range(0, len(self.list_global_best) - 1):
            temp = np.abs(self.list_global_best[i][id_fitness][id_target] - self.list_global_best[i + 1][id_fitness][id_target])
            if temp <= epsilon:
                count += 1
            else:
                count = 0
        return count

    def save_global_best_fitness_chart(self, title='Global Best Fitness', color='b', x_label="#Iteration", y_label="Function Value",
                                       filename="global-best-fitness-chart", verbose=True):
        # Draw global best fitness found so far in previous generations
        export_convergence_chart(self.list_global_best_fit, title=title, color=color, x_label=x_label,
                                 y_label=y_label, filename=filename, verbose=verbose)

    def save_local_best_fitness_chart(self, title='Local Best Fitness', color='b', x_label="#Iteration", y_label="Function Value",
                                      filename="local-best-fitness-chart", verbose=True):
        # Draw current best fitness in each previous generation
        export_convergence_chart(self.list_current_best_fit, title=title, color=color, x_label=x_label,
                                 y_label=y_label, filename=filename, verbose=verbose)

    def save_runtime_chart(self, title='Runtime chart', color='b', x_label="#Iteration", y_label='Second',
                           filename="runtime-chart", verbose=True):
        # Draw runtime for each generation
        export_convergence_chart(self.list_epoch_time, title=title, color=color, x_label=x_label,
                                 y_label=y_label, filename=filename, verbose=verbose)

    ## The paper: On the exploration and exploitation in popular swarm-based metaheuristic algorithms
    def save_exploration_exploitation_chart(self, title="Exploration vs Exploitation Percentages", list_colors=('blue', 'orange'),
                                            filename="exploration-exploitation-chart", verbose=True):
        # This exploration/exploitation chart should draws for single algorithm and single fitness function
        # Draw exploration and exploitation chart
        export_explore_exploit_chart(data=[self.list_exploration, self.list_exploitation], title=title,
                                     list_colors=list_colors, filename=filename, verbose=verbose)

    def save_diversity_chart(self, title='Diversity Measurement Chart', algorithm_name='GA',
                             filename="diversity-chart", verbose=True):
        # This diversity chart should draws for multiple algorithms for a single fitness function at the same time
        # to compare the diversity spreading
        export_diversity_chart(data=[self.list_diversity], title=title, list_legends=[algorithm_name],
                               filename=filename, verbose=verbose)


    ## Because convergence chart is formulated from objective values and weights,
    ## thus we also want to draw objective charts to understand the convergence
    ## Need a little bit more pre-processing

    def save_global_objectives_chart(self, title='Global Objectives Chart', x_label="#Iteration", y_label="Function Value",
                                     filename="global-objectives-chart", verbose=True):
        # 2D array / matrix 2D
        global_obj_list = np.array([agent[1][-1] for agent in self.list_global_best])
        # Make each obj_list as a element in array for drawing
        global_obj_list = [global_obj_list[:, idx] for idx in range(0, len(global_obj_list[0]))]
        export_objectives_chart(global_obj_list, title=title, x_label=x_label, y_label=y_label, filename=filename, verbose=verbose)

    def save_local_objectives_chart(self, title='Local Objectives Chart', x_label="#Iteration", y_label="Objective Function Value",
                                    filename="local-objectives-chart", verbose=True):
        current_obj_list = np.array([agent[1][-1] for agent in self.list_current_best])
        # Make each obj_list as a element in array for drawing
        current_obj_list = [current_obj_list[:, idx] for idx in range(0, len(current_obj_list[0]))]
        export_objectives_chart(current_obj_list, title=title, x_label=x_label, y_label=y_label,
                                filename=filename, verbose=verbose)

    def save_trajectory_chart(self, title="Trajectory of some first agents after generations",
                              list_agent_idx=(1, 2, 3), list_dimensions=(1, 2),
                              filename="trajectory-chart", verbose=True):
        ## Drawing trajectory of some agents in the first and second dimensions
        # Need a little bit more pre-processing
        list_agent_idx = set(list_agent_idx)
        list_dimensions = set(list_dimensions)
        list_agent_idx = sorted(list_agent_idx)
        list_dimensions = sorted(list_dimensions)
        n_dim = len(list_dimensions)

        if n_dim not in [1, 2]:
            print("Can draw trajectory only for 1 or 2 dimensions!")
            exit(0)
        if len(list_agent_idx) < 1 or len(list_agent_idx) > 10:
            print("Can draw trajectory for 1 to 10 agents only!")
            exit(0)
        if list_agent_idx[-1] > len(self.list_population[0]) or list_agent_idx[0] < 1:
            print(f"The index of input agent should be in range of [1, {len(self.list_population[0])}]")
            exit(0)
        if list_dimensions[-1] > len(self.list_population[0][0][0]) or list_dimensions[0] < 1:
            print(f"The index of dimension should be in range of [1, {len(self.list_population[0][0][0])}]")
            exit(0)

        pos_list = []
        list_legends = []

        # pop[0]: Get the first solution
        # pop[0][0]: Get the position of the first solution
        # pop[0][0][0]: Get the first dimension of the position of the first solution
        if n_dim == 1:
            y_label = f"x{list_dimensions[0]}"
            for idx, id_agent in enumerate(list_agent_idx):
                x = [pop[id_agent-1][0][list_dimensions[0]-1] for pop in self.list_population]
                pos_list.append(x)
                list_legends.append(f"Agent {id_agent}.")
            export_trajectory_chart(pos_list, n_dimensions=n_dim, title=title, list_legends=list_legends,
                                    y_label=y_label, filename=filename, verbose=verbose)
        elif n_dim == 2:
            x_label = f"x{list_dimensions[0]}"
            y_label = f"x{list_dimensions[1]}"
            for idx1, id_agent in enumerate(list_agent_idx):
                pos_temp = []
                for idx2, id_dim in enumerate(list_dimensions):
                    x = [pop[id_agent - 1][0][id_dim - 1] for pop in self.list_population]
                    pos_temp.append(x)
                pos_list.append(pos_temp)
                list_legends.append(f"Agent {id_agent}.")
            export_trajectory_chart(pos_list, n_dimensions=n_dim, title=title, list_legends=list_legends, x_label=x_label,
                                    y_label=y_label, filename=filename, verbose=verbose)
