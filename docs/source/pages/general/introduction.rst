============
Introduction
============


* MEALPY (MEta-heuristic ALgorithms in PYthon) is the largest Python module for the most cutting-edge nature-inspired meta-heuristic algorithms and is
distributed under the GNU General Public License (GPL) V3 license.

* Current version: 2.5.1, Total algorithms: 172 (102 original, 45 official variants, 25 developed variants)

* Different versions of mealpy in term of passing hyper-parameters. So please careful check your version before using this library (Check `All releases`_)
	* mealpy < 1.0.5
	* 1.1.0 < mealpy < 1.2.2
	* 2.0.0 <= mealpy <= 2.1.2
	* mealpy == 2.2.0
	* mealpy == 2.3.0
	* 2.4.0 <= mealpy <= 2.4.2 (From this version, algorithms can solve discrete problem)
	* mealpy >= 2.5.0

.. _All releases: https://pypi.org/project/mealpy/#history

* The goals of this framework are:
    * To share knowledge of the meta-heuristic field with everyone at no cost.
    * To help researchers in all fields access optimization algorithms as quickly as possible.
    * To implement both classical and state-of-the-art meta-heuristics, covering the entire history of meta-heuristics.

* What you can do with this library:
    * Analyze the parameters of algorithms.
    * Perform qualitative and quantitative analyses of algorithms.
    * Analyze the rate of convergence of algorithms.
    * Test and analyze the scalability and robustness of algorithms.


* If you would like to request a new algorithm, please open an `Issues ticket`_, or build your own New Optimizer using mealpy's components.

.. _Issues ticket: https://github.com/thieu1995/mealpy/issues



* And please give us some credits if you use this library, check some of my `previous paper`_.

.. _previous paper: https://gist.github.com/thieu1995/2dcebc754bf0038d0c12b26ec9d591aa

::

	@software{nguyen_van_thieu_2022_6684223,
	  author       = {Nguyen Van Thieu and Seyedali Mirjalili},
	  title        = {{MEALPY: a Framework of The State-of-The-Art Meta-Heuristic Algorithms in Python}},
	  month        = jun,
	  year         = 2022,
	  publisher    = {Zenodo},
	  version      = {v2.5.0},
	  doi          = {10.5281/zenodo.6684223},
	  url          = {https://doi.org/10.5281/zenodo.6684223}
	}

------------
Optimization
------------
A very short introduction to meta-heuristic algorithms and how to use them to solve optimization problems. This document also introduces some basic concepts
and conventions.

Meta-heuristic algorithms are becoming increasingly popular in optimization and applications over the last three decades. There are many reasons for this
popularity and success, and one of the main reasons is that these algorithms have been developed by mimicking the most successful processes in nature,
including biological systems, and physical and chemical processes. For most algorithms, we know their fundamental components, how exactly they interact to
achieve efficiency remains partly a mystery, which inspires more active studies. Convergence analysis of a few algorithms such as particle swarm
optimization shows some insight, but in general mathematical analysis of metaheuristic algorithms remains unsolved and still an ongoing active research topic.


The solution to an optimization problem requires the choice and the correct use of the right algorithm. The choice of an algorithm largely depends on the
type of optimization problem at hand. For large-scale nonlinear global optimization problems, there is no agreed guideline for how to choose and what to
choose. We are not even sure whether an efficient algorithm exists, which is especially true for NP-hard problems, and the most real-world problems often
are NP-hard indeed and in most applications, we can in general write an optimization problem as the following generic form:


.. image:: /_static/images/general_format.png


"The components xi of x are called design or decision variables, and they can be real, continuous, discrete, or a mix of these two. The functions fi(x),
where i=1,2,...,M, are called objective functions or simply cost functions. In the case of M=1, there is only a single objective. The space spanned by the
decision variables is called the design space or search space Rn, while the space formed by the objective function values is called the solution space or
response space. The equalities for hj and inequalities for gk are called constraints. It is worth pointing out that we can also write the inequalities in
the other way (≥0), and we can also formulate the objectives as a maximization problem.

The algorithms used for solving optimization problems can be very diverse, ranging from conventional algorithms to modern meta-heuristics.
Most conventional or classic algorithms are deterministic. For example, the simplex method in linear programming is deterministic.
Some deterministic optimization algorithms use gradient information and are called gradient-based algorithms. For example, the well-known Newton-Raphson
algorithm is gradient-based, as it uses the function values and their derivatives, and it works extremely well for smooth unimodal problems. However, if
there is some discontinuity in the objective function, it does not work well. In this case, a non-gradient algorithm is preferred.
Non-gradient-based, or gradient-free/derivative-free, algorithms do not use any derivatives but only the function values.
Hooke-Jeeves pattern search and Nelder-Mead downhill simplex are examples of gradient-free algorithms.

In stochastic algorithms, we generally have two types: heuristic and meta-heuristic, although their difference is small. Loosely speaking, heuristic means
'to find' or 'to discover by trial and error'. Quality solutions to a tough optimization problem can be found in a reasonable amount of time, but there is
no guarantee that optimal solutions will be reached. It is hoped that these algorithms will work most of the time, but not necessarily all the time. This is
useful when we do not necessarily want the best solutions but rather easily reachable and reasonably good solutions."




--------------------------
Meta-heuristics algorithms
--------------------------

In meta-heuristic algorithms,meta-means ‘beyond’ or ‘higher level’, and they generally perform better than simple heuristics. All meta-heuristic algorithms use
certain trade-offs of local search and global exploration. A variety of solutions are often realized via randomization. Despite the popularity of meta-heuristics,
there is no agreed definition of heuristics and meta-heuristics in the literature. Some researchers use heuristics and ‘meta-heuristics’ interchangeably.
However, the recent trend tends to name all stochastic algorithms with randomization and global exploration as meta-heuristic. In this review, we will also follow this convention.


.. image:: /_static/images/bio_inspired.png


Randomization is a useful tool for escaping local optima and facilitating global search. Consequently, the majority of meta-heuristic algorithms aim to be
effective in global optimization. These algorithms provide a practical way to generate satisfactory solutions for complex problems within a reasonable time
frame. Due to the problem's complexity, it is impractical to search for every possible solution or combination, and the goal is to find a feasible solution
of high quality within an acceptable time limit. Although there is no guarantee that the best solutions can be found, a good meta-heuristic algorithm is
expected to produce high-quality solutions most of the time. The key components of any meta-heuristic algorithm are intensification and diversification,
which are also referred to as exploitation and exploration (Blum and Roli 2003). Diversification generates a variety of solutions to explore the search
space globally, while intensification focuses on local search in a region where a promising solution has been found, utilizing the information from that
solution to identify the best solutions.

The selection of the best solutions ensures that the algorithm will converge to optimality. On the other hand, diversification via randomization helps to
avoid the algorithm being trapped at local optima and increases the diversity of solutions explored. A good balance between these two major components of
meta-heuristics can typically lead to global optimality being achieved.


Metaheuristic algorithms can be classified in many ways. One way is to classify them as population-based and trajectory-based. For example, genetic algorithms
are population-based as they use a set of strings, so is the particle swarm optimization (PSO) which uses multiple agents or particles (Kennedy and Eberhart
1995). On the other hand, simulated annealing uses a single agent or solution which moves through the design space or search space in a piecewise style
(Kirkpatrick et al. 1983).


.. image:: /_static/images/history_metaheuristics.png


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4