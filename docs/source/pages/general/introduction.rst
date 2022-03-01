============
Introduction
============


------
MEALPY
------

* MEALPY is a largest python module for the most of cutting-edge nature-inspired meta-heuristic
  algorithms and is distributed under MIT license.

* Current version: 2.1.2, Total algorithms: 176 (original + variants), 89 original algorithms (8 dummy algorithms)

* Three different version of mealpy in term of passing hyper-parameters. So please careful check your version before
  using this library (Check `All releases`_).
   * mealpy < 1.0.5
   * 1.1.0 < mealpy < 1.2.2
   * mealpy >= 2.0.0

.. _All releases: https://pypi.org/project/mealpy/#history

* The goals of this framework are:
    * Sharing knowledge of meta-heuristic fields to everyone without a fee
    * Helping other researchers in all field access to optimization algorithms as quickly as possible
    * Implement the classical as well as the state-of-the-art meta-heuristics (The whole history of meta-heuristics)

* What you can do with this library:
    * Analyse parameters of algorithms.
    * Perform Qualitative Analysis of algorithms.
    * Perform Quantitative Analysis of algorithms.
    * Analyse rate of convergence of algorithms.
    * Test the scalability of algorithms.
    * Analyse the stability of algorithms.
    * Analyse the robustness of algorithms.

* And please giving me some credit if you are using this library. Lots of people just use it without reference, and if you want to cite my paper, check some
  of my `first-author paper`_.

.. _first-author paper: https://gist.github.com/thieu1995/2dcebc754bf0038d0c12b26ec9d591aa

::

	@software{thieu_nguyen_2020_3711949,
	  author       = {Nguyen Van Thieu},
	  title        = {A collection of the state-of-the-art MEta-heuristics ALgorithms in PYthon: Mealpy},
	  month        = march,
	  year         = 2020,
	  publisher    = {Zenodo},
	  doi          = {10.5281/zenodo.3711948},
	  url          = {https://doi.org/10.5281/zenodo.3711948}
	}


* If you guys are familiar with writing documentation and would like to join this project. Please send me an email to
  nguyenthieu2102@gmail.com. Your contribution to this project is greatly appreciated.

* If you guys want me to implement new algorithm, please open an `Issues ticket`_, and better send me an PDF of the original paper so I can read and
  implement it.

.. _Issues ticket: https://github.com/thieu1995/mealpy/issues

* If you are facing multiple/many objective optimization problems, you can use Mealpy with weighted-sum method to
  transform it into single-objective optimization problem. But you want to find Pareto front / Reference front, then
  I recommend to checkout the `PYMOO library`_. If I have time, I will also try to start a new
  library called `momapy`_ (A collection of the state-of-the-art Multiple/Many Objective Metaheuristic Algorithms in
  PYthon).

.. _PYMOO library: https://pymoo.org/
.. _momapy: https://github.com/thieu1995/momapy

------------
Optimization
------------
A very short introduction into meta-heuristic algorithms and how to use it to solve optimization problems. This document also introduces some basic concepts
and conventions.

Meta-heuristic algorithms are becoming increasingly popular in optimisation and applications over the last three decades. There are manyreasons for this
popularity and success, and one of the main reasons is that these algorithms have been developed bymimicking the most successful processes in nature,
including biological systems, and physicaland chemical processes. For most algorithms, we know their fundamental components, howexactly they interact to
achieve efficiency still remains partly a mystery, which inspires more active studies. Convergence analysis of a few algorithms such as the particle
swarm optimisation shows some insight, but in general mathematical analysis of metaheuristic algorithms remains unsolved and still an ongoing active research
topic

The solution of an optimisation problem requires the choiceand the correct use of theright algorithm. The choice of an algorithm largely depends on the type
of the optimisationproblem at hand. For large-scale nonlinear global optimisation problems, there is no agreed guideline for how to choose and what to choose.
In fact, we arenot even sure whether an efficient algorithm exists, which is especially true for NP-hard problems, and most real-world problem often are
NP-hard indeed and most applications, we can in general write an optimisation problem as the following generic form:

.. image:: /_static/images/general_format.png

Here the components xi of x are called design or decision variables, and they can be realcontinuous, discrete or the mixed of these two. The functions fi(x)
wherei= 1,2, ..., Mare called the objective functions or simply cost functions, and in the case of M= 1, thereis only a single objective. The space spanned by
the decision variables is called the design space or search space Rn, while the space formed by the objective function values is called the solution space or
response space. The equalities for hj and inequalities for gk are called constraints. It is worth pointing out that we can also write the inequalities in the
other way ≥ 0, and we can also formulate the objectives as a maximisation problem

The algorithms used for solving optimisation problems can be very diverse, from con-ventional algorithms to modern meta-heuristics

Most conventional or classic algorithms are deterministic.  For example, the simplexmethod in linear programming is deterministic. Some deterministic
optimisation algorithmsused the gradient information, they are called gradient-based algorithms. For example, thewell-known Newton-Raphson algorithm is
gradient-based, as it uses the function values andtheir derivatives, and it works extremely well for smooth unimodal problems. However, ifthere is some
discontinuity in the objective function, it does not work well. In this case,a non-gradient algorithm is preferred. Non-gradient-based, or
gradient-free/derivative-free,algorithms do not use any derivative, but only the function values. Hooke-Jeeves patternsearch and Nelder-Mead downhill simplex
are examples of gradient-free algorithms

For stochastic algorithms, in general we have two types: heuristic and meta-heuristic,though their difference is small. Loosely speaking,heuristic means ‘to
find’ or ‘to discover by trial and error’. Quality solutions to a tough optimisation problem can be found in areasonable amount of time, but there is no
guarantee that optimal solutions are reached. Ithopes that these algorithms work most of the time, but not necessarily all the time. This isgood when we do
not necessarily want the best solutions but rather good solutions whichare easily reachable.


--------------------------
Meta-heuristics algorithms
--------------------------

In meta-heuristic algorithms,meta-means ‘beyond’ or ‘higher level’, and they generally perform better than simple heuristics. All meta-heuristic algorithms use
certain trade off of local search and global exploration. Variety of solutions are often realized via randomisation. Despite the popularity of meta-heuristics,
there is no agreed definition of heuristics and meta-heuristics in the literature. Some researchers use‘heuristics’ and ‘meta-heuristics’interchangeably.
However, the recent trend tends to name all stochastic algorithms with randomisation and global exploration as meta-heuristic. In this review, we will also
follow this convention.

.. image:: /_static/images/bio_inspired.png

Randomisation provides a good way to move away from local search to the search onthe global scale. Therefore, almost all meta-heuristic algorithms intend to
be suitable for global optimisation. Meta-heuristics can be an efficient way to produce acceptable solutions by trial and errorto a complex problem in a
reasonably practical time. The complexity of the problem of interest makes it impossible to search every possible solution or combination, the aim isto find
good feasible solution in an acceptable timescale. There is no guarantee that the best solutions can be found, and we even do not know whether an algorithm
will work and why if it does work. The idea is to have an efficient but practical algorithm that will work most the time and is able to produce good quality
solutions. Among the found quality solutions, it is expected some of them are nearly optimal, though there is no guarantee forsuch optimality.The main
components of any metaheuristic algorithms are: intensification and diversification, or exploitation and exploration (Blum and Roli 2003). Diversification
means to generate diverse solutions so as to explore the search spaceon the global scale, while intensification means to focus on the search in a local
region by exploiting the information that a current good solution is found in this region. This is in combination with with these lection of the best solutions.

The selection of the best ensures that the solutions will converge to the optimality. Onthe other hand, the diversification via randomisation avoids the
solutions being trapped at local optima, while increases the diversity of the solutions. The good combination of these two major components will usually ensure
that the global optimality is achievable.

Metaheuristic algorithms can be classified in many ways. One way is to classify themas: population-based and trajectory-based. For example, genetic algorithms
are population-based as they use a set of strings, so is the particle swarm optimisation (PSO) which uses multiple agents or particles (Kennedy and Eberhart
1995). On the other hand, simulated annealing uses a single agent or solution which moves through the design space or search space in a piecewise style
(Kirkpatrick et al. 1983).

.. image:: /_static/images/history_metaheuristics.png


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4