
# Version 2.4.1

### Update

+ Add after_initialization(), termination_end() to Optimizer class 
+ Update create_solution(), initialization() in Optimizer class
+ Add "starting_positions" parameter to solve() function in Optimizer class
+ Fix missing amend_position function in GA
+ Fix bug fitness value in history object
+ Update 4 training modes in all algorithms
```code 
Type: Parallel (no effect on updating process of agents) has 2 training modes:

1. Process: Using multi-cores to update fitness for whole population
2. Thread: Using multi-threads to update fitness for whole population

Type: Sequential has 2 training modes

3. Swarm (no effect on updating process): Updating fitness after the whole population move
4. Single (effect on updating process): Updating fitness after each agent move
```
+ Add agent's history and starting positions to docs.


---------------------------------------------------------------------


# Version 2.4.0

### Update

+ Add mealpy's support functions in terminal: help(mealpy), dir(mealpy)
+ Add logger module (Logger class)
+ Add validator module (Validator class)
+ Change in Optimizer class:
  + remove function get_global_best_global_worst_solution()
  + replace save_optimization_process() by track_optimize_step() and track_optimize_process()
  + update input of Problem and Termination object in Optimizer.
  + add logger
  + add validator and update all algorithms
  + update function: get_special_solutions()
  + rename function: crossover_arthmetic_recombination() to crossover_arithmetic()
  + rename function: get_fitness_position() to get_target_wrapper()
  + rename function: update_fitness_population() to update_target_wrapper_population()

+ A default method: generate_position() in Problem class.
+ Due to nature's characteristics of different problems, 2 methods can be designed for Optimizer to fit the problem 
  are generate_position() and amend_position(). Both methods are moved from Optimizer 
  class to Problem class, the create_solution() in Optimizer class will call these methods to create a new solution.
+ Update History and Problem class
  + design default amend_position function in Problem class 
  + parameter: obj_weight changed to obj_weights
  + add parameter: save_population to control 
+ Add Pareto-like Sequential Sampling (PSS) to math_based group


---------------------------------------------------------------------

# Version 2.3.0

### Update

+ All algorithms have been updated with the amend_position function for solving the discrete problem.
+ Required packages are version reduction to fit python 3.6
+ Add examples of how to design and custom a new algorithm based on this framework
+ Add examples mealpy solve discrete problems (combinatorial, permutation)


---------------------------------------------------------------------

# Version 2.2.0

### Update models

* You can pass the Problem dictionary or Problem object to the model.
* You can pass the Termination dictionary or Termination object to the model.
* The objective function is renamed as fitness function (obj_func -> fit_func)
* The general format of a solution is: **\[position, target\]**
    * position: numpy vector (1-D array)
    * target: **\[fitness, list_objectives\]**
    * list_objectives: **\[objective 1, objective 2, ...\]**
    * After the training process, everything can be accessed via the objective "history" (model.history)

* You can name your model and name your fitness function when creating a model 
  * model(problem, epoch, pop_size, ...., name='your model name', fit_name='your fitness function name')
* Add new algorithms: 
  * Gradient-Based Optimizer (GBO) in math_based group
  * Chaos Game Optimization (CGO) in math_based group
* Remove all dummy algorithms (Not supported anymore)
* Fix bugs:
  * Find idx of min-distance in BRO algorithm
  * Update more strategy for GA algorithm
  * Update child selection process in MA algorithm

### Update others

+ examples: Update several scenarios for mealpy with other frameworks
+ document: Add document website (https://mealpy.readthedocs.io/)

---------------------------------------------------------------------

# Version 2.1.2

### Update

+ Some algorithms have been updated in the GitHub release but not on PyPI such as Sparrow Search Algorithm.
+ I try to synchronize the version on GitHub and PyPI by trying to delete the vers
+ Add examples for applications of mealpy such as:
  + Tuning hyper-parameter of neural network
  + Replacing Gradient Descent optimizer in neural network
  + Tuning hyper-parameter for other models such as SVM,...

---------------------------------------------------------------------

# Version 2.1.1

### Update

+ Replace all .copy() operator by deepcopy() operator in module copy. Because shallow copy causing the problem with 
  nested list inside list. Especially when copying population with nested of list position inside agent.
+ Add the Knapsack Problem example: examples/applications/discrete-problems/knapsack-problem.py
+ Add the Linear Regression example with Pytorch: examples/applications/pytorch/linear_regression.py
+ Add tutorial videos "How to use Mealpy library" to README.md

---------------------------------------------------------------------

# Version 2.1.0

### Change models

+ Move all parallel function to Optimizer class
+ Remove unused methods in Optimizer class
+ Update all algorithm models with the same code-style as previous version
+ Restructure some hard algorithms include BFO, CRO.

### Change others

+ examples: Update examples for all new algorithms
+ history: Update history of MHAs
+ parallel: Add comment on parallel and sequential mode
+ Add code-of-conduct
+ Add the complete example: examples/example_full_v210.py

---------------------------------------------------------------------

# Version 2.0.0

### Change models
+ Update entire the library based on Optimizer class:
    + Add class Problem and class Termination
    + Add 3 training modes (sequential, thread and process)
    + Add visualization charts:
        + Global fitness value after generations
        + Local fitness value after generations
        + Global Objectives chart (For multi-objective functions)
        + Local Objective chart (For multi-objective functions)
        + The Diversity of population chart
        + The Exploration verse Exploitation chart
        + The Running time chart for each iteration (epoch / generation)
        + The Trajectory of some agents after generations 
+ My batch-size idea is removed due to the parallel training mode
+ User can define the Stopping Condition based on:
    + Epoch (Generation / Iteration) - default
    + Function Evaluation 
    + Early Stopping
    + Time-bound (The running time for a single algorithm for a single task)


### Change others

+ examples: Update examples for all new algorithms
+ history: Update history of MHAs

---------------------------------------------------------------------

# Version 1.2.2

### Change models

+ Add Raven Roosting Optimization (RRO) and its variants to Dummy group
    + OriginalRRO: The original version of RRO
    + IRRO: The improved version of RRO
    + BaseRRO: My developed version (On this version work)
    
+ Add some newest algorithm to the library
    + Arithmetic Optimization Algorithm (AOA) to Math-based group
        + OriginalAOA: The original version of AOA
    + Aquila Optimizer (AO) to Swarm-based group
        + OriginalAO: The original version of AO 
    + Archimedes Optimization Algorithm (ArchOA) to Physics-based group
        + OriginalArchOA: The original version of ArchOA

### Change others

+ examples: Update examples for all new algorithms
+ history: Update history of MHAs

---------------------------------------------------------------------

# Version 1.2.1

### Change models

+ Add Coyote Optimization Algorithm (COA) to Swarm-based group
+ Update code LCBO and MLCO
+ Add variant version of:
    + WOA: Hybrid Improved WOA 
    + DE:
        + SADE: Self-Adaptive DE
        + JADE: Adaptive DE with Optional External Archive
        + SHADE: Success-History Based Parameter Adaptation DE 
        + LSHADE: Linear Population Size Reduction for SHADE
    + PSO: Comprehensive Learning PSO (CL-PSO)
    
### Change others

+ examples: Update examples for all new algorithms

---------------------------------------------------------------------

# Version 1.2.0

### Change models

+ Fix bug reduction dimension in FOA
+ Update Firefly Algorithm for better timing performance
  
+ Add Hunger Games Optimization (HGS) to swarm-based group
+ Add Cuckoo Search Algorithm (CSA) to swarm-based group

+ Replace Root.\_\_init\_\_() function by super().\_\_init()\_\_ function in all algorithms.

### Change others

+ history: Update new algorithms
+ examples: Update all the examples based on algorithm's input

---------------------------------------------------------------------


# Version 1.1.0

### Change models

+ Update the way to passing hyper-parameters to root.py file (Big change)

+ Update all the hyper-parameters to all algorithms available.
  
+ Fix all the division by 0 in some algorithms.

### Change others

+ examples: Update all the examples of all algorithms

---------------------------------------------------------------------


# Version 1.0.5

### Change models
+ System-based group added: 
    + Water Cycle Algorithm (WCA)

+ Human-based group added:
    + Imperialist Competitive Algorithm (ICA)
    + Culture Algorithm (CA)

+ Swarm-based group added:
    + Salp Swarm Optimization (SalpSO)
    + Dragonfly Optimization (DO)
    + Firefly Algorithm (FA)
    + Bees Algorithm (Standard and Probilistic version)
    + Ant Colony Optimization (ACO) for continuous domain
    
+ Math-based group:
    + Add Hill Climbing (HC) 

+ Physics-based group:
    + Add Simulated Annealling (SA) 
    
### Change others

+ models_history.csv: Update history of meta-heuristic algorithms
+ examples: Add examples for all of above added algorithms.

---------------------------------------------------------------------


# Version 1.0.4

### Change models

+ Changed category of Sparrow Search Algorithm (SpaSA) from Fake to Swarm-based group:
    + Added the: OriginalSpaSA
        + This version is taken from the original paper, very weak algorithm
    + BaseSpaSA: My changed version
        + Changed equations
        + Changed flows and operators
        + This version become the BEST algorithm 

+ Added Jaya Algorithm to Swarm-based group:
    + OriginalJA: The original version from original paper
    + BaseJA: My version of original JA for better running time.
        + Remove all third loop in algorithm
        + Change the second random variable r2 to Gaussian instead of uniform
    + LJA: The original version of: Levy-flight Jaya Algorithm (LJA)
        + Paper: An improved Jaya optimization algorithm with Levy flight
        + Link: https://doi.org/10.1016/j.eswa.2020.113902
        + Notes:
            + This version I still remove all third loop in algorithm
            + The beta value of Levy-flight equal to 1.8 as the best value in the paper.

+ DE, its state-of-the-art variants.
    + DESAP: including DESAP-Abs and DESAP-Rel
        + The main ideas is identified the population size without user-defined. Proposed equation: 
            + Initial ps_init = 10*n (n: is the problem size, number of dimensions)
            + DESAP-Abs: ps = round(ps_init + N (0, 1)), (N: is Gaussian value)
            + DESAP-Rel: ps = round(ps_init + U (-0.5, 0.5)), (U: is uniform random function)
            
+ Added Battle Royale Optimization Algorithm to Fake-algorithm
    + OriginalBRO:
        + The paper is very different than the author's matlab code. Even the algorithm's flow is wrong with index i, j.
        + I tested the results is very slow convergence, even with small #dimensions. I guess that is why he cloned the
        crossover process of Genetic Algorithm to his algorithm in the code (but not even mention it in the paper) to
         get the results in the paper. Don't know what to say about this. 
    + BaseBRO:
        + First, I removed all third loop in the algorithm for faster computation.
        + Second, Re-defined the algorithm's flow and algorithm's ideas
        
+ Added Fruit-fly Optimization Algorithm and its variants to Swarm-based group:
    + OriginalFOA:
        + This algorithm is the weakest algorithm in MHAs. It can't run with complicated objective function.
    + BaseFOA:
        + I changed the fitness function (smell function) by taking the distance each 2 adjacent dimensions
         --> Number of variables reduce from N to N-1
        + Update the position if only it find the better fitness value.
    + WFOA:
        + The original version of Whale Fruit-fly Optimization Algorithm (WFOA)
        + Paper: Boosted Hunting-based Fruit Fly Optimization and Advances in Real-world Problems
        + From my point of view, this algorithm is almost the same as Whale, only different in calculate fitness
         function. So it is not surprise that It outperforms BaseFOA
         
        https://www.sciencedirect.com/science/article/abs/pii/S0957417420307545
        https://sci-hub.se/10.1016/j.eswa.2020.113976
        https://sci-hub.se/10.1016/j.eij.2020.08.003
        https://sci-hub.se/10.1016/j.eswa.2020.113902
        https://www.x-mol.com/paper/1239433029684543488
    
+ Update root.py
    + Added improved_ms() function based on mutation and search mechanism - current better than levy-flight technique

  

### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: 
    + Add FBIO examples with large-scale benchmark functions
    
---------------------------------------------------------------------

# Version 1.0.3

### Change models
+ Update AEO and its variants
    + Replace LevyAEO by AdaptiveAEO by using levy-flight in both Consumption and Decomposition process.
    + Added Improved version by paper "Artificial ecosystem optimizer for parameters identification of proton exchange
     membrane fuel cells model"
    + Added Enhanced version by paper "An Enhanced Artificial Ecosystem-Based Optimization for Optimal Allocation of
      Multiple Distributed Generations"
    + Added Modified version by paper "Effective Parameter Extraction of Different Polymer Electrolyte Membrane Fuel
     Cell Stack Models Using a Modified Artificial Ecosystem Optimization Algorithm"

+ Update LCBO and its variants (ILCO > MLCO > LCBO)
    + Changed LevyLCBO to ModifiedLCO 
    + Added the best version ImprovedLCO -- current best version
    
+ Update EO and its variants (MEO > AEO > LevyEO > EO))
    + Added ModifiedEO by paper "An efficient equilibrium optimizer with mutation strategy for numerical optimization"
        + Currently the best version of EO
        + Based on mutation strategy and gaussian distribution search
    + Added AdaptiveEO by paper "A novel interdependence based multilevel thresholding technique using adaptive equilibrium optimizer"
        + The second best version of EO, after ModifiedEO
        + Based on Fitness average and memory saving of previous iteration

+ Update GWO and its variants (GWO > RW_GWO)
    + Added Random Walk Grey Wolf Optimization - RW_GWO
    + BaseGWO always perform better than RW_GWO
    
    
+ Update root.py
    + Added improved_ms() function based on mutation and search mechanism - current better than levy-flight technique
    
+ Add Forensic-Based Investigation Optimization (FBIO) to human_based group: 

  

### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: 
    + Add FBIO examples with large-scale benchmark functions
    
---------------------------------------------------------------------

# Version 1.0.2

### Change models
+ Update : CEM
+ Fix bug division by 0 in: IWO, SMA

+ Add Forensic-Based Investigation Optimization (FBIO) to human_based group: 
    + OriginalFBIO: the original version
    + BaseFBIO: my modified version:
        + Implement the fastest way (Remove all third loop)
        + Change equations
        + Change the flow of algorithm
  

### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: 
    + Add FBIO examples with large-scale benchmark functions
    
---------------------------------------------------------------------


# Version 1.0.1

### Change models
+ Added Slime Mould Algorithm (SMA) to bio_based group:
    + OriginalSMA: the original version of SMA
    + BaseSMA: my modified version:
        + Selected 2 unique and random solution to create new solution (not to create variable) --> remove third loop in original version
        + Check bound and update fitness after each individual move instead of after the whole population move in the original version
        + My version not only faster but also better
        
+ Added Spotted Hyena Optimizer (SHO) to swarm_based group:
    + BaseSHO: my modified version

       
+ Add category for fake algorithms (papers) and proofs: 
    + Butterfly Optimization Algorithm (BOA) to swarm_based group:
        + OriginalBOA: this is fake algorithm 
        + AdaptiveBOA:
        + BaseBOA:
            + Look at this guy (the author of this algorithm):
            + https://scholar.google.co.in/citations?hl=en&user=KvcHovcAAAAJ&view_op=list_works&sortby=pubdate
            + He invent BOA algorithm and public it in 2019, but so many variant version of BOA has been created since 2015.
            + How the hell that happened?
            + This is a plagiarism? I think this is one of the most biggest reason why mathematician researchers
             calling out meta-heuristics community is completely bullshit and unethical.
            + Just for producing more trash paper without any knowledge in it? This is why I listed BOA as the totally
             trash and fake
             
    + Sandpiper Optimization Algorithm (SOA) to swarm_based group:
        + OriginalSOA: the original version is fake
            + This algorithm is trash, unethical when summit a paper to 2 journals.
            + Can't even update its position.
            + A detailed explain in this comment section 
            (https://www.researchgate.net/publication/334897831_Sandpiper_optimization_algorithm_a_novel_approach_for_solving_real-life_engineering_problems/comments)
        + BaseSOA: my modified version which changed some equations and flow.
        
    + Sooty Tern Optimization Algorithm (STOA) is another name of Sandpiper Optimization Algorithm (SOA) 
        + Amandeep Kaur, Sushma Jain, Shivani Goel,  Gaurav Dhiman. What they are doing are plagiarism, uneducated
         and unethical to meta-heuristics community.
    
    + Blue Monkey Optimization (BMO) to swarm_based group:
        + OriginalBMO: the original version is fake 
            + The idea look like "Chicken Swarm Optimization"
            + The pseudo-code totally bullshit in my opinion, just read the paper you will understand.
            + The unclear point here is the "Rate equation": really confuse because It's contain the position. As you know,
                The position is the vector, but finally, the author conclude that Rate is random number in range [0, 1]
                Luckily, using number we can plus/add number and vector or vector and vector.
                So at first, Rate is random number then after the 1st loop, its become vector. 
            + Morever, both equtions movement of blue monkey and children is the same.
            + In addition, they don't check the bound after update position.
            + Keep going, they don't tell you the how to find the global best (I mean from blue monkey group or child group)
        + BaseBMO: my modified version which used my knowledge about meta-heuristics to do it. 


### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: 
    + Update and Add examples for all algorithms
    + All examples tested with CEC benchmark and large-scale problem (1000 - 2000 dimensions)
    
---------------------------------------------------------------------


# Version 1.0.0

### Change models
+ Change root model, then all of the algorithms are now change
    + domain_range -> lower bound and upper bound
    + log -> verbose
    + objective_func -> obj_func
    + batch-size training -> Inspired by the idea of batch-size training in gradient descent algorithm
+ Idea of batch-size training in meta-heuristics
    + Some algorithms update the global best solution when all of the individuals in the population have moved to a new position.
        + This idea is a similarity to the training the whole dataset in GD
    + But some algorithms update after each move to a new position.
        + This idea is a similarity as SGD
    + But the point here is if the algorithm doesn't take advantage of the global best solution when updating individual 
    the position then GD or SGD gives the same results.
    
    + So my idea of batch-size training here is very simple, after batch-size of individuals move, then we will update
    the global best solution. So:
        + batch-size = 1 ==> SGD
        + batch-size = population-size ==> GD 
        + batch-size should set = 10% / 25% / 50% of your population size
    
    + Some algorithms can't apply the idea of batch-size. For examples:
        + If the original algorithm has already divided the population into m-clan (m-group) --> No need batch-size here
        + If the original algorithm contains multiple-part. Each part contains several types of updating --> No need too.
        
+ For music_based:
    + BaseHS (HS): Is the one can't use batch-size idea, but not belong to any reason above.

+ For math_based:
    + BaseSCA (SCA): Updated with batch-size idea. Keep the original version for reference.
    
+ For system_based:
    + BaseAEO (AEO): Updated with the batch-size ideas and some of my new ideas. Still keep the original version 
    + BaseGCO (GCO): Updated with batch-size idea. Keep the original version
    
+ For bio_based:
    + BaseIWO (IWO):
    + BaseWHO (WHO):
    + BaseBBO (BBO):
        + Remove all third loop, make algorithm n-times faster than original 
        + In the migration step, instead of select solution based on the wheel in every variable in position, 
                  using the wheel and select a single position and update based on its all variable of that position.
    + BaseVCS (VCS):
        + Remove all third loop, make algorithm n-times faster than original
        + In Immune response process, updating the whole position instead of updating each variable in position 
        + Drop batch-size idea to 3 main processes of this algorithm, make it more robust
    + BaseSBO (SBO):
        + Remove all third loop, n-times faster than original
        + No need equation (1, 2) in the paper, calculate the probability by roulette-wheel. Also can handle negative values
        + Apply batch-size idea
    + BaseBWO (BWO): This is my changed version and worked. 
        + Using k-way tournament selection to select parent instead of randomizing
        + Repeat cross-over population_size / 2 instead of n_var/2
        + Mutation 50% of position instead of swap only 2 variable in a single position
        + OriginalBWO: is fake and just a variant of Genetic Algorithm
    + BaseAAA (AAA): This is my changed version but still not working
        + OriginalAAA: is fake taken from DE and CRO 
        + I realize in the original paper, parameters, and equations not clear.
        + In the Adaptation phase, what is the point of saving starving value when it doesn't affect the solution at all?
        + The size of the solution always = 2/3, so the friction surface will always stay at the same value.
        + The idea of the equation seems like taken from DE, the adaptation and reproduction process seem like taken from CRO.
        + Appearance from 2015, but still now 2020 none of Matlab code or python code about this algorithm.
    + EOA: 
        + BaseEOA: My modified version from original Matlab version
            + The original version from Matlab code above will not work well, even with small dimensions.
            + I changed updating process
            + Changed the Cauchy process using x_mean
            + Used global best solution
            + Remove the third loop for faster 
   
+ For human_based:
    + BaseTLO (TLO):
        + Remove all third loop
        + Apply batch-size idea
    + BSO:
        + BaseBSO: This is original version
        + ImprovedBSO: My improved version with levy-flight and removal of some parameters.
    + QSA: 4 variant version now runs faster than n-times Original version
        + BaseQSA: Remove all third loop, apply the idea of the global best solution
        + OppoQSA: Based on BaseQSA, apply the idea of opposition-based learning technique
        + LevyQSA: Based on BaseQSA, apply the idea of levy-flight in business 2
        + ImprovedQSA: Combination of OppoQSA and LevyQSA
        + OriginalQSA: The original version of QSA. Not working well
    + SARO:
        + BaseSARO: My version but not better than the original version, just faster than
        + OriginalSARO: Convergence rate better than base version but very slow in time comparison.
    + LCBO:
        + BaseLCBO: Is the original version
        + LevyLCBO: Use levy-flight and is the best among 3 version
        + ImprovedLCBO: 
    + SSDO:
        + BaseSSDO: This is the original version
        + LevySSDO: Apply the idea of levy-flight
    + GSKA:
        + OriginalGSKA: This is the original version, very slow for large-scale and slow convergence
        + BaseGSKA: Remove all third loop, change equations and ideas, faster than Original version
    + CHIO: This algorithm hasn't done yet. Don't use it yet
        + OriginalCHIO: Can fail at any time 
        + BaseCHIO: Can't convergence
        

+ For physics_based group:
    + WDO: 
        + BaseWDO: is the original version
    + MVO:
        + OriginalMVO: is weak and slow algorithm 
        + BaseMVO: can solve large-scale optimization problems
    + TWO:
        + BaseTWO: is the original version
        + OppoTWO: using opposition-based techniques (better than original version)
        + LevyTWO: using only levy-flight and better than OppoTWO
        + ImprovedTWO: using opposition-based and levy-flight and better than all others
    + EFO:
        + OriginalEFO: is the original version, run fast but slow convergence
        + BaseEFO: using levy-flight for large-scale dimension
    + NRO:
        + BaseNRO: is the original version, efficient even with large-scale due to levy-flight techniques
        but running-time will slow because third loop.
    + HGSO:
        + BaseHGSO: is the original version
        + OppoHGSO: uses opposition-based technique
        + LevyHGSO: uses levy-flight technique
    + ASO:
        + BaseASO: is the original version
    + EO:
        + BaseEO: is the original version
        + LevyEO: uses levy-flight technique for large-scale dimensions

+ For probabilistic_based group:
    + CEM:
        + BaseCEM: is the original version
        + CEBaseSBO: is the hybrid version of Satin Bowerbird Optimizer (SBO) and CEM
        + CEBaseSSDO: is the hybrid version of Social-Sky Driving Optimization (SSDO) and CEM
        + CEBaseLCBO and CEBaseLCBONew: are the hybrid version of Life Choice Based Optimization and CEM

+ For evolutionary_based group: (Not good for large-scale problems)
    + EP:
        + BaseEP: is the original version
        + LevyEP: applied levy-flight 
    + ES:
        + BaseES: is the original version
        + LevyES: applied levy-flight
    + MA:
        + BaseMA: is the original version, can't remove third loop, very slow algorithm
    + GA:
        + BaseGA: is the original version 
    + DE:
        + BaseDE: is the original version
    + FPA:
        + BaseFPA: is the original version (already use levy-flight in it)
    + CRO:
        + BaseCRO: is the original version
        + OCRO: is the opposition-based version

+ For swarm_based group: 
    + PSO:
        + BasePSO: is the original version
        + PPSO: Phasor particle swarm optimization: a simple and efficient variant of PSO
        + PSO_W: A modified particle swarm optimizer
        + HPSO_TVA: New self-organising  hierarchical PSO with jumping time-varying acceleration coefficients
    + ABC:
        + BaseABC: my version and taken from Clever Algorithms
    + FA:
        + BaseFA: is the original version, running slow even the all third loop already removed
    + BA:
        + OriginalBA: is the original version 
        + BasicBA: is also the original version with improved parameters
        + BaseBA: my modified version without A parameter
    + PIO: 
        + This is fake algorithm, after changing almost everything, the algorithm works
        + BasePIO: My base version
        + LevyPIO: My version based on levy-flight for large-scale dimensions
    + GWO:
        + BaseGWO: is the original version
    + ALO:
        + OriginalALO: is the original version, slow and less efficient
        + BaseALO: my modified version which using matrix multiplication for faster 
    + MFO:
        + OriginalMFO: is the original version
        + BaseMFO: my modified version which remove third loop, change equations and flow
    + EHO:
        + BaseEHO: is the original version
        + LevyEHO: my levy-flight version of EHO
    + WOA:
        + BaseWOA: is the original version
    + BSA:
        + BaseBSA: is the original version
    + SRSR:
        + BaseSRSR: is the original version
    + GOA:
        + BaseGOA: is the original version with some changed from me:
            + I added normal() component to Eq, 2.7
            + Changed the way to calculate distance between two location
            + Used batch-size idea    
    + MSA:
        + BaseMSA: is my modified version with some changed from original matlab code version
    + RHO:
        + OriginalRHO: is the original version, not working 
        + BaseRHO: my changed version 
        + LevyRHO: levy-flight for large-scale dimensions
            + Change the flow of algorithm
            + Uses normal in equation instead of uniform
            + Uses levy-flight instead of uniform-equation
    + EPO:
        + Original: is the original version, can't converge at all
        + BaseEPO: my modified version:
            + First: I changed the Eq. T_s and no need T and random R.
            + Second: Updated the old position if fitness value better or kept the old position if otherwise
            + Third: Remove the third loop for faster
            + Fourth: Batch size idea
            + Fifth: Add normal() component and change minus sign to a plus
    + NMRA:
        + BaseNMRA: The original version
            + The Matlab code of paper's author here: https://github.com/rohitsalgotra/Naked-Mole-Rat-Algorithm
            + Matlab code and paper are very different. 
        + LevyNMRA: My levy-flight version 
        + ImprovedNMRA:
            + Using mutation probability
            + Using levy-flight
            + Using crossover operator
    + BES:
        + BaseBES: the original version
    + PFA:
        + BasePFA: is the original version, I did redesign the equation based on distance.
            + The problem with using the distance is that when increasing the bound and dimensions 
            --> distance increase very fast --> new position will always over the bound
            --> we should divide the distance to a number of dimensions and the distance of the bound (upper-lower) to
            stabilize the distance 
            + The second problem is a new solution based on all other solutions --> we should also divide the new solution
            by the population size to stabilize it.
        + OPFA: is an enhanced version of PFA based on Opposition-based Learning (better than BasePFA)
        + ImprovedPFA: (sometime better than OPFA)
            + using opposition-based learning
            + using levy-flight 2 times
    + SFO:
        + BaseSFO: is the original version
        + ImprovedSFO: my improved version in which
            + Reform Energy equation,
            + No need parameter A and epxilon
            + Based on idea of Opposition-based Learning
    + SLO:
        + BaseSLO: is the changed version from my student
        + ImprovedSLO: is the improved version 
    + SpaSA:
        + BaseSpaSA: is my modified version, the original paper is fake, tons of unclear like parameters or equations
    + MRFO:
        + BaseMRFO: is the original version
        + LevyMRFO: is my modified version based on levy-flight
    + HHO:
        + BaseHHO: is the original version 
    + SSA:
        + OriginalSSA: is the original version
        + BaseSSA: my modified version 
    + CSO:
        + BaseCSO: is the original version
    + BFO:
        + BaseBFO: is the adaptive version of BFO
        + OriginalBFO: is the original version taken from Clever Algorithms
    + SSO:
        + BaseSSO: is the original version
    

### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: 
    + Update and Add examples for all algorithms
    + All examples tested with CEC benchmark and large-scale problem (1000 - 2000 dimensions)
    
---------------------------------------------------------------------



# Version 0.8.6

### Change models
+ Fix bug return position instead of fitness value in:
    + TLO 
    + SARO
+ Update some algorithms:
    + SLO
    + NRO
    + ABC
    
+ Added some variant version of PSO:
    + PPSO (Phasor particle swarm optimization: a simple and efficient variant of PSO)
    + PSO_W (A modified particle swarm optimizer)
    + HPSO_TVA (New self-organising  hierarchical PSO with jumping time-varying acceleration coefficients)
    
+ Added more algorithm in Swarm-based algorithm
    + SpaSA: Sparrow Search Algorithm (Same name SSA as Social Spider Algorithm --> I changed it to SpaSA)

### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: Added new examples of: 
    + PSO and variant of PSO
    + Update all examples which now using CEC functions
    
---------------------------------------------------------------------

# Version 0.8.5

### Change models
+ Fix bugs in several algorithm related to Division by 0, sqrt(0), 
+ Added more algorithm in Probabilistic-based algorithm
    + CEBaseSBO
+ Added selection by roulette wheel in root (This method now can handle negative fitness values)
+ Changed GA using roulette wheel selection instead of k-tournament method

### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: Added new examples of: 
    + CE_SSDO, CE_SBO
    + GA, SBO
    
---------------------------------------------------------------------

# Version 0.8.4

### Change models
+ Fix bugs in Probabilistic-based algorithm
    + BaseCEM
    + CEBaseLCBO
    + CEBaseLCBONew: No levy
    + CEBaseSSDO
+ Fix bugs in Physics-based algorithm
    + LevyEO
+ Fix bug in Human-based algorithm
    + LCBO

+ Added Coronavirus Herd Immunity Optimization (CHIO) in Human-based group
    + Original version: OriginalCHIO
        + This version stuck in local optimal and early stopping because the infected case quickly become immunity
        + In my version, when infected case all change to immunity. I make 1/3 population become infected then
         optimization step keep going.
    + My version: BaseCHIO
    
---------------------------------------------------------------------

# Version 0.8.3

### Change models
+ Probabilistic-based algorithm
    + Added Cross-Entropy Method (CEM)
    + Added CEM + LCBO
    + Added CEM + SSDO
    
---------------------------------------------------------------------

# Version 0.8.2

### Change models
+ Bio-based group 
    + Added Virus Colony Search (VCS)
        + BaseVCS: This is the very simple version of VCS. Not the original one in the paper 
        
+ Physics-based group
    + Remove EO not good version
    
+ Human-based group
    + Fix LCBO sort population in initialization process
    
+ Added new group: Probabilistic-based algorithm
    + Added Cross-Entropy Method (CEM)
    
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: Added new examples of: 
    + BaseVCS
    + BaseCEM

---------------------------------------------------------------------


# Version 0.8.1

### Change models
+ Evolutionary-based group 
    + Added Evolution Strategies (ES)
        + BaseES
        + LevyES: Idea ==> Top population being mutated based on strategy, Left population try to get out of their
         position based on levy-flight. 
    + Added Evolution Programming (EP)
        + BaseEP: Different than ES by operator and bout_size
        + LevyEP: Idea ==> Top population being selected based on tournament strategy round, 50% Left population
             try to make a comeback to take the good position with levy jump.
    + Added Memetic Algorithm (MA)
        + BaseMA
    
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: Added new examples of: 
    + BaseES and LevyES
    + BaseEP and LevyEP
    + BaseMA

---------------------------------------------------------------------


# Version 0.8.0

### Change models
+ Swarm-based group
    + Added Elephant Herding Optimization (EHO) in Swarm-based group
        + BaseEHO
        + LevyEHO: Changed the Uniform distribution the "Separating operator" by Levy-flight (50%) and Gaussian(50%) 
    + Added Pigeon-Inspired Optimization (PIO) in Swarm-based group
        + BasePIO (Changed almost everything include flow the algorithm)
        + LevyPIO 
            + Changed flow of algorithm
            + Removed some unnecessary loop
            + Removed some parameters
            + Added the levy-flight in second step make algorithm more robust
    + Added Fireworks Algorithm  (FA)
    
+ Human-based group
    + Added Gaining Sharing Knowledge-based Algorithm (GSKA)
    + Added Brain Storm Optimization Algorithm (BSO)
        + BaseBSO
        + ImprovedBSO (Remove some parameters + Changed Equations + Levy-flight + BaseBSO)

+ Evolutionary-based group 
    + Added Flower Pollination Algorithm (FPA)

+ Bio-based group
    + Added Artificial Algae Algorithm (Not working yet)
 
 
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: Added new examples of: 
    + BaseFA 
    + OriginalAAA
    + BaseBSO and ImprovedBSO 
    + BaseGSKA
    + BasePIO and LevyPIO
    + BaseEHO and LevyEHO
    + BaseFPA
    


---------------------------------------------------------------------

# Version 0.7.5

### Change models
+ Added Sea Lion Optimization in Swarm-based group
    + BaseSLO
    + ImprovedSLO (Shrinking Encircling + Levy + SLO)
 
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: Added new examples of: BaseSLO and ImprovedSLO 


---------------------------------------------------------------------


# Version 0.7.4

### Change models
+ Added Coral Reefs Optimization in Evolutionary-based group
    + BaseCRO
    + OCRO (Opposition-based CRO)
 
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: Added new examples of: BaseCRO and CRO   


---------------------------------------------------------------------


# Version 0.7.3

### Change models
+ Added Levy-flight and Opposition-based techniques in Root.py
+ Fixed codes include levy-flight and opposition-based of:
    + QSA
    + HGSO
    + TWO
    + NMRA
    + PFA
    + SFO
    + SSO
+ Added new modified version of models based on Levy-flight:
    + LCBO (LevyLCBO)
    + SSDO (LevySSDO)
    + EO (LevyEO)
    + AEO (LevyAEO)
    + MRFO (LevyMRFO)
    + NMRA (LevyNMRA)
 
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
+ examples: Added new examples tested base-version and levy-version of: LCBO, SSDO, EO, AEO, MRFO, NMRA   

---------------------------------------------------------------------

# Version 0.7.2

### Change 
+ Fix GA and WOA errors

---------------------------------------------------------------------

# Version 0.7.1

### Change 
+ Change input parameters of the root.py file 
+ Update the changed of input parameters of all algorithms
+ Update examples folders

---------------------------------------------------------------------

# Version 0.7.0

### Change models
+ Added new kind of meta-heuristics: math-based and music-based
+ Math_based:
    * SCA - Sine Cosine Algorithm
        + OriginalSCA: The original version
        + BaseSCA: My version changed the flow 
+ Music_based:
    * HS - Harmony Search
        + OriginalHS: The original version - not working
        + BaseHS: My version which changed a few things 
            * First I changed the random usaged of harmony memory by best harmoney memory
            * The fw_rate = 0.0001, fw_damp = 0.9995, number of new harmonies = population size (n_new = pop_size)
            
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
        
---------------------------------------------------------------------

# Version 0.6.0

### Change models
+ Added new kind of meta-heuristics: system-based 
+ System_based: Added the latest system-inspired meta-heuristic algorithms
    * GCO - Germinal Center Optimization
    * AEO - Artificial Ecosystem-based Optimization
     
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
        
---------------------------------------------------------------------

# Version 0.5.1

### Change models
+ Bio_based: Added the latest bio-inspired meta-heuristic algorithms
    * SBO - Satin Bowerbird Optimizer
    * WHO - Wildebeest Herd Optimization
        + OriginalWHO: The original version
        + BaseWHO: I changed the flow of algorithm
    * BWO - Black Widow Optimization
    
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
        
---------------------------------------------------------------------


# Version 0.5.0

### Change models
+ Added new kind of meta-heuristics: bio-based (biology-inspired) 
+ Bio_based: Added some classical bio-inspired meta-heuristic algorithms
    * IWO - Invasive Weed Optimization
    * BBO - Biogeography-Based Optimization
        
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
        
---------------------------------------------------------------------


# Version 0.4.1

### Change models
+ Human_based: Added the newest human-based meta-heuristic algorithms
    * SARO - Search And Rescue Optimization
    * LCBO: Life Choice-Based Optimization
    * SSDO - Social Ski-Driver Optimization
        + OriginalSSDO: The original version
        + BaseSSDO: The flow changed + SSDO
        
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms        
        
---------------------------------------------------------------------

# Version 0.4.0

### Change models
+ Human_based: Added some recent human-based meta-heuristic algorithms
    * TLO - Teaching Learning Optimization
        + OriginalTLO: The original version
        + BaseTLO: The elitist version
    * QSA - Queuing Search Algorithm
        + BaseQSA: The original version
        + OppoQSA: Opposition-based + QSA
        + LevyQSA: Levy + QSA
        + ImprovedQSA: Levy + Opposition-based + QSA
    
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms
    
---------------------------------------------------------------------

# Version 0.3.1

### Change models
+ Physics_based: Added the cutting-edge physics-based meta-heuristic algorithms
    * NRO - Nuclear Reaction Optimization  
    * HGSO - Henry Gas Solubility Optimization
        + BaseHGSO: The original version
        + OppoHGSO: Opposition-based + HGSO
        + LevyHGSO: Levy + HGSO
    * ASO - Atom Search Optimization
    * EO - Equilibrium Optimizer
    
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms
    
---------------------------------------------------------------------


# Version 0.3.0

### Change models
+ Physics_based: Added some recent physics-based meta-heuristic algorithms
    * WDO - Wind Driven Optimization 
    * MVO - Multi-Verse Optimizer 
    * TWO - Tug of War Optimization
        + BaseTWO: The original version
        + OppoTWO / OppoTWO: Opposition-based + TWO
        + LevyTWO: Levy + TWO
        + ITWO: Levy + Opposition-based + TWO
    * EFO - Electromagnetic Field Optimization
        + OriginalEFO: The original version
        + BaseEFO: My version (changed the flow of the algorithm)
    
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms
    
---------------------------------------------------------------------


# Version 0.2.2

### Change models
+ Swarm_based: Added the state-of-the-art swarm-based meta-heuristic algorithms
    * SRSR - Swarm Robotics Search And Rescue 
    * GOA - Grasshopper Optimisation Algorithm
    * EOA - Earthworm Optimisation Algorithm 
    * MSA - Moth Search Algorithm 
    * RHO - Rhino Herd Optimization 
        + BaseRHO: The original 
        + MyRHO: A little bit changed from BaseRHO version
        + Version3RH: A little bit changed from MyRHO version
    * EPO - Emperor Penguin Optimizer
        + OriginalEPO: Not working
        + BaseEPO: My version and works
    * NMRA - Nake Mole\-rat Algorithm
        + BaseNMRA: The original 
        + LevyNMR: Levy + BaseNMRA 
    * BES - Bald Eagle Search 
    * PFA - Pathfinder Algorithm
        + BasePFA: The original
        + OPFA: Opposition-based PFA
        + LPFA: Levy-based PFA
        + IPFA: Improved PFA (Levy + Opposition + PFA)
        + DePFA: DE + PFA
        + LevyDePFA: Levy + DE + PFA
    * SFO - Sailfish Optimizer
        + BaseSFO: The original
        + ImprovedSFO: Changed Equations + Opposition-based + SFO
    * HHO - Harris Hawks Optimization 
    * MRFO - Manta Ray Foraging Optimization
        + BaseMRFO: The original
        + MyMRFO: The version I changed the flow of the original one
    
    
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms
    
---------------------------------------------------------------------

# Version 0.2.1

### Change models
+ Swarm_based: Added more recently algorithm (since 2010 to 2016)
    * OriginalALO, BaseALO - Ant Lion Optimizer
    * OriginalBA, AdaptiveBA, BaseBA - Bat Algorithm
    * BSA - Bird Swarm Algorithm 
    * GWO - Grey Wolf Optimizer
    * MFO - Moth-flame optimization 
    * SSA - Social Spider Algorithm 
    * SSO - Social Spider Optimization
    
### Change others
+ models_history.csv: Update history of meta-heuristic algorithms
    
---------------------------------------------------------------------

# Version 0.2.0 

### Change models
+ root.py : Add 1 more helper functions
+ Swarm_based: Added
    * PSO - Particle Swarm Optimization
    * BFO, ABFOLS (Adaptive version of BFO) - Bacterial Foraging Optimization
    * CSO - Cat Swarm Optimization
    * ABC - Artificial Bee Colony
    * WOA - Whale Optimization Algorithm

### Change others
+ models_history.csv: Adding history of meta-heuristic algorithms
    
---------------------------------------------------------------------
# Version 0.1.1 

### Change models
+ root.py : Add more helper functions
+ Evolutionary_based
    * GA : Change the format of input parameters
    * DE : Change the format of input parameters
### Change others
+ Examples: Adding more complex examples
+ Library: "Opfunu" update the latest version 0.4.3
    
---------------------------------------------------------------------
# Version 0.1.0 (First version)

### Changed models
+ root.py (Very first file, the root of all algorithms)
+ Evolutionary_based
    * GA - Genetic Algorithm
    * DE - Differential Evolution

