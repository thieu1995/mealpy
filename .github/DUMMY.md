## This file describes the dummy algorithm based on my understanding and my implementation

* All algorithms in this library were implemented by me (my code), including the original version (I read the paper and implemented it). 
* Some original papers are very unclear (parameters, equations, algorithm's flow) as I categorize them as dummy papers and algorithms 
* (I have carefully checked the paper and the related papers and searched for Matlab code or any programming code for it).



# Dummy Algorithms


**AAA - Artificial Algae Algorithm** .

  * **OriginalAAA**: Uymaz, S. A., Tezel, G., & Yel, E. (2015). Artificial algae algorithm (AAA) for nonlinear global optimization. Applied Soft Computing, 31, 153-171.
  * **BaseAAA**: My trial version

**BWO - Black Widow Optimization** .

  * **OriginalBWO**: Hayyolalam, V., & Kazem, A. A. P. (2020). Black Widow Optimization Algorithm: A novel meta-heuristic approach for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 87, 103249.
  * **BaseBWO**: My trial version

**BOA - Butterfly Optimization Algorithm**.

  * **OriginalBOA**: Arora, S., & Singh, S. (2019). Butterfly optimization algorithm: a novel approach for global optimization. Soft Computing, 23(3), 715-734.
  * **BaseBOA**: My trial version
  * **AdaptiveBOA**: Singh, B., & Anand, P. (2018). A novel adaptive butterfly optimization algorithm. International Journal of Computational Materials Science and Engineering, 7(04), 1850026.

**BMO - Blue Monkey Optimization** .
  * **OriginalBMO**: Blue Monkey Optimization: (2019) The Blue Monkey: A New Nature Inspired Metaheuristic Optimization Algorithm. DOI: http://dx.doi.org/10.21533/pen.v7i3.621
  * **BaseBMO**: My trial version

**EPO - Emperor Penguin Optimizer** .
  * **OriginalEPO**: Dhiman, G., & Kumar, V. (2018). Emperor penguin optimizer: A bio-inspired algorithm for engineering problems. Knowledge-Based Systems, 159, 20-50.
  * **BaseEPO**: My trial version

**PIO - Pigeon-Inspired Optimization** .
  * **None**: Duan, H., & Qiao, P. (2014). Pigeon-inspired optimization: a new swarm intelligence optimizer for air robot path planning. International journal of intelligent computing and cybernetics.
  * **BasePIO**: My trial version, since the Original version not working.
  * **LevyPIO**: My trial version using Levy-flight

**RHO - Rhino Herd Optimization** .
  * **OriginalRHO**: Wang, G. G., Gao, X. Z., Zenger, K., & Coelho, L. D. S. (2018, December). A novel metaheuristic algorithm inspired by rhino herd behavior. In Proceedings of The 9th EUROSIM Congress on Modelling and Simulation, EUROSIM 2016, The 57th SIMS Conference on Simulation and Modelling SIMS 2016 (No. 142, pp. 1026-1033). Linköping University Electronic Press.
  * **BaseRHO**: My developed version
  * **LevyRHO**: My developed using Levy-flight

**SOA - Sandpiper Optimization Algorithm** .
  * **OriginalSOA**: Kaur, A., Jain, S., & Goel, S. (2020). Sandpiper optimization algorithm: a novel approach for solving real-life engineering problems. Applied Intelligence, 50(2), 582-619.
  * **BaseSOA**: My trial version

**STOA - Sooty Tern Optimization Algorithm**.
  * **BaseSTOA**: Sooty Tern Optimization Algorithm: Dhiman, G., & Kaur, A. (2019). STOA: A bio-inspired based optimization algorithm for industrial engineering problems. Engineering Applications of Artificial Intelligence, 82, 148-174.

**RRO - Raven Roosting Optimizaiton**.
  * **OriginalRRO**: Brabazon, A., Cui, W., & O’Neill, M. (2016). The raven roosting optimisation algorithm. Soft Computing, 20(2), 525-545.
  * **IRRO**: Torabi, S., & Safi-Esfahani, F. (2018). Improved raven roosting optimization algorithm (IRRO). Swarm and Evolutionary Computation, 40, 144-154.
  * **BaseRRO**: My developed version



### 1. Raven Roosting Optimization (RRO)

#### 1a) OriginalRRO: The original version 
* Link: 
  * https://doi.org/10.1007/s00500-014-1520-5
  * Brabazon, A., Cui, W., & O’Neill, M. (2016). The raven roosting optimisation algorithm. Soft Computing, 20(2),
    525-545.
    
* Questions: 
```code 
1. How to set the value of R? I guess R = (UB - LB) / 2
2. How to handle the case raven fly outside of radius? I guess redo the fly from previous position.
3. How to select Perception Follower? I guess randomly selected
4. The Pseudo-code is wrong. After each iteration, create N random locations?
5. The sentence "There is a Prob_stop chance the raven will stop". What is Prob_stop. Not mention?
6. The whole paper contains only simple equation: x(t) = x(t-1) + d. Really? 
```

* **Conclusion**: The algorithm can't even converge for a simple problem (sphere function).

#### 1b) IRRO: The improved version of RRO 

* Link: 
  * https://doi.org/10.1016/j.swevo.2017.11.006
  * Torabi, S., & Safi-Esfahani, F. (2018). Improved raven roosting optimization algorithm (IRRO). Swarm and
    Evolutionary Computation, 40, 144-154.  
* Questions:

```code 
0. HOW? How can this paper be accepted in the most strict journal like this? I DON'T GET IT. This is not science. This is like people saying "pseudo-science or fake science."
1. Like the above code, RRO is weak algorithm (fake), why would someone try to improve it?
2. And of course, because it is weak algorithm, so with a simple equation you can improve it.
3. What is contribution of this paper to get accepted in this journal?
4. Where is the Algorithm. 2 (OMG, the reviewers don't they see that missing?)
```

* **Conclusion**: How can your paper get published in SEC journal?


#### 1c) BaseRRO: My developed version

* What I have done?

```code 
1. My dev version based on the Improved version but.
2. I removed the food source probability --> Remove 1 parameter (probability stopping)
This means when raven found better location, it will stop flying immediately.
3. The updating equation is changed like this:
  x_new = g_best + rand * (g_best - x_old)          # Fly to global best
  x_new = local_best + rand * (local_best - x_old)  # Fly to local best
4. So using those two above equations above, no need to use radius --> Remove 2 parameters (R perception and R leader)        
```

* **Conclusion**:

```code 
1. Reduce 3 parameters of algorithm.
2. Can truly optimize CEC benchmark functions.
3. Faster than both dummy version above.
```


## Teamwork Optimization Algorithm (TOA), Coati Optimization Algorithm (CoatiOA), Osprey Optimization Algorithm (OOA)

```code 
1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA),
Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Pelican Optimization Algorithm (POA), Northern goshawk optimization (NGO),
Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
2. Check the matlab code of all above algorithms
3. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
4. Check the results of benchmark functions in the papers, they are mostly make up results
```





## Fennec Fox Optimization (FFO), Serval Optimization Algorithm (ServalOA)

```code 
0. This is really disgusting, because the source code for this algorithm is almost exactly the same as the source code for Pelican Optimization Algorithm (POA)
1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA),
Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
Pelican Optimization Algorithm (POA), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO),
Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
2. Check the matlab code of all above algorithms
2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
4. Check the results of benchmark functions in the papers, they are mostly make up results
```

## Northern Goshawk Optimization (NGO), Pelican Optimization Algorithm (POA)
```code 
0. This is really disgusting, because the source code for this algorithm is exactly the same as the source code for Pelican Optimization Algorithm (POA).
1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA),
Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Pelican Optimization Algorithm (POA),
Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
2. Check the matlab code of all above algorithms
2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
4. Check the results of benchmark functions in the papers, they are mostly make up results
```


## Siberian Tiger Optimization (STO)

```code 
0. This is really disgusting, because the source code for this algorithm is exact the same as the source code for Osprey Optimization Algorithm (OOA)
1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA),
Northern Goshawk Optimization (NGO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Pelican Optimization Algorithm (POA),
Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
2. Check the matlab code of all above algorithms
2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
4. Check the results of benchmark functions in the papers, they are mostly make up results
```

## Tasmanian Devil Optimization (TDO)

```code 
0. This is really disgusting, because the source code for this algorithm is almost exactly the same as the source code of Osprey Optimization Algorithm
    1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Pelican optimization algorithm (POA),
    Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
    Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO),
    Osprey Optimization Algorithm (OOA), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
    2. Check the matlab code of all above algorithms
    2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
    4. Check the results of benchmark functions in the papers, they are mostly make up results

```



## Walrus Optimization Algorithm (WaOA)

```code 
0. This is really disgusting, because the source code for this algorithm is exactly the same as the source code of Northern Goshawk Optimization (NGO)
1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA),
Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Northern Goshawk Optimization (NGO),
Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Pelican Optimization Algorithm (POA),
Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
2. Check the matlab code of all above algorithms
2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
4. Check the results of benchmark functions in the papers, they are mostly make up results
```

## Zebra Optimization Algorithm (ZOA)

```code 
1. Algorithm design is very similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Pelican optimization algorithm (POA),
Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO),
Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
2. Check the matlab code of all above algorithms
2. Same authors, self-plagiarized article with kinda same algorithm with different meta-metaphors
4. Check the results of benchmark functions in the papers, they are mostly make up results

```

+ **Warning**: The list of all algorithms below we should avoid to use it
  + Add Zebra Optimization Algorithm (ZOA)
    + Ref: Zebra Optimization Algorithm: A New Bio-Inspired Optimization Algorithm for Solving Optimization Algorithm
  + Add Osprey Optimization Algorithm (OOA)
    + Ref: Osprey optimization algorithm: A new bio-inspired metaheuristic algorithm for solving engineering optimization problems
  + Add Coati Optimization Algorithm (CoatiOA)
    + Ref: Coati Optimization Algorithm: A New Bio-Inspired Metaheuristic Algorithm for Solving Optimization Problems
  + Add Pelican Optimization Algorithm (POA)
    + Ref: Pelican optimization algorithm: A novel nature-inspired algorithm for engineering applications
  + Add Northern Goshawk Optimization (NGO)
    + Ref: Northern Goshawk Optimization: A New Swarm-Based Algorithm for Solving Optimization Problems
  + Add Serval Optimization Algorithm (ServalOA) 
    + Ref: Serval Optimization Algorithm: A New Bio-Inspired Approach for Solving Optimization Problems
  + Add Siberian Tiger Optimization (STO)
    + Ref: Siberian Tiger Optimization: A New Bio-Inspired Metaheuristic Algorithm for Solving Engineering Optimization Problems
  + Add Walrus Optimization Algorithm (WaOA)
    + Ref: Walrus Optimization Algorithm: A New Bio-Inspired Metaheuristic Algorithm
  + Add Tasmanian Devil Optimization (TDO)
    + Ref: Tasmanian devil optimization: a new bio-inspired optimization algorithm for solving optimization algorithm
  + Add Fennec Fox Optimization (FFO)
    + Ref: Fennec Fox Optimization: A New Nature-Inspired Optimization Algorithm
  + Add Teamwork Optimization Algorithm (TOA)
    + Ref: Teamwork Optimization Algorithm: A New Optimization Approach for Function Minimization/Maximization


+ **Notes on plagiarism and fake algorithm:**
    + OOA and STO with the same exact code 
    + POA ServalOA, NGO, WaOA, and TDO with almost the same exact code 
    + ZOA and CoatiOA with almost the same exact code 
    + FFO is swap two phases of POA
    + TOA is kinda same as OOA and POA 
    + 


+ Sandpiper Optimization Algorithm (SOA) to swarm_based group:
    + OriginalSOA: the original version is made up one
        + This algorithm suffers from local optimal and lower convergence rate. 
        + It cannot update the position, so how to converge without update position?
        + I am curious about the algorithm's publication history, as I have found it submitted to multiple journals.
        + A detailed explain in this comment section 
        (https://www.researchgate.net/publication/334897831_Sandpiper_optimization_algorithm_a_novel_approach_for_solving_real-life_engineering_problems/comments)
    + BaseSOA: my modified version which changed some equations and flow.
      
+ Sooty Tern Optimization Algorithm (STOA) is another name of Sandpiper Optimization Algorithm (SOA) 
    + Amandeep Kaur, Sushma Jain, Shivani Goel,  Gaurav Dhiman. What they are doing are plagiarism, uneducated
     and unethical to meta-heuristics community.


+ Blue Monkey Optimization (BMO) to swarm_based group:
    + OriginalBMO: 
        + It is a made-up algorithm with a similar idea to "Chicken Swarm Optimization," which raises questions about its originality.
        + The pseudo-code is confusing, particularly the "Rate equation," which starts as a random number and then becomes a vector after the first loop. The movement of the blue monkey and children is also the same.
        + The unclear point here is the "Rate equation": really confuse because It's contain the position. As you know,
            The position is the vector, but finally, the author conclude that Rate is random number in range [0, 1]
            Luckily, using number we can plus/add number and vector or vector and vector.
            So at first, Rate is random number then after the 1st loop, its become vector. 
        + Morever, both equtions movement of blue monkey and children is the same.
        + In addition, they don't check the bound after update position.
        + Keep going, they don't tell you the how to find the global best (I mean from blue monkey group or child group)
    + BaseBMO: my modified version which used my knowledge about meta-heuristics to do it. 