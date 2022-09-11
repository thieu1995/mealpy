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


