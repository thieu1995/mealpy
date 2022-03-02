## This file describes the dummy algorithm based on my understanding and my implementation

* All algorithms in this library were implemented by me (my code). Including the original version (I read the paper and
  implement it). Some original papers are very unclear (parameters, equations, algorithm's flow) as I categories it to
  dummy papers and algorithms (I have already checked carefully the paper, the related papers and searched for Matlab
  code or any programming code for it).


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
4. The Pseudo-code is wrong, 100%. After each iteration, create N random locations. For real?
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
0. HOW? REALLY? How can this paper is accepted at the most strictly journal like this. I DON'T GET IT?
   This is not science, this is like people to say "pseudo-science or fake science".
1. Like the above code, RRO is fake algorithm, why would someone try to improve it?
2. And of course, because it is fake algorithm, so with a simple equation you can improve it.
3. What is contribution of this paper to get accepted in this journal?
4. Where is the Algorithm. 2 (OMG, the reviewers don't they see that missing?)
```

* **Conclusion**: 

```code 
1. How much money you have to pay to get accepted in this journal? Iran author?
2. Please send me your code, if I'm wrong, I will publicly apology.
```

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
