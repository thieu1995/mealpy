## This file describes the dummy algorithm based on my implementation

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
