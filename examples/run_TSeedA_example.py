#!/usr/bin/env python
# Tree-Seed Algorithm (TSeedA) Example
# Reference: Kiran, M. S. (2015). TSA: Tree-seed algorithm for continuous optimization.
# Expert Systems with Applications, 42(19), 6686-6698. DOI: 10.1016/j.eswa.2015.04.055

import numpy as np
from mealpy import FloatVar, TSeedA


def sphere_function(solution):
    """Sphere function - simple unimodal test function"""
    return np.sum(solution ** 2)


def rastrigin_function(solution):
    """Rastrigin function - multimodal test function"""
    n = len(solution)
    return 10 * n + np.sum(solution ** 2 - 10 * np.cos(2 * np.pi * solution))


def ackley_function(solution):
    """Ackley function - multimodal test function"""
    n = len(solution)
    sum1 = np.sum(solution ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * solution))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


if __name__ == "__main__":
    # Define the problem
    problem_dict = {
        "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
        "minmax": "min",
        "obj_func": sphere_function
    }

    # Test with different ST values
    st_values = [0.1, 0.3, 0.5]
    
    print("=" * 60)
    print("Tree-Seed Algorithm (TSA) Benchmark")
    print("=" * 60)
    
    for st in st_values:
        print(f"\n--- ST = {st} ---")
        model = TSeedA.OriginalTSeedA(epoch=500, pop_size=50, st=st)
        g_best = model.solve(problem_dict)
        print(f"Best Fitness: {g_best.target.fitness:.6e}")
        print(f"Solution (first 5 dims): {g_best.solution[:5]}")
    
    # Test on Rastrigin function
    print("\n" + "=" * 60)
    print("Testing on Rastrigin function (ST=0.1)...")
    problem_dict["obj_func"] = rastrigin_function
    
    model = TSeedA.OriginalTSeedA(epoch=500, pop_size=50, st=0.1)
    g_best = model.solve(problem_dict)
    print(f"Best Fitness: {g_best.target.fitness:.6e}")
    
    # Test on Ackley function
    print("\n" + "=" * 60)
    print("Testing on Ackley function (ST=0.1)...")
    problem_dict["obj_func"] = ackley_function
    problem_dict["bounds"] = FloatVar(lb=(-32.,) * 30, ub=(32.,) * 30, name="delta")
    
    model = TSeedA.OriginalTSeedA(epoch=500, pop_size=50, st=0.1)
    g_best = model.solve(problem_dict)
    print(f"Best Fitness: {g_best.target.fitness:.6e}")
    
    print("\n" + "=" * 60)
    print("Tree-Seed Algorithm Example completed successfully!")
    print("=" * 60)
