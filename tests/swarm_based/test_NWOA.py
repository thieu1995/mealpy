"""
Test Script for NWOA Mealpy Integration
Tests the NWOA algorithm using mealpy's API
"""

import numpy as np
import sys
import os

# Add the parent directory to path to import NWOA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from NWOA import OriginalNWOA
from mealpy import FloatVar


def sphere_function(solution):
    """Simple sphere function for testing"""
    return np.sum(solution**2)


def rastrigin_function(solution):
    """Rastrigin function - multimodal test function"""
    A = 10
    n = len(solution)
    return A * n + np.sum(solution**2 - A * np.cos(2 * np.pi * solution))


def test_nwoa_basic():
    """Test NWOA with basic sphere function"""
    print("="*80)
    print("Test 1: NWOA on Sphere Function (30D)")
    print("="*80)
    
    # Define problem
    problem_dict = {
        "obj_func": sphere_function,
        "bounds": FloatVar(lb=(-10.,)*30, ub=(10.,)*30, name="delta"),
        "minmax": "min",
        "log_to": "console",
    }
    
    # Create and run optimizer
    model = OriginalNWOA(epoch=100, pop_size=50)
    g_best = model.solve(problem_dict)
    
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"Best fitness: {g_best.target.fitness:.6e}")
    print(f"Best solution (first 5 dims): {g_best.solution[:5]}")
    print("="*80)
    print()
    
    return g_best


def test_nwoa_rastrigin():
    """Test NWOA with Rastrigin function"""
    print("="*80)
    print("Test 2: NWOA on Rastrigin Function (30D)")
    print("="*80)
    
    # Define problem
    problem_dict = {
        "obj_func": rastrigin_function,
        "bounds": FloatVar(lb=(-5.12,)*30, ub=(5.12,)*30, name="delta"),
        "minmax": "min",
        "log_to": "console",
    }
    
    # Create and run optimizer
    model = OriginalNWOA(epoch=200, pop_size=50)
    g_best = model.solve(problem_dict)
    
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"Best fitness: {g_best.target.fitness:.6e}")
    print(f"Best solution (first 5 dims): {g_best.solution[:5]}")
    print("="*80)
    print()
    
    return g_best


def test_nwoa_parameters():
    """Test NWOA with custom parameters"""
    print("="*80)
    print("Test 3: NWOA with Custom Parameters")
    print("="*80)
    
    # Define problem
    problem_dict = {
        "obj_func": sphere_function,
        "bounds": FloatVar(lb=(-100.,)*10, ub=(100.,)*10, name="delta"),
        "minmax": "min",
    }
    
    # Create optimizer with custom parameters
    model = OriginalNWOA(
        epoch=50,
        pop_size=30,
        A=1.0,
        k=2*np.pi,
        omega=2*np.pi,
        delta=0.01,
        lambda_decay=0.001
    )
    
    print("Parameters:")
    print(f"  Epoch: {model.epoch}")
    print(f"  Population size: {model.pop_size}")
    print(f"  Wave amplitude (A): {model.A}")
    print(f"  Wave number (k): {model.k:.4f}")
    print(f"  Angular frequency (omega): {model.omega:.4f}")
    print(f"  Decay constant (delta): {model.delta}")
    print(f"  Energy decay rate (lambda): {model.lambda_decay}")
    print()
    
    g_best = model.solve(problem_dict)
    
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"Best fitness: {g_best.target.fitness:.6e}")
    print("="*80)
    print()
    
    return g_best


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*25 + "NWOA MEALPY INTEGRATION TEST")
    print("="*80)
    print()
    
    try:
        # Run tests
        result1 = test_nwoa_basic()
        result2 = test_nwoa_rastrigin()
        result3 = test_nwoa_parameters()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print()
        print("Summary:")
        print(f"  Test 1 (Sphere): {result1.target.fitness:.6e}")
        print(f"  Test 2 (Rastrigin): {result2.target.fitness:.6e}")
        print(f"  Test 3 (Custom params): {result3.target.fitness:.6e}")
        print("="*80)
        print()
        print("✅ NWOA is ready for mealpy integration!")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR OCCURRED!")
        print("="*80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*80)
