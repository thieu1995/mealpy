#!/usr/bin/env python
# Created by "Thieu" at 13:24, 27/02/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from setuptools import setup, find_packages


def readme():
    with open('README.md', encoding='utf-8') as f:
        README = f.read()
    return README


setup(
    name="mealpy",
    version="3.0.1",
    author="Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="MEALPY: An Open-source Library for Latest Meta-heuristic Algorithms in Python",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords=["optimization", "metaheuristics", "MHA", "mathematical optimization", "nature-inspired algorithms",
              "evolutionary computation", "soft computing", "population-based algorithms",
              "Stochastic optimization", "Global optimization", "Convergence analysis", "Search space exploration",
              "Local search", "Computational intelligence", "Black-box optimization", "Robust optimization",
              "Hybrid algorithms", "Benchmark functions", "Metaheuristic design", "Performance analysis",
              "Exploration versus exploitation", "Self-adaptation", "Constrained optimization",
              "Intelligent optimization", "Adaptive search", "Simulations", "Algorithm selection"],
    url="https://github.com/thieu1995/mealpy",
    project_urls={
        'Documentation': 'https://mealpy.readthedocs.io/',
        'Source Code': 'https://github.com/thieu1995/mealpy',
        'Bug Tracker': 'https://github.com/thieu1995/mealpy/issues',
        'Change Log': 'https://github.com/thieu1995/mealpy/blob/master/ChangeLog.md',
        'Forum': 'https://t.me/+fRVCJGuGJg1mNDg1',
    },
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",    
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=["numpy>=1.17.5", "matplotlib>=3.3.0", "scipy>=1.7.1", "pandas>=1.2.0", "opfunu>=1.0.0"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.1"],
    },
    python_requires='>=3.7',
)
