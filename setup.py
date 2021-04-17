from setuptools import setup, find_packages

def readme():
    with open('README.md', encoding='utf-8') as f:
        README = f.read()
    return README

setup(
    name="mealpy",
    version="1.1.1-alpha",
    author="Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="A collection of the state-of-the-art MEta-heuristics ALgorithms in PYthon (mealpy)",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/thieu1995/mealpy",
    download_url="https://github.com/thieu1995/mealpy/archive/v1.1.1-alpha.zip",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
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
    install_requires=["numpy", "opfunu", "matplotlib", "scipy"],
    python_requires='>=3.6',
)