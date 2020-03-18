from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
    name="mealpy",
    version="0.5.1",
    author="Thieu Nguyen",
    author_email="nguyenthieu2102@gmail.com",
    description="A collection of the state-of-the-art MEta-heuristics ALgorithms in PYthon (mealpy)",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/thieunguyen5991/mealpy",
    download_url="https://github.com/thieunguyen5991/mealpy/archive/v0.5.1.zip",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    install_requires=["numpy", "scikit-learn", "matplotlib", "opfunu"],
    python_requires='>=3.6',
)