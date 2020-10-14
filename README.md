# DeepSRGR: A Spurious Region Guided Refinement for Abstract Domains
## Requirements
We highly recommend conda([anaconda](https://www.anaconda.com/products/individual)/[miniconda](https://docs.conda.io/en/latest/miniconda.html)) to install all the requirements.
### Linux (CentOS 7.7 recommended)
The instructions to install all the requirements for Linux are:
'''
conda create -n DeepSRGR python=3.7.8
conda activate DeepSRGR
conda install cvxopt cvxpy-base cvxpy glpk numpy scipy blas libblas libcblas coin-or-cbc
pip install cylp
'''
### Windows (Windows 10 recommended)
The instructions to install all the requirements for Windows are:
'''
conda create -n DeepSRGR python=3.7.8
conda activate DeepSRGR
conda install cvxopt cvxpy-base cvxpy glpk numpy scipy blas libblas libcblas
pip install cbcpy
pip install cylp
'''