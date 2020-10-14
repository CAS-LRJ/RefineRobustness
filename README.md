# DeepSRGR: A Spurious Region Guided Refinement for Abstract Domains
## Requirements
We highly recommend conda([anaconda](https://www.anaconda.com/products/individual)/[miniconda](https://docs.conda.io/en/latest/miniconda.html)) to install all the requirements.
### Linux (CentOS 7.7 recommended)
The instructions to install all the requirements for Linux are:
```
conda create -n DeepSRGR python=3.7.8
conda activate DeepSRGR
conda install cvxopt cvxpy-base cvxpy glpk numpy scipy blas libblas libcblas coin-or-cbc
pip install cylp
```
### Windows (Windows 10 recommended)
The instructions to install all the requirements for Windows are:
```
conda create -n DeepSRGR python=3.7.8
conda activate DeepSRGR
conda install cvxopt cvxpy-base cvxpy glpk numpy scipy blas libblas libcblas
pip install cbcpy
pip install cylp
```
## Usage
### Example
An example of _main()_ in network.py
```
net=network()
net.load_nnet('nnet/ACASXU_experimental_v2a_4_2.nnet')
net.verify_lp_split(PROPERTY='properties/local_robustness_2.txt',DELTA=0.064,MAX_ITER=5,WORKERS=12,SPLIT_NUM=5,SOLVER=cp.GUROBI)
```
This code verifies 2nd local robustness property of ACASXu_4_2 network which radius is 0.064.
### Network Format
Now DeepSRGR supports .rlv and .nnet file as network file. There is some existing network in _nnet_ and _rlv_ folder.
To load _nnet_ file using:
```
net.load_nnet(args)
```
To load _rlv_ file using:
```
net.load_rlv(args)
```