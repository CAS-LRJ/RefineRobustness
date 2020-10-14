# DeepSRGR: A Spurious Region Guided Refinement for DeepPoly
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
### Network File Format
Now DeepSRGR supports .rlv and .nnet file as network file. There is some existing network in _nnet_ and _rlv_ folder.

To load _nnet_ file using:
```
net.load_nnet(args)
```
To load _rlv_ file using:
```
net.load_rlv(args)
```
### Property File Format
An example of property file _local\_robustness\_1.txt_:
```
0.6719
0.0019
-0.4967
0.4694
-0.4532
-1 1 0 0 0 0
0 1 -1 0 0 0
0 1 0 -1 0 0
0 1 0 0 -1 0
```
The ACASXu network has 5 inputs which is introduced in first 5 lines. The following 4 lines describe a property in the form of _**A**x+**b**<=0_. In this case, the last 4 lines indicates that _OUT1_ is the minimal in _OUT0_ to _OUT4_.

There is some existing property files in _property_ folder. _local\_robustness\_x.txt_ is for ACASXu and _mnist\_x\_local\_property.in_ is for MNIST.