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
# Create a network instance
net=network()
# Load the network file
net.load_nnet('nnet/ACASXU_experimental_v2a_4_2.nnet')
# Verify the property
net.verify_lp_split(PROPERTY='properties/local_robustness_2.txt',DELTA=0.064,MAX_ITER=5,WORKERS=12,SPLIT_NUM=5,SOLVER=cp.GUROBI)
```
This code verifies 2nd local robustness property of ACASXu_4_2 network which radius is 0.064.
### Network File Format
Now DeepSRGR supports .rlv and .nnet files as network files. There are some existing networks in the _nnet_ and _rlv_ folders.

To load an _nnet_ file using:
```
net.load_nnet(args)
```
To load an _rlv_ file using:
```
net.load_rlv(args)
```
### Property File Format
An example of a property file _local\_robustness\_1.txt_:
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
The ACAS Xu network has 5 inputs which are introduced in the first 5 lines. The following 4 lines describe a property in the form of _**A**x+**b**<=0_. In this case, the last 4 lines indicates that _OUT1_ is the minimal among _OUT0_ to _OUT4_.

There are some existing property files in the _properties_ folder, where _local\_robustness\_x.txt_ is for ACAS Xu and _mnist\_x\_local\_property.in_ is for MNIST.

### Parameters
- ```PROPERTY``` This is the property file.
- ```DELTA```  The radius to verify.
- ```SPLIT_NUM``` The number of dimensions to split, default value is 0. E.g. ```SPLIT_NUM=5``` implies that the number of blocks in the split is 2<sup>5</sup>=32.
- ```WORKERS``` The max number of processes, default value is 12.
- ```TRIM``` Whether to trim the input interval, e.g. _[-0.2, 0.7]_ to _[0, 0.7]_, default value is _FALSE_.
- ```SOLVER``` The linear programming solver, default value is _cp.GUROBI_, please use _cp.CBC_ if _GUROBI_ is not installed.
- ```MODE``` Verification Mode, 0 means _QUANTITIVE_, 1 means _ROBUSTNESS_.In _QUANTITVE_ mode, program will give a overapproximation of unsafe region. While in _ROBUSTNESS_ mode, program only cares about robust or not.The default value is 0.
- ```USE_OPT_2``` Whether to use optimization 2, default value is False.

Extra parameters in _find\_max\_disturbance_ and _find\_max\_disturbance\_lp_:
- ```L``` The lower bound to find max robust disturbance
- ```R``` The upper bound to find max robust disturbance

## Experiment
We evaluate our method in three different types of experiment.

### Improvement in precision

The following example code calculates the max robustness radius of mnist_fnn_1 for three different local robustness.
```main()
    net_list=['rlv/caffeprototxt_AI2_MNIST_FNN_'+str(i)+'_testNetworkB.rlv' for i in range(1,2)]
    property_list=['properties/mnist_'+str(i)+'_local_property.in' for i in range(3)]
    rlist=[]
    for net_i in net_list:
        plist=[]
        for property_i in property_list:
            print("Net:",net_i,"Property:",property_i)
            net=network()
            net.load_rlv(net_i)
            delta_base=net.find_max_disturbance(PROPERTY=property_i,TRIM=True)            
            lp_ans=net.find_max_disturbance_lp(PROPERTY=property_i,L=int(delta_base*1000),R=int(delta_base*1000+63),TRIM=True,WORKERS=96,SOLVER=cp.CBC)
            print("DeepPoly Max Verified Distrubance:",delta_base)
            print("Our method Max Verified Disturbance:",lp_ans)
            plist.append([delta_base,lp_ans])
        rlist.append(plist)
    print(rlist)
```
**_Hint_ The ```find_max_distrubance_lp``` uses only delta_base+63 which may not sufficient. If the results are equal to delta_base+0.063, please consider using a larger ```R```**

### Robustness verification performance

The following example code verifies a batch of properties of mnist_fnn_4 in a given radius.
```main()
    net_list=['rlv/caffeprototxt_AI2_MNIST_FNN_'+str(i)+'_testNetworkB.rlv' for i in range(4,5)]
    property_list=['properties/mnist_'+str(i)+'_local_property.in' for i in range(50)]
    delta=0.037
    for net_i in net_list:
        pass_list=[]
        nopass_list=[]
        net=network()
        net.load_rlv(net_i)
        index=0
        count_deeppoly=0
        count_deepsrgr=0
        mintime=None
        maxtime=None
        avgtime=0
        print('Verifying Network:',net_i)
        for property_i in property_list:
            net.clear()
            net.load_robustness(property_i,delta,TRIM=True)
            net.deeppoly()
            flag=True
            for neuron_i in net.layers[-1].neurons:                
                if neuron_i.concrete_upper>0:
                    flag=False
            if flag==True:
                count_deeppoly+=1
                print(property_i,'DeepPoly Success!')
            else:
                print(property_i,'DeepPoly Failed!')

            start=time.time()
            if net.verify_lp_split(PROPERTY=property_i,DELTA=delta,MAX_ITER=5,SPLIT_NUM=0,WORKERS=96,TRIM=True,SOLVER=cp.CBC,MODE=1,USE_OPT_2=True):
                print(property_i,'DeepSRGR Success!')
                count_deepsrgr+=1
                pass_list.append(index)
            else:
                print(property_i,'DeepSRGR Failed!')
                nopass_list.append(index)
            end=time.time()
            runningtime=end-start
            print(property_i,'Running Time',runningtime)
            if (mintime==None) or (runningtime<mintime):
                mintime=runningtime
            if (maxtime==None) or (runningtime>maxtime):
                maxtime=runningtime
            avgtime+=runningtime
            index+=1
        print('DeepPoly Verified:',count_deeppoly,'DeepSRGR Verified:',count_deepsrgr)
        print('Min Time:',mintime,'Max Time',maxtime,'Avg Time',avgtime/50)
        print('Passlist:',pass_list)
        print('Nopasslist:',nopass_list)   
```
**_Hint_ Batch Verify Experiments in paper use optimization 2**

### Quantitative robustness verification on ACAS Xu networks

The quantitative robustness experiment can give an over-approximation of the unsafe region. We firstly calculate the max robustness radius of a property using DeepPoly. After adding a disturbance, we use our method and DeepPoly to give the over-approximation of the unsafe region.
```
    net_list=['nnet/ACASXU_experimental_v2a_4_2.nnet']
    property_list=['properties/local_robustness_2.txt']
    disturbance_list=[0.02,0.03,0.04]    
    rlist=[]
    for net_i in net_list:
        plist=[]
        for property_i in property_list:
            net=network()
            net.load_nnet(net_i)
            delta_base=net.find_max_disturbance(PROPERTY=property_i)
            dlist=[]
            for disturbance_i in disturbance_list:
                print("Net:",net_i,"Property:",property_i,"Delta:",delta_base+disturbance_i)
                start=time.time()
                net=network()
                net.load_nnet(net_i)
                dlist.append(net.verify_lp_split(PROPERTY=property_i,DELTA=delta_base+disturbance_i,MAX_ITER=5,WORKERS=96,SPLIT_NUM=5,SOLVER=cp.CBC))
                end=time.time()
                print("Finished Time:",end-start)
            plist.append(dlist)
        rlist.append(plist)
    print(rlist)
```

## Log File

We store our experiment log in _log_ folder. Exp1, exp2, exp3 means precision experiment, performance experiment, quantitive experiment respectively.

## License and Copyright

Licensed under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0)