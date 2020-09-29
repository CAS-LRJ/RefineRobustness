import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import cvxpy as cp
import time
from copy import deepcopy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed,wait

def lpsolve(vars,cons,obj,solver=cp.GUROBI):
    prob=cp.Problem(obj,cons)
    prob.solve(solver=solver)
    return prob.value

class neuron(object):
    """
    Attributes:
        algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of previous neurons and a constant) 
        algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of previous neurons and a constant)
        concrete_algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of input neurons and a constant) 
        concrete_algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of input neurons and a constant)
        concrete_lower (float): neuron's concrete lower bound
        concrete_upper (float): neuron's concrete upper bound
        concrete_highest_lower (float): neuron's highest concrete lower bound
        concrete_lowest_upper (float): neuron's lowest concrete upper bound
        weight (numpy ndarray of float): neuron's weight        
        bias (float): neuron's bias
        certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0)
        prev_abs_mode (int): indicates abstract mode of relu nodes in previous iteration.0 use first,1 use second
    """

    def __init__(self):
        self.algebra_lower=None
        self.algebra_upper=None
        self.concrete_algebra_lower=None
        self.concrete_algebra_upper=None
        self.concrete_lower=None
        self.concrete_upper=None
        self.concrete_highest_lower=None
        self.concrete_lowest_upper=None
        self.weight=None
        self.bias=None
        self.prev_abs_mode=None
        self.certain_flag=0
    
    def clear(self):
        self.certain_flag=0
        self.concrete_highest_lower=None
        self.concrete_lowest_upper=None
        self.prev_abs_mode=None

    def print(self):
        print('algebra_lower:',self.algebra_lower)
        print('algebra_upper:',self.algebra_upper)
        print('concrete_algebra_lower:',self.concrete_algebra_lower)
        print('concrete_algebra_upper:',self.concrete_algebra_upper)
        print('concrete_lower:',self.concrete_lower)
        print('concrete_upper:',self.concrete_upper)
        print('weight:',self.weight)
        print('bias:',self.bias)
        print('certain_flag:',self.certain_flag)

class layer(object):
    """
    Attributes:
        neurons (list of neuron): Layer neurons
        size (int): Layer size
        layer_type (int) : Layer type 0 input 1 affine 2 relu
    """
    INPUT_LAYER=0
    AFFINE_LAYER=1
    RELU_LAYER=2    

    def __init__(self):
        self.size=None
        self.neurons=None
        self.layer_type=None
    
    def clear(self):
        for i in range(len(self.neurons)):
            self.neurons[i].clear()

    def print(self):
        print('Layer size:',self.size)
        print('Layer type:',self.layer_type)
        print('Neurons:')
        for neu in self.neurons:
            neu.print()
            print('\n')

class network(object):
    
    """
    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs
        ranges (list of floats): Ranges of inputs and range of outputs
        layers (list of layer): Network Layers
        unsafe_region (list of ndarray):coeffient of output and a constant
        property_flag (bool) : indicates the network have verification layer or not
        property_region (float) : Area of the input box
        abs_mode_changed (int) : count of uncertain relu abstract mode changed
        self.MODE_ROBUSTNESS=0
        self.MODE_QUANTITIVE=1
    """

    def __init__(self):
        self.MODE_QUANTITIVE=0
        self.MODE_ROBUSTNESS=1

        self.numlayers=None
        self.layerSizes=None
        self.inputSize=None
        self.outputSize=None
        self.mins=None
        self.maxes=None
        self.ranges=None
        self.layers=None
        self.property_flag=None
        self.property_region=None
        self.abs_mode_changed=None
    
    def clear(self):
        for i in range(len(self.layers)):
            self.layers[i].clear()
    
    def verify_lp_split(self,PROPERTY,DELTA,MAX_ITER=5,SPLIT_NUM=0,WORKERS=12,TRIM=False,SOLVER=cp.GUROBI,MODE=0):
        if SPLIT_NUM>self.inputSize:
            SPLIT_NUM=self.inputSize
        if self.property_flag==True:
                self.layers.pop()
                self.property_flag=False
        self.load_robustness(PROPERTY,DELTA,TRIM=TRIM)
        delta_list=[self.layers[0].neurons[i].concrete_upper-self.layers[0].neurons[i].concrete_lower for i in range(self.inputSize)]
        self.clear()
        self.deeppoly()
        verify_layer=self.layers[-1]
        verify_neuron_upper=np.array([neur.concrete_upper for neur in verify_layer.neurons])
        verify_list=np.argsort(verify_neuron_upper)
        for i in range(len(verify_list)):
            if verify_neuron_upper[verify_list[i]]>=0:
                verify_list=verify_list[i:]
                break
        if verify_neuron_upper[verify_list[0]]<0:
            print("Property Verified")
            if MODE==self.MODE_ROBUSTNESS:
                return True
            return
        # print(verify_neuron_upper)
        # print(verify_list)
        # return
        split_list=[]        
        for i in range(2**SPLIT_NUM):
            cur_split=[]
            for j in range(SPLIT_NUM):
                if i & (2**j) == 0:
                    cur_split.append([self.layers[0].neurons[j].concrete_lower,(self.layers[0].neurons[j].concrete_upper+self.layers[0].neurons[j].concrete_lower)/2])
                else:
                    cur_split.append([(self.layers[0].neurons[j].concrete_upper+self.layers[0].neurons[j].concrete_lower)/2,self.layers[0].neurons[j].concrete_upper])
            for j in range(SPLIT_NUM,self.inputSize):
                cur_split.append([self.layers[0].neurons[j].concrete_lower,self.layers[0].neurons[j].concrete_upper])
            split_list.append(cur_split)
        
        # Below using executor
        # executor = ProcessPoolExecutor(max_workers=WORKERS)
        obj=None
        prob=None
        constraints=None
        variables=[]
        for i in range(len(self.layers)):
            variables.append(cp.Variable(self.layers[i].size))
        unsafe_set=set()
        unsafe_set_deeppoly=set()
        unsafe_area_list=np.zeros(len(split_list))
        verified_list=[]
        verified_area=0
        for i in verify_list:
            verification_neuron=self.layers[-1].neurons[i]
            total_area=0
            for splits_num in range(len(split_list)):
                splits=split_list[splits_num]
                assert(len(splits)==self.inputSize)
                for j in range(self.inputSize):
                    self.layers[0].neurons[j].concrete_lower=splits[j][0]
                    self.layers[0].neurons[j].concrete_algebra_lower=np.array([splits[j][0]])
                    self.layers[0].neurons[j].algebra_lower=np.array([splits[j][0]])
                    self.layers[0].neurons[j].concrete_upper=splits[j][1]
                    self.layers[0].neurons[j].concrete_algebra_upper=np.array([splits[j][1]])
                    self.layers[0].neurons[j].algebra_upper=np.array([splits[j][1]])            
                self.clear()
                for j in range(MAX_ITER):
                    self.deeppoly()
                    print("Abstract Mode Changed:",self.abs_mode_changed)
                    if (j==0) and (verification_neuron.concrete_upper>0):
                        unsafe_set_deeppoly.add(splits_num)
                    constraints=[]
                    #Build Constraints for each layer
                    for k in range(len(self.layers)):
                        cur_layer=self.layers[k]
                        cur_neuron_list=cur_layer.neurons
                        if cur_layer.layer_type==layer.INPUT_LAYER:
                            for p in range(cur_layer.size):
                                constraints.append(variables[k][p]>=cur_neuron_list[p].concrete_lower)
                                constraints.append(variables[k][p]<=cur_neuron_list[p].concrete_upper)
                        elif cur_layer.layer_type==layer.AFFINE_LAYER:
                            assert(k>0)
                            for p in range(cur_layer.size):
                                constraints.append(variables[k][p]==cur_neuron_list[p].weight@variables[k-1]+cur_neuron_list[p].bias)
                        elif cur_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==self.layers[k-1].size)
                            assert(k>0)
                            for p in range(cur_layer.size):
                                constraints.append(variables[k][p]<=cur_neuron_list[p].algebra_upper[:-1]@variables[k-1]+cur_neuron_list[p].algebra_upper[-1])
                                # constraints.append(variables[k][p]>=cur_neuron_list[p].algebra_lower[:-1]@variables[k-1]+cur_neuron_list[p].algebra_lower[-1])
                                # Modified:using two lower bounds
                                constraints.append(variables[k][p]>=0)
                                constraints.append(variables[k][p]>=variables[k-1][p])
                    #Build the verification neuron constraint
                    for k in verified_list:
                        # print("Verified Neuron :",k)
                        constraints.append(variables[-1][k]<=0)
                    #Modified:If MODE IS ROBUSTNESS THEN CONSTRAINTS CAN BE ==0
                    if MODE==self.MODE_ROBUSTNESS:
                        constraints.append(variables[-1][i]==0)
                    else:
                        constraints.append(variables[-1][i]>=0)
                    
                    #Check the feasibility
                    prob=cp.Problem(cp.Maximize(0),constraints)
                    prob.solve(solver=SOLVER)
                    if prob.status!=cp.OPTIMAL:
                        # print(prob.status)
                        print("Split:",splits_num,"Infeasible")
                        break

                    # #Refresh the input layer bounds
                    # input_neurons=self.layers[0].neurons
                    # for k in range(self.inputSize):
                    #     obj=cp.Minimize(variables[0][k])
                    #     prob=cp.Problem(obj,constraints)
                    #     prob.solve()
                    #     if prob.status!=cp.OPTIMAL:
                    #         print(verification_neuron.concrete_upper)
                    #         print(prob.status)
                    #         break
                    #     # assert(prob.value>=input_neurons[k].concrete_lower)
                    #     if prob.value>=input_neurons[k].concrete_lower:
                    #         input_neurons[k].concrete_lower=prob.value
                    #         input_neurons[k].concrete_algebra_lower=np.array([prob.value])
                    #         input_neurons[k].algebra_lower=np.array([prob.value])

                    #     obj=cp.Maximize(variables[0][k])
                    #     prob=cp.Problem(obj,constraints)
                    #     prob.solve()                    
                    #     if prob.status!=cp.OPTIMAL:
                    #         print(verification_neuron.concrete_upper)
                    #         print(prob.status)
                    #         break
                    #     # assert(prob.value<=input_neurons[k].concrete_upper)
                    #     if prob.value<=input_neurons[k].concrete_upper:
                    #         input_neurons[k].concrete_upper=prob.value
                    #         input_neurons[k].concrete_algebra_upper=np.array([prob.value])
                    #         input_neurons[k].algebra_upper=np.array([prob.value])
                    # if prob.status!=cp.OPTIMAL:
                    #     break

                    #Refresh the input layer bounds
                    mppool=mp.Pool(WORKERS)
                    tasklist=[]
                    input_neurons=self.layers[0].neurons
                    for k in range(self.inputSize):
                        obj=cp.Minimize(variables[0][k])
                        #Below using mp Pool
                        tasklist.append((variables,constraints,obj,SOLVER))
                        #Below using executor
                        # tasklist.append(executor.submit(lpsolve,variables,constraints,obj,solver=SOLVER))
                        #
                        obj=cp.Maximize(variables[0][k])
                        #Below using mp Pool
                        tasklist.append((variables,constraints,obj,SOLVER))
                        #Below using executor
                        # tasklist.append(executor.submit(lpsolve,variables,constraints,obj,solver=SOLVER))
                    #Below using mp Pool
                    resultlist=mppool.starmap(lpsolve,tasklist)
                    mppool.terminate()
                    # print(resultlist)
                    #Below using executor
                    # wait(tasklist)
                    # resultlist=[task.result() for task in tasklist]
                    # print(resultlist)
                    for k in range(self.inputSize):
                        if resultlist[k*2]>=input_neurons[k].concrete_lower:
                            input_neurons[k].concrete_lower=resultlist[k*2]
                            input_neurons[k].concrete_algebra_lower=np.array([resultlist[k*2]])
                            input_neurons[k].algebra_lower=np.array([resultlist[k*2]])
                        #
                        if resultlist[k*2+1]<=input_neurons[k].concrete_upper:
                            input_neurons[k].concrete_upper=resultlist[k*2+1]
                            input_neurons[k].concrete_algebra_upper=np.array([resultlist[k*2+1]])
                            input_neurons[k].algebra_upper=np.array([resultlist[k*2+1]])

                    #Refresh the uncertain ReLu's lowerbound
                    mppool=mp.Pool(WORKERS)
                    count=0
                    tasklist=[]
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    obj=cp.Minimize(variables[k][p])
                                    #Below using mp Pool
                                    tasklist.append((variables,constraints,obj,SOLVER))
                                    #Below using executor
                                    # tasklist.append(executor.submit(lpsolve,variables,constraints,obj,solver=SOLVER))
                    # Below using mp Pool
                    resultlist=mppool.starmap(lpsolve,tasklist)
                    mppool.terminate()
                    # Below using executor
                    # wait(tasklist)
                    # resultlist=[task.result() for task in tasklist]
                    index=0
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    if resultlist[index]>cur_layer.neurons[p].concrete_highest_lower:
                                        cur_layer.neurons[p].concrete_highest_lower=resultlist[index]
                                    if resultlist[index]>=0:
                                        next_layer.neurons[p].certain_flag=1
                                        count+=1
                                    index+=1

                    #Refresh the uncertain ReLu's upperbound
                    mppool=mp.Pool(WORKERS)
                    tasklist=[]
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    obj=cp.Maximize(variables[k][p])
                                    #Below using mp Pool
                                    tasklist.append((variables,constraints,obj,SOLVER))
                                    #Below using executor
                                    # tasklist.append(executor.submit(lpsolve,variables,constraints,obj,solver=SOLVER))
                    # Below using mp Pool
                    resultlist=mppool.starmap(lpsolve,tasklist)
                    mppool.terminate()
                    # Below using executor
                    # wait(tasklist)
                    # resultlist=[task.result() for task in tasklist]
                    index=0
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                if next_layer.neurons[p].certain_flag==0:
                                    if resultlist[index]<cur_layer.neurons[p].concrete_lowest_upper:
                                        cur_layer.neurons[p].concrete_lowest_upper=resultlist[index]
                                    if resultlist[index]<=0:
                                        next_layer.neurons[p].certain_flag=2
                                        count+=1
                                    index+=1
                    print('Refreshed ReLu nodes:',count)

                    # count=0
                    # for k in range(len(self.layers)-1):
                    #     cur_layer=self.layers[k]
                    #     next_layer=self.layers[k+1]
                    #     if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                    #         assert(cur_layer.size==next_layer.size)
                    #         for p in range(cur_layer.size):
                    #             if next_layer.neurons[p].certain_flag==0:
                    #                 obj=cp.Maximize(variables[k][p])
                    #                 prob=cp.Problem(obj,constraints)
                    #                 prob.solve()
                    #                 if prob.status==cp.OPTIMAL and prob.value<=0:
                    #                     # print('Cur Layer:',k,'Cur Neuron',p)
                    #                     # print('Previous Upper Bound',cur_layer.neurons[p].concrete_upper)
                    #                     # print('Refreshed Upper Bound',prob.value)
                    #                     next_layer.neurons[p].certain_flag=2
                    #                     count+=1
                    #                 obj=cp.Minimize(variables[k][p])
                    #                 prob=cp.Problem(obj,constraints)
                    #                 prob.solve()
                    #                 if prob.status==cp.OPTIMAL and prob.value>=0:
                    #                     # print('Cur Layer:',k,'Cur Neuron',p)
                    #                     # print('Previous Lower Bound',cur_layer.neurons[p].concrete_lower)
                    #                     # print('Refreshed Lower Bound',prob.value)
                    #                     next_layer.neurons[p].certain_flag=1
                    #                     count+=1
                    # print('Refreshed ReLu nodes:',count)
                    # self.layers[0].print()
                if prob.status==cp.OPTIMAL:
                    area=1
                    for j in range(self.inputSize):
                        # area*=(self.layers[0].neurons[j].concrete_upper-self.layers[0].neurons[j].concrete_lower)/DELTA/2 OLD BUG
                        area*=(self.layers[0].neurons[j].concrete_upper-self.layers[0].neurons[j].concrete_lower)/delta_list[j]
                    # self.layers[0].print()
                    print("Split:",splits_num,"Area:",area*100)
                    if area>0:
                        if MODE==self.MODE_ROBUSTNESS:
                            return False
                        unsafe_area_list[splits_num]+=area
                        unsafe_set.add(splits_num)
                        total_area+=area
            print('verification neuron:',i,'Unsafe Overapproximate(Box)%:',total_area*100)
            verified_area+=total_area
            verified_list.append(i)
        print('Overall Unsafe Overapproximate(Area)%',verified_area*100)
        verified_area=0
        for i in unsafe_area_list:
            if i>1/len(unsafe_area_list):
                verified_area+=1/len(unsafe_area_list)
            else:
                verified_area+=i
        print('Overall Unsafe Overapproximate(Smart Area)%',verified_area*100)
        print('Overall Unsafe Overapproximate(Box)%:',len(unsafe_set)/len(split_list)*100)
        print('Overall Unsafe Overapproximate(Deeppoly)%:',len(unsafe_set_deeppoly)/len(split_list)*100)
        if MODE==self.MODE_ROBUSTNESS:
            return True
        if MODE==self.MODE_QUANTITIVE:
            if verified_area<len(unsafe_set)/len(split_list):
                return [verified_area*100,len(unsafe_set_deeppoly)/len(split_list)*100]
            else:
                return [len(unsafe_set)/len(split_list)*100,len(unsafe_set_deeppoly)/len(split_list)*100]
    
    def disprove(self,PROPERTY,DELTA,MAX_ITER=5,WORKERS=12,TRIM=False,SOLVER=cp.GUROBI):        
        if self.property_flag==True:
                self.layers.pop()
                self.property_flag=False
        self.load_robustness(PROPERTY,DELTA,TRIM=TRIM)
        delta_list=[self.layers[0].neurons[i].concrete_upper-self.layers[0].neurons[i].concrete_lower for i in range(self.inputSize)]
        self.clear()
        self.deeppoly()
        flag=True
        for neuron_i in self.layers[-1].neurons:            
            if neuron_i.concrete_upper>0:
                flag=False
        if flag==True:
            return True
                
        obj=None
        prob=None
        constraints=None
        variables=[]
        for i in range(len(self.layers)):
            variables.append(cp.Variable(self.layers[i].size))        
        
        self.clear()
        for j in range(MAX_ITER):
            self.deeppoly()
            print("Abstract Mode Changed:",self.abs_mode_changed)            
            constraints=[]
            for k in range(len(self.layers)):
                cur_layer=self.layers[k]
                cur_neuron_list=cur_layer.neurons
                if cur_layer.layer_type==layer.INPUT_LAYER:
                    for p in range(cur_layer.size):
                        constraints.append(variables[k][p]>=cur_neuron_list[p].concrete_lower)
                        constraints.append(variables[k][p]<=cur_neuron_list[p].concrete_upper)
                elif cur_layer.layer_type==layer.AFFINE_LAYER:
                    assert(k>0)
                    for p in range(cur_layer.size):
                        constraints.append(variables[k][p]==cur_neuron_list[p].weight@variables[k-1]+cur_neuron_list[p].bias)
                elif cur_layer.layer_type==layer.RELU_LAYER:
                    assert(cur_layer.size==self.layers[k-1].size)
                    assert(k>0)
                    for p in range(cur_layer.size):
                        constraints.append(variables[k][p]<=cur_neuron_list[p].algebra_upper[:-1]@variables[k-1]+cur_neuron_list[p].algebra_upper[-1])                                
                        constraints.append(variables[k][p]>=0)
                        constraints.append(variables[k][p]>=variables[k-1][p])
            for k in range(len(self.layers[-1].neurons)):
                constraints.append(variables[-1][k]<=0)
            prob=cp.Problem(cp.Maximize(0),constraints)
            prob.solve(solver=SOLVER)
            if prob.status!=cp.OPTIMAL:                        
                return False                

            mppool=mp.Pool(WORKERS)
            tasklist=[]
            input_neurons=self.layers[0].neurons
            for k in range(self.inputSize):
                obj=cp.Minimize(variables[0][k])                        
                tasklist.append((variables,constraints,obj,SOLVER))                        
                obj=cp.Maximize(variables[0][k])                        
                tasklist.append((variables,constraints,obj,SOLVER))                        
            resultlist=mppool.starmap(lpsolve,tasklist)
            mppool.terminate()                    
            for k in range(self.inputSize):
                if resultlist[k*2]>=input_neurons[k].concrete_lower:
                    print("Previous lower:",input_neurons[k].concrete_lower,"Refreshed:",resultlist[k*2])
                    input_neurons[k].concrete_lower=resultlist[k*2]
                    input_neurons[k].concrete_algebra_lower=np.array([resultlist[k*2]])
                    input_neurons[k].algebra_lower=np.array([resultlist[k*2]])                        
                if resultlist[k*2+1]<=input_neurons[k].concrete_upper:
                    print("Previous upper:",input_neurons[k].concrete_upper,"Refreshed:",resultlist[k*2+1])
                    input_neurons[k].concrete_upper=resultlist[k*2+1]
                    input_neurons[k].concrete_algebra_upper=np.array([resultlist[k*2+1]])
                    input_neurons[k].algebra_upper=np.array([resultlist[k*2+1]])
            
            mppool=mp.Pool(WORKERS)
            count=0
            tasklist=[]
            for k in range(len(self.layers)-1):
                cur_layer=self.layers[k]
                next_layer=self.layers[k+1]
                if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                    assert(cur_layer.size==next_layer.size)
                    for p in range(cur_layer.size):
                        if next_layer.neurons[p].certain_flag==0:
                            obj=cp.Minimize(variables[k][p])                                    
                            tasklist.append((variables,constraints,obj,SOLVER))                                    
            resultlist=mppool.starmap(lpsolve,tasklist)
            mppool.terminate()                    
            index=0
            for k in range(len(self.layers)-1):
                cur_layer=self.layers[k]
                next_layer=self.layers[k+1]
                if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                    assert(cur_layer.size==next_layer.size)
                    for p in range(cur_layer.size):
                        if next_layer.neurons[p].certain_flag==0:
                            if resultlist[index]>cur_layer.neurons[p].concrete_highest_lower:
                                cur_layer.neurons[p].concrete_highest_lower=resultlist[index]
                            if resultlist[index]>=0:
                                next_layer.neurons[p].certain_flag=1
                                count+=1
                            index+=1
            
            mppool=mp.Pool(WORKERS)
            tasklist=[]
            for k in range(len(self.layers)-1):
                cur_layer=self.layers[k]
                next_layer=self.layers[k+1]
                if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                    assert(cur_layer.size==next_layer.size)
                    for p in range(cur_layer.size):
                        if next_layer.neurons[p].certain_flag==0:
                            obj=cp.Maximize(variables[k][p])                                    
                            tasklist.append((variables,constraints,obj,SOLVER))                                    
            resultlist=mppool.starmap(lpsolve,tasklist)
            mppool.terminate()                    
            index=0
            for k in range(len(self.layers)-1):
                cur_layer=self.layers[k]
                next_layer=self.layers[k+1]
                if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                    assert(cur_layer.size==next_layer.size)
                    for p in range(cur_layer.size):
                        if next_layer.neurons[p].certain_flag==0:
                            if resultlist[index]<cur_layer.neurons[p].concrete_lowest_upper:
                                cur_layer.neurons[p].concrete_lowest_upper=resultlist[index]
                            if resultlist[index]<=0:
                                next_layer.neurons[p].certain_flag=2
                                count+=1
                            index+=1
            print('Refreshed ReLu nodes:',count)
        if prob.status==cp.OPTIMAL:
            area=1
            for j in range(self.inputSize):                        
                area*=(self.layers[0].neurons[j].concrete_upper-self.layers[0].neurons[j].concrete_lower)/delta_list[j]                    
            print("Over Approximate Safe Area:",area*100)
            if area<1:
                return False
        return True

    def verify_split(self,PROPERTY,DELTA,MAX_ITER=5,SPLIT_NUM=0,REFRESH_INPUT=True):
        if SPLIT_NUM>self.inputSize:
            SPLIT_NUM=self.inputSize
        if self.property_flag==True:
                self.layers.pop()
                self.property_flag=False
        self.load_robustness(PROPERTY,DELTA)        
        self.clear()
        self.deeppoly()
        verify_layer=self.layers[-1]
        verify_neuron_list=verify_layer.neurons
        verify_list=[]
        for i in range(len(verify_neuron_list)):
            if verify_layer.neurons[i].concrete_upper>=0:
                verify_list.append(i)
        split_list=[]        
        for i in range(2**SPLIT_NUM):
            cur_split=[]
            for j in range(SPLIT_NUM):
                if i & (2**j) == 0:
                    cur_split.append([self.layers[0].neurons[j].concrete_lower,(self.layers[0].neurons[j].concrete_upper+self.layers[0].neurons[j].concrete_lower)/2])
                else:
                    cur_split.append([(self.layers[0].neurons[j].concrete_upper+self.layers[0].neurons[j].concrete_lower)/2,self.layers[0].neurons[j].concrete_upper])
            for j in range(SPLIT_NUM,self.inputSize):
                cur_split.append([self.layers[0].neurons[j].concrete_lower,self.layers[0].neurons[j].concrete_upper])
            split_list.append(cur_split)
        # print(split_list)
        # self.layers[0].neurons[0].concrete_lower=-100
        # print(split_list)        
        infeasi_flag=False
        obj=None
        prob=None
        constraints=None
        var=None
        var=cp.Variable(self.inputSize)
        unsafe_set=set()
        unsafe_set_deeppoly=set()
        for i in verify_list:
            verification_neuron=self.layers[-1].neurons[i]
            total_area=0
            for splits_num in range(len(split_list)):
                splits=split_list[splits_num]
                assert(len(splits)==self.inputSize)
                for j in range(self.inputSize):
                    self.layers[0].neurons[j].concrete_lower=splits[j][0]
                    self.layers[0].neurons[j].concrete_algebra_lower=np.array([splits[j][0]])
                    self.layers[0].neurons[j].algebra_lower=np.array([splits[j][0]])
                    self.layers[0].neurons[j].concrete_upper=splits[j][1]
                    self.layers[0].neurons[j].concrete_algebra_upper=np.array([splits[j][1]])
                    self.layers[0].neurons[j].algebra_upper=np.array([splits[j][1]])            
                self.clear()            
                for j in range(MAX_ITER):
                    # print("split:",splits_num+1,"total:",len(split_list),'Iter:',j+1,'MAX_ITER:',MAX_ITER)
                    self.deeppoly()
                    if (j==0) and (verification_neuron.concrete_upper>0):
                        unsafe_set_deeppoly.add(splits_num)
                    # self.layers[-1].print()
                    constraints=[]
                    input_neurons=self.layers[0].neurons
                    #Build the input constraints
                    for k in range(self.inputSize):
                        constraints.append(var[k]<=input_neurons[k].concrete_upper)
                        constraints.append(var[k]>=input_neurons[k].concrete_lower)
                    #Build the verification constraints
                    constraints.append(verification_neuron.concrete_algebra_upper[:-1]@var+verification_neuron.concrete_algebra_upper[-1]>=0)
                    # print(REFRESH_INPUT,j,MAX_ITER,(REFRESH_INPUT) or (j==MAX_ITER-1))
                    #Refresh the input layer bounds
                    if (REFRESH_INPUT) or (j==MAX_ITER-1):
                        infeasi_flag=False
                        for k in range(self.inputSize):
                            if verification_neuron.concrete_algebra_upper[k]>0:
                                #Refresh the lower concrete bound of input neuron k
                                aux=0
                                for k1 in range(self.inputSize):
                                    if k!=k1:
                                        aux2=-verification_neuron.concrete_algebra_upper[k1]/verification_neuron.concrete_algebra_upper[k]
                                        if aux2>0:
                                            aux+=aux2*input_neurons[k1].concrete_lower
                                        else:
                                            aux+=aux2*input_neurons[k1].concrete_upper
                                aux+=-verification_neuron.concrete_algebra_upper[-1]/verification_neuron.concrete_algebra_upper[k]
                                # print(input_neurons[k].concrete_lower,aux,input_neurons[k].concrete_upper,'Refresh lower')
                                if aux>input_neurons[k].concrete_upper:
                                    infeasi_flag=True
                                    break
                                if aux>input_neurons[k].concrete_lower:
                                    input_neurons[k].concrete_lower=aux
                                    input_neurons[k].concrete_algebra_lower=np.array([aux])
                                    input_neurons[k].algebra_lower=np.array([aux])                                
                            elif verification_neuron.concrete_algebra_upper[k]<0:
                                #Refresh the upper concrete bound of input neuron k
                                aux=0
                                for k1 in range(self.inputSize):
                                    if k!=k1:
                                        aux2=-verification_neuron.concrete_algebra_upper[k1]/verification_neuron.concrete_algebra_upper[k]
                                        if aux2>0:
                                            aux+=aux2*input_neurons[k1].concrete_upper
                                        else:
                                            aux+=aux2*input_neurons[k1].concrete_lower
                                aux+=-verification_neuron.concrete_algebra_upper[-1]/verification_neuron.concrete_algebra_upper[k]
                                # print(input_neurons[k].concrete_lower,aux,input_neurons[k].concrete_upper,'Refresh upper')
                                if aux<input_neurons[k].concrete_lower:
                                    infeasi_flag=True
                                    break
                                if aux<input_neurons[k].concrete_upper:
                                    input_neurons[k].concrete_upper=aux
                                    input_neurons[k].concrete_algebra_upper=np.array([aux])
                                    input_neurons[k].algebra_upper=np.array([aux])
                        if infeasi_flag:
                            break
                        #     # print(k)
                        #     obj=cp.Minimize(var[k])
                        #     prob=cp.Problem(obj,constraints)
                        #     # prob.solve()
                        #     prob.solve(solver=cp.GUROBI)
                        #     if prob.status!=cp.OPTIMAL:
                        #         # print(verification_neuron.concrete_upper)
                        #         # print(prob.status)
                        #         break
                        #     # print(input_neurons[k].concrete_lower)
                        #     # print('MIN',k,prob.value)
                        #     # assert(prob.value>=input_neurons[k].concrete_lower)
                        #     if prob.value>=input_neurons[k].concrete_lower:
                        #         input_neurons[k].concrete_lower=prob.value
                        #         input_neurons[k].concrete_algebra_lower=np.array([prob.value])
                        #         input_neurons[k].algebra_lower=np.array([prob.value])

                        #     obj=cp.Maximize(var[k])
                        #     prob=cp.Problem(obj,constraints)
                        #     # prob.solve()    
                        #     prob.solve(solver=cp.GUROBI)
                        #     if prob.status!=cp.OPTIMAL:
                        #         # print(verification_neuron.concrete_upper)
                        #         # print(prob.status)
                        #         break
                        #     # print(input_neurons[k].concrete_upper)
                        #     # print('MAX',k,prob.value)
                        #     # assert(prob.value<=input_neurons[k].concrete_upper)
                        #     if prob.value<=input_neurons[k].concrete_upper:
                        #         input_neurons[k].concrete_upper=prob.value
                        #         input_neurons[k].concrete_algebra_upper=np.array([prob.value])
                        #         input_neurons[k].algebra_upper=np.array([prob.value])
                        # if prob.status!=cp.OPTIMAL:
                        #     break

                    #Refresh the uncertain ReLu
                    count=0
                    for k in range(len(self.layers)-1):
                        cur_layer=self.layers[k]
                        next_layer=self.layers[k+1]
                        if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                            assert(cur_layer.size==next_layer.size)
                            for p in range(cur_layer.size):
                                # print(p)
                                if next_layer.neurons[p].certain_flag==0:
                                    auxvar=[]
                                    for k1 in range(self.inputSize):
                                        if cur_layer.neurons[p].concrete_algebra_upper[k1]>0:
                                            auxvar.append(input_neurons[k1].concrete_upper)
                                        else:
                                            auxvar.append(input_neurons[k1].concrete_lower)
                                    aux=auxvar@cur_layer.neurons[p].concrete_algebra_upper[:-1]+cur_layer.neurons[p].concrete_algebra_upper[-1]
                                    if aux<=0:
                                        next_layer.neurons[p].certain_flag=2
                                        count+=1
                                        continue
                                    elif verification_neuron.concrete_algebra_upper[:-1]@auxvar+verification_neuron.concrete_algebra_upper[-1]<0:
                                        obj=cp.Maximize(cur_layer.neurons[p].concrete_algebra_upper[:-1]@var+cur_layer.neurons[p].concrete_algebra_upper[-1])
                                        prob=cp.Problem(obj,constraints)
                                        # prob.solve()
                                        prob.solve(solver=cp.GUROBI)
                                        if prob.status==cp.OPTIMAL and prob.value<=0:
                                            # print('Cur Layer:',k,'Cur Neuron',p)
                                            # print('Previous Upper Bound',cur_layer.neurons[p].concrete_upper)
                                            # print('Refreshed Upper Bound',prob.value)
                                            next_layer.neurons[p].certain_flag=2                                        
                                            count+=1
                                            continue
                                        if prob.status!=cp.OPTIMAL:                                
                                            break
                                    auxvar=[]
                                    for k1 in range(self.inputSize):
                                        if cur_layer.neurons[p].concrete_algebra_lower[k1]>0:
                                            auxvar.append(input_neurons[k1].concrete_lower)
                                        else:
                                            auxvar.append(input_neurons[k1].concrete_upper)
                                    aux=auxvar@cur_layer.neurons[p].concrete_algebra_lower[:-1]+cur_layer.neurons[p].concrete_algebra_lower[-1]
                                    if aux>=0:
                                        next_layer.neurons[p].certain_flag=1
                                        count+=1
                                    elif verification_neuron.concrete_algebra_upper[:-1]@auxvar+verification_neuron.concrete_algebra_upper[-1]<0:
                                        obj=cp.Minimize(cur_layer.neurons[p].concrete_algebra_lower[:-1]@var+cur_layer.neurons[p].concrete_algebra_lower[-1])
                                        prob=cp.Problem(obj,constraints)
                                        # prob.solve()
                                        prob.solve(solver=cp.GUROBI)
                                        if prob.status==cp.OPTIMAL and prob.value>=0:
                                            # print('Cur Layer:',k,'Cur Neuron',p)
                                            # print('Previous Lower Bound',cur_layer.neurons[p].concrete_lower)
                                            # print('Refreshed Lower Bound',prob.value)
                                            next_layer.neurons[p].certain_flag=1
                                            count+=1
                                        if prob.status!=cp.OPTIMAL:                                
                                            break
                    if (prob!=None) and (prob.status!=cp.OPTIMAL):
                        break
                    # print('Refreshed ReLu nodes:',count)
                    # self.layers[0].print()
                if (not infeasi_flag) and ((prob==None) or (prob.status==cp.OPTIMAL)):
                    area=1
                    for j in range(self.inputSize):
                        area*=(self.layers[0].neurons[j].concrete_upper-self.layers[0].neurons[j].concrete_lower)/DELTA/2
                    if area>0:
                        unsafe_set.add(splits_num)
                    total_area+=area
                else:
                    pass
                    # print(prob.status)
            # print('verification neuron:',i,'Unsafe Overapproximate(Box)%:',total_area/self.property_region*100)
            print('verification neuron:',i,'Unsafe Overapproximate(Box)%:',total_area*100)
        print('Overall Unsafe Overapproximate(Box)%:',len(unsafe_set)/len(split_list)*100)
        print('Overall Unsafe Overapproximate(Deeppoly)%:',len(unsafe_set_deeppoly)/len(split_list)*100)

    def verify_algebra(self,PROPERTY,DELTA,MAX_ITER=5):
        if self.property_flag==True:
                self.layers.pop()
                self.property_flag=False
        self.load_robustness(PROPERTY,DELTA)        
        self.clear()
        self.deeppoly()
        verify_layer=self.layers[-1]
        verify_neuron_list=verify_layer.neurons
        verify_list=[]
        for i in range(len(verify_neuron_list)):
            if verify_layer.neurons[i].concrete_upper>=0:
                verify_list.append(i)
        obj=None
        prob=None
        constraints=None
        variables=[]
        for i in range(len(self.layers)):
            variables.append(cp.Variable(self.layers[i].size))
        for i in verify_list:
            if self.property_flag==True:
                self.layers.pop()
                self.property_flag=False
            self.load_robustness(PROPERTY,DELTA)
            self.clear()
            verification_neuron=self.layers[-1].neurons[i]
            for j in range(MAX_ITER):
                self.deeppoly()
                constraints=[]
                #Build Constraints for each layer
                for k in range(len(self.layers)):
                    cur_layer=self.layers[k]
                    cur_neuron_list=cur_layer.neurons
                    if cur_layer.layer_type==layer.INPUT_LAYER:
                        for p in range(cur_layer.size):
                            constraints.append(variables[k][p]>=cur_neuron_list[p].concrete_lower)
                            constraints.append(variables[k][p]<=cur_neuron_list[p].concrete_upper)
                    elif cur_layer.layer_type==layer.AFFINE_LAYER:
                        assert(k>0)
                        for p in range(cur_layer.size):
                            constraints.append(variables[k][p]==cur_neuron_list[p].weight@variables[k-1]+cur_neuron_list[p].bias)
                    elif cur_layer.layer_type==layer.RELU_LAYER:
                        assert(k>0)
                        for p in range(cur_layer.size):
                            constraints.append(variables[k][p]<=cur_neuron_list[p].algebra_upper[:-1]@variables[k-1]+cur_neuron_list[p].algebra_upper[-1])
                            constraints.append(variables[k][p]>=cur_neuron_list[p].algebra_lower[:-1]@variables[k-1]+cur_neuron_list[p].algebra_lower[-1])
                #Build the verification neuron constraint
                constraints.append(variables[-1][i]>=0)
                
                #Refresh the input layer bounds
                input_neurons=self.layers[0].neurons
                for k in range(self.inputSize):
                    obj=cp.Minimize(variables[0][k])
                    prob=cp.Problem(obj,constraints)
                    prob.solve()
                    if prob.status!=cp.OPTIMAL:
                        print(verification_neuron.concrete_upper)
                        print(prob.status)
                        break
                    print(input_neurons[k].concrete_lower)
                    print('MIN',k,prob.value)
                    # assert(prob.value>=input_neurons[k].concrete_lower)
                    if prob.value>=input_neurons[k].concrete_lower:
                        input_neurons[k].concrete_lower=prob.value
                        input_neurons[k].concrete_algebra_lower=np.array([prob.value])
                        input_neurons[k].algebra_lower=np.array([prob.value])

                    obj=cp.Maximize(variables[0][k])
                    prob=cp.Problem(obj,constraints)
                    prob.solve()                    
                    if prob.status!=cp.OPTIMAL:
                        print(verification_neuron.concrete_upper)
                        print(prob.status)
                        break
                    print(input_neurons[k].concrete_upper)
                    print('MAX',k,prob.value)
                    # assert(prob.value<=input_neurons[k].concrete_upper)
                    if prob.value<=input_neurons[k].concrete_upper:
                        input_neurons[k].concrete_upper=prob.value
                        input_neurons[k].concrete_algebra_upper=np.array([prob.value])
                        input_neurons[k].algebra_upper=np.array([prob.value])
                if prob.status!=cp.OPTIMAL:
                    break

                #Refresh the uncertain ReLu
                count=0
                for k in range(len(self.layers)-1):
                    cur_layer=self.layers[k]
                    next_layer=self.layers[k+1]
                    if cur_layer.layer_type==layer.AFFINE_LAYER and next_layer.layer_type==layer.RELU_LAYER:
                        assert(cur_layer.size==next_layer.size)
                        for p in range(cur_layer.size):
                            if next_layer.neurons[p].certain_flag==0:
                                obj=cp.Maximize(variables[k][p])
                                prob=cp.Problem(obj,constraints)
                                prob.solve()
                                if prob.status==cp.OPTIMAL and prob.value<=0:
                                    print('Cur Layer:',k,'Cur Neuron',p)
                                    print('Previous Upper Bound',cur_layer.neurons[p].concrete_upper)
                                    print('Refreshed Upper Bound',prob.value)
                                    next_layer.neurons[p].certain_flag=2
                                    count+=1
                                obj=cp.Minimize(variables[k][p])
                                prob=cp.Problem(obj,constraints)
                                prob.solve()
                                if prob.status==cp.OPTIMAL and prob.value>=0:
                                    print('Cur Layer:',k,'Cur Neuron',p)
                                    print('Previous Lower Bound',cur_layer.neurons[p].concrete_lower)
                                    print('Refreshed Lower Bound',prob.value)
                                    next_layer.neurons[p].certain_flag=1
                                    count+=1
                print('Refreshed ReLu nodes:',count)
                # self.layers[0].print()
            if prob.status==cp.OPTIMAL:
                area=1
                for j in range(self.inputSize):
                    area*=(self.layers[0].neurons[j].concrete_upper-self.layers[0].neurons[j].concrete_lower)
                print('verification neuron:',i,'Unsafe Overapproximate(Box)%:',area/self.property_region*100)

    def deeppoly(self):

        def pre(cur_neuron,i):
            if i==0:
                cur_neuron.concrete_algebra_lower=deepcopy(cur_neuron.algebra_lower)
                cur_neuron.concrete_algebra_upper=deepcopy(cur_neuron.algebra_upper)
            lower_bound=deepcopy(cur_neuron.algebra_lower)
            upper_bound=deepcopy(cur_neuron.algebra_upper)
            for k in range(i+1)[::-1]:
                # print(k)
                tmp_lower=np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                tmp_upper=np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                assert(self.layers[k].size+1==len(lower_bound))
                assert(self.layers[k].size+1==len(upper_bound))
                for p in range(self.layers[k].size):
                    if lower_bound[p]>=0:  
                        # print(lower_bound[p]*self.layers[k].neurons[p].algebra_lower)                                 
                        tmp_lower+=lower_bound[p]*self.layers[k].neurons[p].algebra_lower
                    else:
                        # print(lower_bound[p]*self.layers[k].neurons[p].algebra_upper)
                        tmp_lower+=lower_bound[p]*self.layers[k].neurons[p].algebra_upper

                    if upper_bound[p]>=0:                        
                        tmp_upper+=upper_bound[p]*self.layers[k].neurons[p].algebra_upper
                    else:
                        tmp_upper+=upper_bound[p]*self.layers[k].neurons[p].algebra_lower                
                # print(tmp_lower)
                tmp_lower[-1]+=lower_bound[-1]
                tmp_upper[-1]+=upper_bound[-1]
                lower_bound=deepcopy(tmp_lower)
                upper_bound=deepcopy(tmp_upper)
                if k==1:
                    cur_neuron.concrete_algebra_upper=deepcopy(upper_bound)
                    cur_neuron.concrete_algebra_lower=deepcopy(lower_bound)
            assert(len(lower_bound)==1)
            assert(len(upper_bound)==1)
            cur_neuron.concrete_lower=lower_bound[0]
            cur_neuron.concrete_upper=upper_bound[0]
            #add lowest and uppest history
            if (cur_neuron.concrete_highest_lower==None) or (cur_neuron.concrete_highest_lower<cur_neuron.concrete_lower):
                cur_neuron.concrete_highest_lower=cur_neuron.concrete_lower
            if (cur_neuron.concrete_lowest_upper==None) or (cur_neuron.concrete_lowest_upper>cur_neuron.concrete_upper):
                cur_neuron.concrete_lowest_upper=cur_neuron.concrete_upper            


        self.abs_mode_changed=0
        for i in range(len(self.layers)-1):
            # print('i=',i)
            pre_layer=self.layers[i]
            cur_layer=self.layers[i+1]
            pre_neuron_list=pre_layer.neurons
            cur_neuron_list=cur_layer.neurons
            if cur_layer.layer_type==layer.AFFINE_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron=cur_neuron_list[j]
                    cur_neuron.algebra_lower=np.append(cur_neuron.weight,[cur_neuron.bias])
                    cur_neuron.algebra_upper=np.append(cur_neuron.weight,[cur_neuron.bias])
                    pre(cur_neuron,i)
            elif cur_layer.layer_type==layer.RELU_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron=cur_neuron_list[j]
                    pre_neuron=pre_neuron_list[j]
                    #modified using lowest and uppest bound
                    if pre_neuron.concrete_highest_lower>=0 or cur_neuron.certain_flag==1:
                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_lower[j]=1
                        cur_neuron.algebra_upper[j]=1
                        cur_neuron.concrete_algebra_lower=deepcopy(pre_neuron.concrete_algebra_lower)
                        cur_neuron.concrete_algebra_upper=deepcopy(pre_neuron.concrete_algebra_upper)
                        cur_neuron.concrete_lower=pre_neuron.concrete_lower
                        cur_neuron.concrete_upper=pre_neuron.concrete_upper
                        #added
                        cur_neuron.concrete_highest_lower=pre_neuron.concrete_highest_lower
                        cur_neuron.concrete_lowest_upper=pre_neuron.concrete_lowest_upper
                        cur_neuron.certain_flag=1
                    elif pre_neuron.concrete_lowest_upper<=0 or cur_neuron.certain_flag==2:
                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)                        
                        cur_neuron.concrete_algebra_lower=np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper=np.zeros(self.inputSize)
                        cur_neuron.concrete_lower=0
                        cur_neuron.concrete_upper=0
                        #added
                        cur_neuron.concrete_highest_lower=0
                        cur_neuron.concrete_lowest_upper=0
                        cur_neuron.certain_flag=2
                    elif pre_neuron.concrete_highest_lower+pre_neuron.concrete_lowest_upper<=0:
                        #Relu abs mode changed
                        if (cur_neuron.prev_abs_mode!=None) and (cur_neuron.prev_abs_mode!=0):
                            self.abs_mode_changed+=1
                        cur_neuron.prev_abs_mode=0

                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        aux=pre_neuron.concrete_lowest_upper/(pre_neuron.concrete_lowest_upper-pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper[j]=aux
                        cur_neuron.algebra_upper[-1]=-aux*pre_neuron.concrete_highest_lower
                        pre(cur_neuron,i)
                    else:
                        #Relu abs mode changed
                        if (cur_neuron.prev_abs_mode!=None) and (cur_neuron.prev_abs_mode!=1):
                            self.abs_mode_changed+=1
                        cur_neuron.prev_abs_mode=1

                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_lower[j]=1
                        aux=pre_neuron.concrete_lowest_upper/(pre_neuron.concrete_lowest_upper-pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper[j]=aux
                        cur_neuron.algebra_upper[-1]=-aux*pre_neuron.concrete_highest_lower
                        pre(cur_neuron,i)

    def print(self):
        print('numlayers:%d' % (self.numLayers))
        print('layerSizes:',self.layerSizes)        
        print('inputSize:%d' % (self.inputSize))
        print('outputSize:%d' % (self.outputSize))
        print('mins:',self.mins)        
        print('maxes:',self.maxes)        
        print('ranges:',self.ranges)
        print('Layers:')
        for l in self.layers:
            l.print()
            print('\n')
    
    def load_property(self, filename):
        self.property_flag=True
        self.property_region=1
        with open(filename) as f:
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata=[float(x) for x in line.strip().split(' ')]
                self.layers[0].neurons[i].concrete_lower=linedata[0]
                self.layers[0].neurons[i].concrete_upper=linedata[1]
                self.property_region*=linedata[1]-linedata[0]
                self.layers[0].neurons[i].concrete_algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper=np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper=np.array([linedata[1]])
                # print(linedata)
            self.unsafe_region=[]
            line=f.readline()
            verify_layer=layer()
            verify_layer.neurons=[]
            while line:                                
                linedata=[float(x) for x in line.strip().split(' ')]
                assert(len(linedata)==self.outputSize+1)
                verify_neuron=neuron()
                verify_neuron.weight=np.array(linedata[:-1])
                verify_neuron.bias=linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata=np.array(linedata)
                # print(linedata)
                self.unsafe_region.append(linedata)
                assert(len(linedata)==self.outputSize+1)
                line=f.readline()
            verify_layer.size=len(verify_layer.neurons)
            verify_layer.layer_type=layer.AFFINE_LAYER
            if len(verify_layer.neurons)>0:
                self.layers.append(verify_layer)

    def load_robustness(self, filename,delta, TRIM=False):
        if self.property_flag==True:
                self.layers.pop()
                # self.clear()
        self.property_flag=True        
        with open(filename) as f:
            self.property_region=1
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata=[float(line.strip())-delta,float(line.strip())+delta]
                if TRIM:
                    if linedata[0]<0:
                        linedata[0]=0
                    if linedata[1]>1:
                        linedata[1]=1
                self.layers[0].neurons[i].concrete_lower=linedata[0]
                self.layers[0].neurons[i].concrete_upper=linedata[1]
                self.property_region*=linedata[1]-linedata[0]
                self.layers[0].neurons[i].concrete_algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper=np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper=np.array([linedata[1]])
                # print(linedata)
            self.unsafe_region=[]
            line=f.readline()
            verify_layer=layer()
            verify_layer.neurons=[]
            while line:                                
                linedata=[float(x) for x in line.strip().split(' ')]
                assert(len(linedata)==self.outputSize+1)
                verify_neuron=neuron()
                verify_neuron.weight=np.array(linedata[:-1])
                verify_neuron.bias=linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata=np.array(linedata)
                # print(linedata)
                self.unsafe_region.append(linedata)
                assert(len(linedata)==self.outputSize+1)
                line=f.readline()
            verify_layer.size=len(verify_layer.neurons)
            verify_layer.layer_type=layer.AFFINE_LAYER
            if len(verify_layer.neurons)>0:
                self.layers.append(verify_layer)
    
    def load_nnet(self, filename):
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line=f.readline() 
                cnt+= 1
            #numLayers does't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line=f.readline()

            #input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

            line=f.readline()
            symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

            #process the input layer
            self.layers=[]
            new_layer=layer()
            new_layer.layer_type=layer.INPUT_LAYER
            new_layer.size=layerSizes[0]
            new_layer.neurons=[]
            for i in range(layerSizes[0]):
                new_neuron=neuron()                
                new_layer.neurons.append(new_neuron)
            self.layers.append(new_layer)

            for layernum in range(numLayers):

                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum+1]
                new_layer=layer()
                new_layer.size=currentLayerSize
                new_layer.layer_type=layer.AFFINE_LAYER
                new_layer.neurons=[]
                for i in range(currentLayerSize):
                    line=f.readline()
                    new_neuron=neuron()
                    aux = [float(x) for x in line.strip().split(",")[:-1]]
                    assert(len(aux)==previousLayerSize)
                    new_neuron.weight=np.array(aux)
                    new_layer.neurons.append(new_neuron)

                #biases                
                for i in range(currentLayerSize):
                    line=f.readline()
                    x = float(line.strip().split(",")[0])
                    new_layer.neurons[i].bias=x
                
                self.layers.append(new_layer)                

                #add relu layer
                if layernum+1==numLayers:
                    break
                new_layer=layer()
                new_layer.size=currentLayerSize
                new_layer.layer_type=layer.RELU_LAYER
                new_layer.neurons=[]
                for i in range(currentLayerSize):
                    new_neuron=neuron()
                    new_layer.neurons.append(new_neuron)
                self.layers.append(new_layer)

            self.numLayers = numLayers
            self.layerSizes = layerSizes
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges  
            self.property_flag=False        

    def load_rlv(self,filename):        
        layersize=[]
        dicts=[]        
        self.layers=[]
        with open(filename,'r') as f:
            line=f.readline()
            while(line):            
                if(line.startswith('#')):
                    linedata=line.replace('\n','').split(' ')
                    layersize.append(int(linedata[3]))
                    layerdict={}
                    if(linedata[4]=='Input'):
                        new_layer=layer()
                        new_layer.layer_type=layer.INPUT_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_layer.neurons.append(new_neuron)
                            line=f.readline()
                            linedata=line.split(' ')
                            layerdict[linedata[1].replace('\n','')]=i
                        dicts.append(layerdict)
                        self.layers.append(new_layer)
                    elif (linedata[4]=='ReLU'):
                        new_layer=layer()
                        new_layer.layer_type=layer.AFFINE_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]                        
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_neuron.weight=np.zeros(layersize[-2])
                            line=f.readline()
                            linedata=line.replace('\n','').split(' ')                        
                            layerdict[linedata[1]]=i
                            new_neuron.bias=float(linedata[2])                            
                            nodeweight=linedata[3::2]
                            nodename=linedata[4::2]
                            assert(len(nodeweight)==len(nodename))
                            for j in range(len(nodeweight)):
                                new_neuron.weight[dicts[-1][nodename[j]]]=float(nodeweight[j])
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)                                     
                        dicts.append(layerdict)
                        #add relu layer
                        new_layer=layer()
                        new_layer.layer_type=layer.RELU_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)
                    elif (linedata[4]=='Linear') and (linedata[5]!='Accuracy'):
                        new_layer=layer()
                        new_layer.layer_type=layer.AFFINE_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]                        
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_neuron.weight=np.zeros(layersize[-2])
                            line=f.readline()
                            linedata=line.replace('\n','').split(' ')                        
                            layerdict[linedata[1]]=i
                            new_neuron.bias=float(linedata[2])                            
                            nodeweight=linedata[3::2]
                            nodename=linedata[4::2]
                            assert(len(nodeweight)==len(nodename))
                            for j in range(len(nodeweight)):
                                new_neuron.weight[dicts[-1][nodename[j]]]=float(nodeweight[j])
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)                                     
                        dicts.append(layerdict)
                line=f.readline()
        self.layerSizes=layersize
        self.inputSize=layersize[0]
        self.outputSize=layersize[-1]
        self.numLayers=len(layersize)-1
        pass

    def find_max_disturbance(self,PROPERTY,L=0,R=1000,TRIM=False):
        ans=0
        while L<=R:
            # print(L,R)
            mid=int((L+R)/2)
            self.load_robustness(PROPERTY,mid/1000,TRIM=TRIM)
            self.clear()
            self.deeppoly()
            flag=True
            for neuron_i in self.layers[-1].neurons:
                # print(neuron_i.concrete_upper)
                if neuron_i.concrete_upper>0:
                    flag=False
            if flag==True:
                ans=mid/1000
                L=mid+1
            else:
                R=mid-1
        return ans
    
    def find_max_disturbance_lp(self,PROPERTY,L,R,TRIM,WORKERS=12,SOLVER=cp.GUROBI):
        ans=L
        while L<=R:
            mid=int((L+R)/2)
            if self.verify_lp_split(PROPERTY=PROPERTY,DELTA=mid/1000,MAX_ITER=5,SPLIT_NUM=0,WORKERS=WORKERS,TRIM=TRIM,SOLVER=SOLVER,MODE=1):
                print("Disturbance:",mid/1000,"Success!")
                ans=mid/1000
                L=mid+1
            else:
                print("Disturbance:",mid/1000,"Failed!")
                R=mid-1
        return ans

def main():
    # #Below is Acas Split experiment
    # net_list=['nnet/ACASXU_experimental_v2a_4_2.nnet','nnet/ACASXU_experimental_v2a_4_3.nnet','nnet/ACASXU_experimental_v2a_4_4.nnet']
    # property_list=['properties/local_robustness_2.txt','properties/local_robustness_3.txt','properties/local_robustness_4.txt','properties/local_robustness_5.txt','properties/local_robustness_6.txt']
    # # disturbance_list=[0.02,0.03,0.04]
    # disturbance_list=[0.05,0.06]
    # rlist=[]
    # for net_i in net_list:
    #     plist=[]
    #     for property_i in property_list:
    #         net=network()
    #         net.load_nnet(net_i)
    #         delta_base=net.find_max_disturbance(PROPERTY=property_i)
    #         dlist=[]
    #         for disturbance_i in disturbance_list:
    #             print("Net:",net_i,"Property:",property_i,"Delta:",delta_base+disturbance_i)
    #             start=time.time()
    #             net=network()
    #             net.load_nnet(net_i)
    #             dlist.append(net.verify_lp_split(PROPERTY=property_i,DELTA=delta_base+disturbance_i,MAX_ITER=5,WORKERS=96,SPLIT_NUM=5,SOLVER=cp.CBC))
    #             end=time.time()
    #             print("Finished Time:",end-start)
    #         plist.append(dlist)
    #     rlist.append(plist)
    # print(rlist)

    # #Below is Mnist max robustness experiment
    # net_list=['rlv/caffeprototxt_AI2_MNIST_FNN_'+str(i)+'_testNetworkB.rlv' for i in range(2,8)] #8
    # property_list=['properties/mnist_'+str(i)+'_local_property.in' for i in range(3)] #3
    # rlist=[]
    # for net_i in net_list:
    #     plist=[]
    #     for property_i in property_list:
    #         print("Net:",net_i,"Property:",property_i)
    #         net=network()
    #         net.load_rlv(net_i)
    #         delta_base=net.find_max_disturbance(PROPERTY=property_i,TRIM=True)            
    #         lp_ans=net.find_max_disturbance_lp(PROPERTY=property_i,L=int(delta_base*1000),R=int(delta_base*1000+63),TRIM=True,WORKERS=96,SOLVER=cp.CBC)
    #         print("DeepPoly Max Verified Distrubance:",delta_base)
    #         print("Our method Max Verified Disturbance:",lp_ans)
    #         plist.append([delta_base,lp_ans])
    #     rlist.append(plist)
    # print(plist)

    #Below is Mnist batch verify experiment
    net_list=['rlv/caffeprototxt_AI2_MNIST_FNN_'+str(i)+'_testNetworkB.rlv' for i in range(6,7)]
    property_list=['properties/mnist_'+str(i)+'_local_property.in' for i in range(50)]
    delta=0.015
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
                # print(neuron_i.concrete_upper)
                if neuron_i.concrete_upper>0:
                    flag=False
            if flag==True:
                count_deeppoly+=1
                print(property_i,'DeepPoly Success!')
            else:
                print(property_i,'DeepPoly Failed!')

            start=time.time()
            if net.verify_lp_split(PROPERTY=property_i,DELTA=delta,MAX_ITER=5,SPLIT_NUM=0,WORKERS=96,TRIM=True,SOLVER=cp.CBC,MODE=1):
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
        print('Min Time:',mintime,'Max Time',maxtime,'Avg Time',avgtime/50)
        print('Passlist:',pass_list)
        print('Nopasslist:',nopass_list)

    # start = time.time()
    # net=network()
    # net.load_nnet('nnet/ACASXU_experimental_v2a_4_2.nnet') 
    # print(net.find_max_disturbance(PROPERTY='properties/local_robustness_4.txt'))
    # net.load_robustness('properties/local_robustness_2.txt',0.05)
    # net.verify_lp_split(PROPERTY='properties/local_robustness_2.txt',DELTA=0.064,MAX_ITER=5,WORKERS=12,SPLIT_NUM=5,SOLVER=cp.GUROBI)

    # net.load_rlv('rlv/caffeprototxt_AI2_MNIST_FNN_2_testNetworkB.rlv')
    # print(net.find_max_disturbance(PROPERTY='properties/mnist_0_local_property.in',TRIM=True))
    # net.verify_lp_split(PROPERTY='properties/mnist_0_local_property.in',DELTA=0.047,TRIM=True,MAX_ITER=5,WORKERS=96,SPLIT_NUM=0,SOLVER=cp.CBC)
    # net.verify_lp_split(PROPERTY='properties/mnist_0_local_property.in',DELTA=0.048,TRIM=True,MAX_ITER=5,WORKERS=12,SPLIT_NUM=0,SOLVER=cp.GUROBI,MODE=1)
    # print(net.disprove(PROPERTY='properties/mnist_0_local_property.in',DELTA=0.1,TRIM=True,MAX_ITER=5,WORKERS=12))
    # net.load_robustness('properties/mnist_0_local_property.in',0.045,TRIM=True)

    # net.deeppoly()
    # flag=True
    # for neuron_i in net.layers[-1].neurons:
    #     print(neuron_i.concrete_upper)
    #     if neuron_i.concrete_upper>0:
    #         flag=False
    # print(flag)
    # end = time.time()
    # print(end-start)
    # pass
    

if __name__ == "__main__":
    main()