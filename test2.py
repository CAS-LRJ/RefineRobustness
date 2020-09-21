import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed,wait
from cvxpy import *
import numpy as np


def fib(n):
    if n < 3:
        return 1
    return fib(n - 1) + fib(n - 2)

def sleepfunc(n):
    time.sleep(n)
    return n

def lpsolve(vars,cons,obj,SOLVER=cvxpy.GUROBI):
    prob=Problem(obj,cons)
    prob.solve(solver=SOLVER)
    return [vars.value,prob.value]
    pass

def echo(n):
    return n

def multi():

    start_time = time.time()
    executor = ProcessPoolExecutor(max_workers=4)
    # task_list = [executor.submit(fib, n) for n in range(3, 35)]
    # process_results = [task.result() for task in as_completed(task_list)]
    task_list = [executor.submit(echo, n) for n in range(3,35)]
    process_results = [task.result() for task in as_completed(task_list)]
    print(process_results)
    print("ProcessPoolExecutor time is: {}".format(time.time() - start_time))
    start_time = time.time()
    task_list = [executor.submit(fib, n) for n in range(3, 35)]
    process_results = [task.result() for task in as_completed(task_list)]
    print(process_results)
    print("ProcessPoolExecutor time is: {}".format(time.time() - start_time))
    start_time = time.time()
    task_list = [executor.submit(sleepfunc, n) for n in [3,1,2]]
    process_results = [task.result() for task in as_completed(task_list)]
    print(process_results)
    print("ProcessPoolExecutor time is: {}".format(time.time() - start_time))
    pass

if __name__=='__main__':
    multiprocessing.freeze_support()
    # multi()

    # executor = ProcessPoolExecutor(max_workers=4)
    # start_time = time.time()
    # task_list = [executor.submit(sleepfunc, n) for n in [3,1,2]]
    # wait(task_list)
    # process_results=[task.result() for task in task_list]
    # print(process_results)
    # print("ProcessPoolExecutor time is: {}".format(time.time() - start_time))
    # start_time = time.time()
    # executor = ProcessPoolExecutor(max_workers=4)
    # # task_list = [executor.submit(fib, n) for n in range(3, 35)]
    # # process_results = [task.result() for task in as_completed(task_list)]
    # task_list = [executor.submit(echo, n) for n in range(3,35)]
    # process_results = [task.result() for task in as_completed(task_list)]
    # print(process_results)
    # print("ProcessPoolExecutor time is: {}".format(time.time() - start_time))
    # A=np.random.rand(784,2)
    # x=Variable(784)
    # constraints=[]
    # for i in range(784):
    #     if A[i][0]<=A[i][1]:
    #         constraints.append(x[i]>=A[i][0])
    #         constraints.append(x[i]<=A[i][1])
    #     else:
    #         constraints.append(x[i]>=A[i][1])
    #         constraints.append(x[i]<=A[i][0])
    # B=np.random.rand(784)
    # B2=np.random.rand(784)
    # constraints.append(B@x>=0)
    # # constraints.append(B2@x>=0)
    # obj=Maximize(B2@x)
    # prob=Problem(obj,constraints)

    x=Variable(2)
    constraints=[x[0]<=2,x[1]<=2,x[0]+x[1]>=2]
    obj=Maximize(x[0]+x[1]*2)
    # prob=Problem(obj,constraints)
    executor = ProcessPoolExecutor(max_workers=4)
    start_time = time.time()
    task_list = [executor.submit(lpsolve, x, constraints, obj)]
    obj=Minimize(x[0]+x[1]*2)
    # prob=Problem(obj,constraints)
    task_list.append(executor.submit(lpsolve, x, constraints, obj))
    wait(task_list)
    # print(task_list)
    process_results=[task.result() for task in task_list]
    print(process_results)
    print("ProcessPoolExecutor time is: {}".format(time.time() - start_time))    
    obj=Maximize(0)    