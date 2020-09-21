from cvxpy import *
import numpy as np
import time
# # Create two scalar optimization variables.
# # 在CVXPY中变量有标量(只有数值大小)，向量，矩阵。
# # 在CVXPY中有常量(见下文的Parameter)
# x = Variable() # 定义变量x,定义变量y。两个都是标量
# y = Variable()
# # Create two constraints.
# # 定义两个约束式
# constraints = [x + y == 1,
#               x - y >= 1]
# # 优化的目标函数
# obj = Minimize(square(x - y))
# # 把目标函数与约束传进Problem函数中
# prob = Problem(obj, constraints)
# prob.solve()  # Returns the optimal value.
# print("status:", prob.status)
# print("optimal value", prob.value) # 最优值
# print("optimal var", x.value, y.value) # x与y的解
# # 状态域被赋予'optimal'，说明这个问题被成功解决。
# # 最优值是针对所有满足约束条件的变量x,y中目标函数的最小值
# # prob.solve()返回最优值，同时更新prob.status,prob.value,和所有变量的值。
# A=np.array([1,2])
# x=[Variable(2),Variable(3200)]
# constraints = [x[0][0]+x[0][1]==1,x[0][0]+x[0][1]>=1,x[0][0]>=2]
# # x = Variable(2)
# # constraints = [x[0]+x[1]==1,x[0]+x[1]>=1.2,x[0]>=2]
# obj = Maximize(A@x[0])
# prob = Problem(obj,constraints)
# prob.solve(solver=GLPK)
# print("status:", prob.status)
# print("optimal value", prob.value) # 最优值
# print("optimal var",x[0].value)
# constraints.pop(1)
# prob = Problem(obj,constraints)
# prob.solve(solver=GLPK)
# print("status:", prob.status)
# print("optimal value", prob.value) # 最优值
# print("optimal var",x[0].value)
A=np.random.rand(784,2)
x=Variable(784)
constraints=[]
for i in range(784):
    if A[i][0]<=A[i][1]:
        constraints.append(x[i]>=A[i][0])
        constraints.append(x[i]<=A[i][1])
    else:
        constraints.append(x[i]>=A[i][1])
        constraints.append(x[i]<=A[i][0])
    B=np.random.rand(784)
    constraints.append(B@x>=0)
print(len(constraints))
# B=np.random.rand(784)
B2=np.random.rand(784)
# constraints.append(B@x>=0)
# constraints.append(B2@x>=0)
obj=Maximize(B2@x)
# obj=Maximize(0)
prob=Problem(obj,constraints)
start = time.time()
# prob.solve(solver=cvxpy.GLPK_MI)
prob.solve(solver=cvxpy.CBC)
# prob.solve(solver=cvxpy.GUROBI)
# prob.solve()
print(prob.value)
end = time.time()
print(end-start)

# prob=Problem(Minimize(B2@x),prob.constraints)
# start = time.time()
# # prob.solve(solver=cvxpy.GLPK_MI)
# prob.solve(solver=cvxpy.GUROBI)
# # prob.solve()
# end = time.time()
# print(end-start)

# prob=Problem(Maximize(B2@x),prob.constraints)
# start = time.time()
# # prob.solve(solver=cvxpy.GLPK_MI)
# prob.solve(solver=cvxpy.GUROBI)
# # prob.solve()
# end = time.time()
# print(end-start)

# prob=Problem(Maximize(0),prob.constraints+[B2@x>=0])
# start = time.time()
# # prob.solve(solver=cvxpy.GLPK_MI)
# prob.solve(solver=cvxpy.GUROBI)
# # prob.solve()
# end = time.time()
# print(end-start)

# sol=0
# for i in range(784):
#     sol+=x.value[i]*B[i]
# print('Real sol:',sol)
# print("status:", prob.status)
# print("optimal value", prob.value) # 最优值
# print("optimal var",x.value)

# # class tmp(object):

# #     def __init__(self):
# #         self.a=0

# # a=np.array([1,2])
# # print(a*3)
# # print(a[:-1])
# print(2**3)
# for i in range(1,4):
#     print(i)