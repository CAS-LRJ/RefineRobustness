exp1=[[[[0.0, 15.625], [0.0, 62.5], [13.680864514342419, 96.875]], [[0.0, 43.75], [9.08409759473407, 65.625], [30.07088160142643, 78.125]], [[0.0, 31.25], [0.0, 59.375], [12.816100916414753, 81.25]], [[0.0, 25.0], [1.1087722675538187, 40.625], [21.875, 53.125]], [[0.0, 21.875], [0.0, 59.375], [18.786951310556994, 75.0]]], [[[0.0, 25.0], [0.0, 37.5], [0.0, 56.25]], [[0.0, 50.0], [0.0, 68.75], [0.0, 75.0]], [[0.0, 15.625], [0.0, 37.5], [0.0, 53.125]], [[0.0, 50.0], [0.0, 50.0], [0.0, 62.5]], [[0.0, 31.25], [0.0, 37.5], [0.0, 37.5]]], [[[0.0, 25.0], [0.0, 25.0], [0.0, 68.75]], [[0.0, 15.625], [0.0, 25.0], [0.0, 46.875]], [[0.0, 6.25], [0.0, 28.125], [0.0, 68.75]], [[0.0, 50.0], [0.0, 59.375], [0.0, 75.0]], [[0.0, 18.75], [0.0, 37.5], [0.0, 59.375]]]]
exp2=[[[[25.0, 100.0], [49.736389430115835, 100.0]], [[46.875, 100.0], [56.25, 100.0]], [[39.355252993881706, 96.875], [65.25568853273735, 100.0]], [[40.625, 78.125], [58.95298931158563, 100.0]], [[36.17991446642305, 87.5], [53.125, 90.625]]], [[[0.0, 71.875], [7.106151179494635e-05, 87.5]], [[0.0, 68.75], [0.0, 78.125]], [[0.28480304257476347, 75.0], [6.736490953889272, 93.75]], [[0.0, 75.0], [0.0, 87.5]], [[0.0, 56.25], [0.0, 65.625]]], [[[0.0, 87.5], [0.0, 90.625]], [[0.0, 71.875], [0.0, 93.75]], [[0.0, 100.0], [0.0, 100.0]], [[6.249999999999989, 100.0], [34.374999999999986, 100.0]], [[0.0, 75.0], [3.1249999999999933, 75.0]]]]
exp_002_our=[]
exp_002_deep=[]
exp_003_our=[]
exp_003_deep=[]
exp_004_our=[]
exp_004_deep=[]
exp_005_our=[]
exp_005_deep=[]
exp_006_our=[]
exp_006_deep=[]
for i in range(3):
    for j in range(5):
        exp_002_our.append(exp1[i][j][0][0])
        exp_002_deep.append(exp1[i][j][0][1])

        exp_003_our.append(exp1[i][j][1][0])
        exp_003_deep.append(exp1[i][j][1][1])

        exp_004_our.append(exp1[i][j][2][0])
        exp_004_deep.append(exp1[i][j][2][1])

        exp_005_our.append(exp2[i][j][0][0])
        exp_005_deep.append(exp2[i][j][0][1])

        exp_006_our.append(exp2[i][j][1][0])
        exp_006_deep.append(exp2[i][j][1][1])

print(exp_002_our,exp_002_deep)
print(exp_003_our,exp_003_deep)
print(exp_004_our,exp_004_deep)
print(exp_005_our,exp_005_deep)
print(exp_006_our,exp_006_deep)