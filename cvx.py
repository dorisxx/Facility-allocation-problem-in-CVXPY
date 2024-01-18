import cvxpy as cvx
import pandas as pd
import numpy as np

import math

##File path
filepath=''
oh=pd.read_csv(f"{filepath}OHCA.csv")
aa=pd.read_csv(f"{filepath}candidate_site.csv")


##Distance calculation
def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    meter = 6367000 * c
    return meter


##Distance matrix: candidate site <-> OHCA case
##If the distance <= num, aij == 1; Otherwise, -1
def get_distance_data(x, y, num):
    distance_all = np.zeros((x.shape[0], y.shape[0]))
    # x=x.sort_values('Date')
    for i in np.arange(x.shape[0]):
        for j in np.arange(y.shape[0]):
            distance_all[i, j] = np.sign(num - haversine(x.iloc[i, 0], x.iloc[i, 1], y.iloc[j, 0], y.iloc[j, 1]))
            # distance_all[i,j] = haversine(x.iloc[i, 0] , x.iloc[i, 1],y.iloc[j, 0], y.iloc[j, 1])

    return distance_all

##Row: OHCA case
##Column: AED candidate site
##num = 100 meters
distance_matrix=get_distance_data(oh,aa,100)
##Distance <= 100, mark it as 1; Otherwise, 0
distance_matrix[distance_matrix>=0]=1
distance_matrix[distance_matrix==-1]=0


##Defined variables
x=cvx.Variable(distance_matrix.shape[1],boolean=True)
mat=distance_matrix
##Find the maximum value of each OHAC. 
##Maximum value: if it is less than 1 -> no candidate site meets the constraint '<= 100 meters'
loc=np.where(np.max(mat,axis=1)==0)
lb=np.ones(mat.shape[0])
lb[loc]=0
##Constraint
constraints=[mat@x>=lb]
##Objective
obj=cvx.Minimize(cvx.sum(x))
##Create a problem
prob=cvx.Problem(obj,constraints)
##GUROBI solver
prob.solve(solver=cvx.GUROBI)

aa['x']=x.value
aa=aa[aa['x']==1]
##Output the locations of the selected candidate site
aa.to_csv(f'{filepath}candidate_site_result.csv')

