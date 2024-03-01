import cvxpy as cvx
import pandas as pd
import numpy as np

import math

oh=pd.read_csv(f"{filepath}OHCA.csv")
aa=pd.read_csv(f"{filepath}candidate_site.csv")

#Reduce the scale
#oh=oh.iloc[:30,:]
#aa=aa.iloc[:50,:]

#Distance calculation
def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    meter = 6367000 * c
    return meter


def get_distance_data(x, y, num):
    distance_all = np.zeros((x.shape[0], y.shape[0]))
    # x=x.sort_values('Date')
    for i in np.arange(x.shape[0]):
        for j in np.arange(y.shape[0]):
            distance_all[i, j] = np.sign(num - haversine(x.iloc[i, 0], x.iloc[i, 1], y.iloc[j, 0], y.iloc[j, 1]))
            # distance_all[i,j] =haversine(x.iloc[i, 0] , x.iloc[i, 1],y.iloc[j, 0], y.iloc[j, 1])

    return distance_all


distance_matrix=get_distance_data(oh,aa,100)

distance_matrix1=distance_matrix.copy()

distance_matrix[distance_matrix>=0]=1
distance_matrix[distance_matrix==-1]=0

def get_solve(mat,mat1):
    
    x=cvx.Variable(mat.shape[1],boolean=True)
    y=cvx.Variable((mat.shape[0],mat.shape[1]),boolean=True)
    loc=np.where(np.max(mat,axis=1)==0)
    lb=np.ones(mat.shape[0])
    lb[loc]=0
    p=1
    constraints=[mat@x>=lb,
                cvx.sum(y,axis=1)==np.ones(mat.shape[0]), #vector
                cvx.sum(y,axis=0)<=mat.shape[0] * x,
                cvx.sum(x)==p]
    obj=cvx.Minimize(cvx.sum(cvx.multiply(mat1,y)))
    prob=cvx.Problem(obj,constraints)
    prob.solve(solver=cvx.GUROBI,verbose=True)
    return x.value

aa['x']=get_solve(distance_matrix,distance_matrix1)
aa=aa[aa['x']==1]
aa.to_csv(f'{filepath}candidate_site.csv')