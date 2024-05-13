import cvxpy as cvx
import pandas as pd
import numpy as np
import math

filepath = ''
oh = pd.read_csv(f"{filepath}OHCA_2022_KwunTong.csv")
aa = pd.read_csv(f"{filepath}new_candidate_site.csv")

# Reduce the scale
'''oh = oh.iloc[:30, :]
aa = aa.iloc[:50, :]'''


# Distance calculation
def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    meter = 6367000 * c
    return meter


'''def get_time(x, y):
    time_all = np.zeros((x.shape[0], y.shape[0]))
    # x=x.sort_values('Date')
    for i in np.arange(x.shape[0]):
        for j in np.arange(y.shape[0]):
            # Calculate round trip time in minute, assume the running speed = 200m/min
            time_all[i, j] = haversine(x.iloc[i, 0], x.iloc[i, 1], y.iloc[j, 0], y.iloc[j, 1]) * 2 / 200
    return time_all'''


def get_cost(x, y, num):
    cost_all = np.zeros((x.shape[0], y.shape[0]))
    # x=x.sort_values('Date')
    for i in np.arange(x.shape[0]):
        for j in np.arange(y.shape[0]):
            # Calculate round trip time in minute, assume the running speed = 200m/min
            try:
                cost_all[i, j] = math.exp(
                    (max(0, ((haversine(x.iloc[i, 0], x.iloc[i, 1], y.iloc[j, 0], y.iloc[j, 1]) * 2 / 200) - num))))
            # Solve "overflowerror: math range error"
            except OverflowError:
                cost_all[i, j] = 1000000000

    # cost_all = np.float128(cost_all)
    # cost_all = np.exp(cost_all)
    return cost_all


cost = get_cost(oh, aa, 3)
cost_matrix = cost.copy()


def get_solve(mat, mat1, N):
    x = cvx.Variable(mat.shape[1], boolean=True)
    y = cvx.Variable((mat.shape[0], mat.shape[1]), boolean=True)

    constraints = [
        cvx.sum(y, axis=1) == np.ones(mat.shape[0]),
        cvx.sum(y, axis=0) <= mat.shape[0] * x,
        cvx.sum(x) == N]
    obj = cvx.Minimize(cvx.sum(cvx.multiply(mat1, y)))
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.GUROBI, verbose=True)
    # prob.solve(solver=cvx.SCS,verbose=True)
    print(prob.value)
    return y.value, x.value


'''cost_matrix=get_cost(oh,aa,3)
print(cost_matrix)'''
for N in range(1,26):
    y_all, aa['y'] = get_solve(cost, cost_matrix, N)
    print(np.sum(cost_matrix * y_all))
    y_all = pd.DataFrame(y_all)

    yy = aa['y']
    aa = aa[aa['y'] == 1]

    y_all.to_csv(f"{filepath}time_y_{N}.csv")
    aa.to_csv(f"{filepath}time_candidate_site_y_value{N}.csv")