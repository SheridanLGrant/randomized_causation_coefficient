PATH_X_TR = "syn_pairs.csv"
PATH_Y_TR = "syn_target.csv"
PATH_X_TE = "tuebingen_pairs.csv"
PATH_Y_TE = "tuebingen_target.csv"

import numpy as np
from   sklearn.preprocessing    import scale

import os
os.chdir('C:/Users/Sheridongle/Documents/projects/randomized_causation_coefficient/code/')

import time
import multiprocessing

def featurize_row(row,i,j):
  r  = row.split(",",2)
  x  = scale(np.array(r[i].split(),dtype=np.float))
  y  = scale(np.array(r[j].split(),dtype=np.float))
  return np.hstack((x,y))

def featurize(x,flip):
  f1  = np.array([featurize_row(row,1,2) for row in x])
  if(flip==1):
    return np.vstack((f1,np.array([featurize_row(row,2,1) for row in x])))
  else:
    return f1

def read_pairs(filename):
  f = open(filename);
  pairs = f.readlines();
  f.close();
  del pairs[0];
  return pairs

# Read training data
y = np.genfromtxt(PATH_Y_TR, delimiter=",")[:,1]
x = read_pairs(PATH_X_TR)
# Featurize with flip, i.e. reverse direction
x = featurize(x,1)
y = np.hstack((y,-y))
# Read test data
x_te = read_pairs(PATH_X_TE)
x_te = featurize(x_te,0)
y_te = 2*((np.genfromtxt(PATH_Y_TE,delimiter=",")[:,1])==1)-1
m_te = np.genfromtxt(PATH_Y_TE,delimiter=",")[:,2]

def kernel_dist(p,q,weights=np.ones(3),gamma=10,kernel='gaussian'):
    if kernel=='gaussian':
        k = lambda x,y: np.exp(-gamma*np.dot(x-y,x-y))
    else:
        raise ValueError('Only gaussian kernel implemented so far')
        
    n = int(len(p)/2)
    m = int(len(q)/2)

    d_x = 0
    for i in range(n):
        for j in range(m):
            d_x += k(p[i],q[j])
    d_x /= (n*m)

    d_y = 0
    for i in range(n,2*n):
        for j in range(m,2*m):
            d_y += k(p[i],q[j])
    d_y /= (n*m)
    
    d_xy = 0
    for i in range(n):
        for j in range(m):
            d_xy += k(np.array(p[i],p[i+n]),np.array(q[j],q[j+m]))
    d_xy /= (n*m)
    return np.dot(weights,(d_x,d_y,d_xy))

def classify(test):
    return np.mean([kernel_dist(x[i],test)*y[i] for i in range(len(x))])


num_tests = 10


if __name__ == '__main__':
    indices = np.random.choice(x_te.shape[0], num_tests, False)
    n_cores = multiprocessing.cpu_count()
    start = time.clock()
    with multiprocessing.Pool(n_cores-1) as P:
        scores = np.array(P.map(classify, x_te[indices]))
    end = time.clock()
    print(end-start)
    y_hats = 2*(scores > 0)-1
    print('Accuracy:', np.mean(y_hats== y_te[indices]))
    
