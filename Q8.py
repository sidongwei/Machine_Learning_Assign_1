import numpy as np
from data_func import import_data,export_data
from regression_func import lasso,add_bias,plot_err,nonzeros,performance

X = import_data("housing_X_train.csv")
y = import_data("housing_y_train.csv")
X_test = import_data("housing_X_test.csv")
y_test = import_data("housing_y_test.csv")

X = add_bias(X)
X_test = add_bias(X_test)

fold = 10
min_lam = 0
max_lam = 100
step = 10
lam = np.arange(min_lam,max_lam+step,step)
k = len(lam)
zeros = np.zeros(k)

(perf_train,perf_valid,perf_test,w) = performance(X,y,fold,lam,X_test,y_test,lasso)
for i in range(k):                          #count nonzero ratio in all entries of w
    zeros[i] = nonzeros(w[:,i])

plot_err(lam,perf_valid,"Q8")
header = ["lamda","training set error","validation set error","test set error","nonzero ratio"]
DATA = (lam.reshape(k,1),perf_train.reshape(k,1),perf_valid.reshape(k,1),perf_test.reshape(k,1),zeros.reshape(k,1))
result = np.concatenate(DATA,axis=1).astype(str)
export_data(header,result,"Q8")
