import numpy as np
from data_func import import_data,export_data
from regression_func import ridge,add_bias,plot_err,performance

X = import_data("housing_X_train.csv")
y = import_data("housing_y_train.csv")
X_test = import_data("housing_X_test.csv")
y_test = import_data("housing_y_test.csv")

X = add_bias(X)
X_test = add_bias(X_test)

n = X.shape[1]
x = np.random.randint(n)
X[:,x] = X[:,x]*1e6           #multiply x and y
y[x] = y[x]*1e3

fold = 10
min_lam = 0
max_lam = 100
step = 10
lam = np.arange(min_lam,max_lam+step,step)
k = len(lam)

(perf_train,perf_valid,perf_test,w) = performance(X,y,fold,lam,X_test,y_test,ridge)

plot_err(lam,perf_valid,"Q6")
header = ["lamda","training set error","validation set error","test set error"]
DATA = (lam.reshape(k,1),perf_train.reshape(k,1),perf_valid.reshape(k,1),perf_test.reshape(k,1))
result = np.concatenate(DATA,axis=1).astype(str)
export_data(header,result,"Q6")
