import numpy as np
import matplotlib.pyplot as plt

def ridge(X,y,lam):             #implement the ridge regression
    d = X.shape[0]
    if lam<0:
        return -1
    A = np.dot(X,X.T)+2*lam*np.eye(d)
    b = np.dot(X,y)
    w = np.linalg.solve(A,b)
    return w            #return w as a k*1 vector

def soft_thre(w,lam):           #implement soft-thresholding operator
    if abs(w)-lam>0:
        return np.sign(w)*(abs(w)-lam)
    else:
        return 0

def lasso (X,y,lam):            #implement lasso algorithm
    n = X.shape[1]
    d = X.shape[0]
    w = np.zeros(d)
    tol = 10e-3                 #change tolerance for termination here
    delta_w = np.array([tol*10]*d)          #the change of w_j in one iteration
    sum = np.zeros(n)           #sum represent the sum of all rows of X multuplied by its corresponding w_j
    while(np.linalg.norm(delta_w,ord=np.inf)>=tol):         #using inf-norm of delta_w to terminate
        for j in range(d):
            sum -= X[j,:]*w[j]          #subtract current row from sum
            A = X[j,:]       #define A and B s.t. expression fits 1/2(Az+B)^2
            B = y.reshape(n)-sum
            w_temp = np.dot(A,B)/np.dot(A,A)        #divided by A^2 to fit in soft thresholding formula
            lam_temp = lam/np.dot(A,A)
            new_w = soft_thre(w_temp,lam_temp)          #compute new w_j using soft thresholding formula
            delta_w[j] = new_w-w[j]
            w[j] = new_w
            sum += X[j,:]*w[j]          #add new row to sum
    return w        #return w in a 1-D array

def compute_err(X,y,w):         #return mean square error
    n = X.shape[1]
    error = pow(np.linalg.norm(np.dot(X.T,w).reshape((n,1))-y),2)       #in case w is 1-D vector, must convert to 2-D
    error /= n
    return error

def plot_err(lam,perf,picname):         #plot validation error w.r.t. lambda
    plt.figure()
    plt.plot(lam,perf)
    plt.xlabel("lambda")
    plt.ylabel("mse on validation set")
    plt.savefig(picname + ".jpg")

def nonzeros(w):            #return ratio of nonzeros in w
    count = 0
    n = len(w)
    for item in w:
        if np.abs(item) > 0:
            count += 1
    return float(count)/n

def add_bias(X):            #add bias row
    n = X.shape[1]
    bias = np.ones(n)
    X = np.insert(X, -1, values=bias, axis=0)
    return X

def add_gaussian(X,k):          #add k row of gaussian distribution data
    n = X.shape[1]
    x1 = np.random.standard_normal(size=(k,n))
    X = np.insert(X, -1, values=x1, axis=0)
    return X

def slice(n,fold):        #it wll return 2-D array, where partition[i][0] and *[1] represent  the start and end of each part
    sec = int(np.ceil(float(n)/fold))
    partition = np.zeros((fold,2))           #each fold save its starting and ending point
    for i in range(fold):
        if i != fold-1:
            partition[i][0] = i*sec
            partition[i][1] = (i+1)*sec
        else:
            partition[i][0] = i*sec
            partition[i][1] = n
    return partition

def cross_valid(X,y,fold,method,lam):       #implement cross-validation, return average square error over all folds
    d = X.shape[0]
    n = X.shape[1]
    permutation = np.random.permutation(n)  # permute data set randomly
    X = X[:, permutation]
    y = y[permutation]
    partition = slice(n,fold).astype(int)
    perf_valid = 0
    for j in range(fold):       #use each part as validation set
        scale = n-(partition[j][1]-partition[j][0])        #the scale of training set

        X_train = np.zeros((d,scale))
        X_train[:,:partition[j][0]] = X[:,:partition[j][0]]
        X_train[:,partition[j][0]:] = X[:,partition[j][1]:]

        y_train = np.zeros((scale,1))
        y_train[:partition[j][0]] = y[:partition[j][0]]
        y_train[partition[j][0]:] = y[partition[j][1]:]

        X_valid = X[:,partition[j][0]:partition[j][1]]
        y_valid = y[partition[j][0]:partition[j][1]]

        w = method(X_train,y_train,lam)           #training w using training set
        perf_valid += compute_err(X_valid,y_valid,w)
    perf_valid /= fold
    return perf_valid

def performance(X,y,fold,lam,X_test,y_test,method):         #return all kind of performance, together with w using whole trainging set
    k = len(lam)
    perf_train = np.zeros(k)
    perf_valid = np.zeros(k)
    perf_test = np.zeros(k)
    w = np.zeros((X.shape[0],k))

    for i in range(k):  # iterate for each lambda
        w[:,i] = method(X,y,lam[i]).reshape(X.shape[0])
        perf_train[i] = compute_err(X,y,w[:,i])
        perf_valid[i] = cross_valid(X,y,fold,method,lam[i])
        perf_test[i] = compute_err(X_test,y_test,w[:,i])
        print ("lambda = ", lam[i], "train set error", perf_train[i], "validation set error", perf_valid[i], "test set error",perf_test[i],"w",np.linalg.norm(w))

    return perf_train,perf_valid,perf_test,w        #perf return in 1-D array, w in 2-D array with each column corespond to a lambda