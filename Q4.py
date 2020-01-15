import numpy as np
from data_func import import_data,perceptron,plot

X = import_data("spambase_X.csv")
y = import_data("spambase_y.csv")
max_pass = 500      #set the number of max passes here
for t in range(5):      #try for 5 times
    permutation = np.random.permutation(X.shape[1])     #create randomized indices
    X_s = X[:,permutation]
    y_s = y[permutation]
    w = np.zeros(X.shape[0])  # initialize w and b each time
    b = 0
    (w,b,mistake) = perceptron(X_s,y_s,w,b,max_pass)
    plot(max_pass,mistake,"Q4_"+str(t+1))        #plot 5 figures with name "Q4_t+1.jpg"
    #print mistake[-1]