import numpy as np
from data_func import import_data,perceptron,plot

X = import_data("spambase_X.csv")
y = import_data("spambase_y.csv")
w = np.zeros(X.shape[0])       #initialize w and b as zero
b = 0
max_pass = 500      #set the number of max passes here
(w,b,mistake) = perceptron(X,y,w,b,max_pass,update = 1)      #using update version of the algorithm
plot(max_pass,mistake,"Q2")        #plot the figure with name "Q2.jpg"
#print mistake[-1]