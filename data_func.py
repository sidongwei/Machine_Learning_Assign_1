import numpy as np
import matplotlib.pyplot as plt
import csv

def perceptron (X,y,w,b,max_pass,update=0,permute=0):      #if update==1 then update each loop, if permute==1 then permute each pass
    n = len(y)
    mistake = [0]*max_pass
    if update == 0 and permute == 0:     #only make judgement once to save time
        for i in range(max_pass):
            for j in range(n):
                if y[j]*(np.dot(X[:,j],w)+b) <= 0:      #update only when making a mistake
                    w = w+y[j]*X[:,j]       #each row of X correspond to a single data
                    b = b+y[j]
                    mistake[i]+=1
    elif update == 1 and permute == 0:
        for i in range(max_pass):
            for j in range(n):       #update each loop
                if y[j]*(np.dot(X[:,j],w)+b) <= 0:      #only calcule number of mistakes
                    mistake[i]+=1
                w = w+y[j]*X[:,j]
                b = b+y[j]
    elif update == 0 and permute == 1:
        for i in range(max_pass):       #permute X and y each pass
            permutation = np.random.permutation(X.shape[1])  # create randomized indices
            X_s = X[:,permutation]
            y_s = y[permutation]
            for j in range(n):
                if y_s[j]*(np.dot(X_s[:,j],w)+b) <= 0:
                    w = w+y_s[j]*X_s[:,j]
                    b = b+y_s[j]
                    mistake[i]+=1
    return (w,b,mistake)

def plot(max_pass,mistake,picname):     #plot the figure of mistakes and name it with "picname.jpg"
    x = [i+1 for i in range(max_pass)]
    y = mistake
    plt.figure()
    plt.plot(x,y)
    plt.xlabel("time of passes")
    plt.ylabel("number of mistakes")
    plt.savefig(picname+".jpg")

def import_data(filename):
    CSV = open(filename,"r")       #read the X data from x file
    reader = csv.reader(CSV)
    rows = [row for row in reader]
    X = np.array(rows,dtype="float")     #transform data to float
    CSV.close()
    return X

def export_data(header,data,filename):
    CSV = open(filename+".csv","w")
    writer = csv.writer(CSV)
    writer.writerow(header)
    for row in data:
        writer.writerow(row)
    CSV.close()